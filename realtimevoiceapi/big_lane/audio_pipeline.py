"""
Audio Pipeline - Big Lane Component

Composable audio processing pipeline with pluggable processors.
Supports chaining multiple audio operations for quality enhancement.
"""

import asyncio
import time
import logging
from typing import List, Optional, Dict, Any, Protocol, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from realtimevoiceapi.audio_types import AudioBytes, AudioConfig, AudioMetadata
from realtimevoiceapi.audio_processor import AudioProcessor as BaseAudioProcessor
from realtimevoiceapi.exceptions import AudioError


class ProcessorPriority(Enum):
    """Processing priority order"""
    CRITICAL = 0      # Must run first (e.g., format validation)
    HIGH = 10         # Important preprocessing (e.g., gain control)
    NORMAL = 50       # Standard processing
    LOW = 90          # Optional enhancement
    FINAL = 100       # Must run last


@dataclass
class ProcessorMetrics:
    """Metrics for a single processor"""
    name: str
    processed_chunks: int = 0
    processing_time_ms: float = 0.0
    errors: int = 0
    bytes_processed: int = 0
    
    @property
    def avg_time_per_chunk(self) -> float:
        if self.processed_chunks == 0:
            return 0.0
        return self.processing_time_ms / self.processed_chunks


class AudioProcessor(ABC):
    """
    Abstract base class for audio processors in the pipeline.
    
    Different from BaseAudioProcessor which is for core operations.
    This is for pipeline components.
    """
    
    def __init__(
        self, 
        name: str,
        priority: ProcessorPriority = ProcessorPriority.NORMAL,
        enabled: bool = True
    ):
        self.name = name
        self.priority = priority
        self.enabled = enabled
        self.metrics = ProcessorMetrics(name=name)
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def process(
        self, 
        audio: AudioBytes,
        metadata: Optional[AudioMetadata] = None
    ) -> Optional[AudioBytes]:
        """
        Process audio data.
        
        Returns:
            Processed audio or None to filter out
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset processor state"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            "name": self.name,
            "priority": self.priority.value,
            "enabled": self.enabled,
            "metrics": {
                "chunks": self.metrics.processed_chunks,
                "time_ms": self.metrics.processing_time_ms,
                "errors": self.metrics.errors
            }
        }


# ============== Core Audio Processors ==============

class AudioValidator(AudioProcessor):
    """Validates audio format and properties"""
    
    def __init__(self, config: AudioConfig):
        super().__init__("AudioValidator", ProcessorPriority.CRITICAL)
        self.config = config
        self.base_processor = BaseAudioProcessor(config)
    
    async def process(
        self, 
        audio: AudioBytes,
        metadata: Optional[AudioMetadata] = None
    ) -> Optional[AudioBytes]:
        """Validate audio format"""
        start_time = time.time()
        
        try:
            # Validate format
            valid, error = self.base_processor.validate_format(audio)
            if not valid:
                self.logger.warning(f"Invalid audio: {error}")
                self.metrics.errors += 1
                return None
            
            # Validate reasonable size
            if len(audio) > self.config.chunk_size_bytes(5000):  # 5 seconds max
                self.logger.warning("Audio chunk too large")
                self.metrics.errors += 1
                return None
            
            self.metrics.processed_chunks += 1
            self.metrics.bytes_processed += len(audio)
            
            return audio
            
        finally:
            self.metrics.processing_time_ms += (time.time() - start_time) * 1000
    
    def reset(self) -> None:
        """No state to reset"""
        pass


class NoiseReducer(AudioProcessor):
    """Simple noise reduction processor"""
    
    def __init__(self, noise_floor: float = 0.02):
        super().__init__("NoiseReducer", ProcessorPriority.HIGH)
        self.noise_floor = noise_floor
        self.noise_profile = None
        self.calibration_samples = 0
        
    async def process(
        self, 
        audio: AudioBytes,
        metadata: Optional[AudioMetadata] = None
    ) -> Optional[AudioBytes]:
        """Apply noise reduction"""
        start_time = time.time()
        
        try:
            # Convert to numpy
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0  # Normalize to [-1, 1]
            
            # Simple spectral subtraction approach
            if self.calibration_samples < 5:
                # Calibration phase - learn noise profile
                self._update_noise_profile(samples)
                self.calibration_samples += 1
                
            # Apply noise reduction
            if self.noise_profile is not None:
                # Simple gate - more sophisticated methods would use FFT
                mask = np.abs(samples) > self.noise_floor
                samples = samples * mask
            
            # Convert back
            samples = np.clip(samples * 32768, -32768, 32767)
            processed = samples.astype(np.int16).tobytes()
            
            self.metrics.processed_chunks += 1
            self.metrics.bytes_processed += len(audio)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Noise reduction failed: {e}")
            self.metrics.errors += 1
            return audio  # Return original on error
            
        finally:
            self.metrics.processing_time_ms += (time.time() - start_time) * 1000
    
    def _update_noise_profile(self, samples: np.ndarray):
        """Update noise profile during calibration"""
        rms = np.sqrt(np.mean(samples ** 2))
        if self.noise_profile is None:
            self.noise_profile = rms
        else:
            # Moving average
            self.noise_profile = 0.9 * self.noise_profile + 0.1 * rms
        
        # Update noise floor
        self.noise_floor = max(0.01, self.noise_profile * 2)
    
    def reset(self) -> None:
        """Reset noise profile"""
        self.noise_profile = None
        self.calibration_samples = 0


class VolumeNormalizer(AudioProcessor):
    """Normalizes audio volume"""
    
    def __init__(
        self, 
        target_level: float = 0.3,
        max_gain: float = 3.0
    ):
        super().__init__("VolumeNormalizer", ProcessorPriority.NORMAL)
        self.target_level = target_level
        self.max_gain = max_gain
        self.current_gain = 1.0
        
    async def process(
        self, 
        audio: AudioBytes,
        metadata: Optional[AudioMetadata] = None
    ) -> Optional[AudioBytes]:
        """Normalize volume"""
        start_time = time.time()
        
        try:
            # Convert to numpy
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
            
            # Calculate current level
            rms = np.sqrt(np.mean(samples ** 2))
            
            if rms > 0.001:  # Not silence
                # Calculate desired gain
                desired_gain = self.target_level / rms
                
                # Limit gain
                desired_gain = min(desired_gain, self.max_gain)
                
                # Smooth gain changes
                self.current_gain = 0.9 * self.current_gain + 0.1 * desired_gain
                
                # Apply gain
                samples = samples * self.current_gain
            
            # Convert back with clipping
            samples = np.clip(samples * 32768, -32768, 32767)
            processed = samples.astype(np.int16).tobytes()
            
            self.metrics.processed_chunks += 1
            self.metrics.bytes_processed += len(audio)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Volume normalization failed: {e}")
            self.metrics.errors += 1
            return audio
            
        finally:
            self.metrics.processing_time_ms += (time.time() - start_time) * 1000
    
    def reset(self) -> None:
        """Reset gain"""
        self.current_gain = 1.0


class EchoCanceller(AudioProcessor):
    """Simple echo cancellation (demonstration only)"""
    
    def __init__(self, buffer_size: int = 4800):  # 100ms at 24kHz
        super().__init__("EchoCanceller", ProcessorPriority.HIGH)
        self.buffer_size = buffer_size
        self.reference_buffer = np.zeros(buffer_size, dtype=np.float32)
        
    async def process(
        self, 
        audio: AudioBytes,
        metadata: Optional[AudioMetadata] = None
    ) -> Optional[AudioBytes]:
        """Apply echo cancellation"""
        # This is a placeholder - real echo cancellation is complex
        # and requires reference signal from speaker output
        start_time = time.time()
        
        try:
            self.metrics.processed_chunks += 1
            self.metrics.bytes_processed += len(audio)
            
            # For now, just pass through
            # Real implementation would subtract echo estimate
            return audio
            
        finally:
            self.metrics.processing_time_ms += (time.time() - start_time) * 1000
    
    def reset(self) -> None:
        """Reset buffers"""
        self.reference_buffer.fill(0)


class VADProcessor(AudioProcessor):
    """Voice Activity Detection processor for filtering"""
    
    def __init__(
        self, 
        threshold: float = 0.02,
        min_speech_duration_ms: int = 200
    ):
        super().__init__("VADProcessor", ProcessorPriority.NORMAL)
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_duration_ms * 24)  # 24kHz
        self.is_speech = False
        self.speech_buffer = []
        self.silence_samples = 0
        self.accumulated_samples = 0  # Track total accumulated
        
    
    async def process(
        self, 
        audio: AudioBytes,
        metadata: Optional[AudioMetadata] = None
    ) -> Optional[AudioBytes]:
        """Filter out non-speech audio"""
        start_time = time.time()
        
        try:
            # Quick energy check
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
            
            energy = np.sqrt(np.mean(samples ** 2))
            
            if energy > self.threshold:
                # Speech detected
                self.is_speech = True
                self.silence_samples = 0
                self.speech_buffer.append(audio)
                self.accumulated_samples += len(audio) // 2  # 2 bytes per sample
                
                # Check if we have enough speech accumulated
                if self.accumulated_samples >= self.min_speech_samples:
                    # Return accumulated speech
                    result = b''.join(self.speech_buffer)
                    self.speech_buffer = []
                    self.accumulated_samples = 0
                    
                    self.metrics.processed_chunks += 1
                    self.metrics.bytes_processed += len(result)
                    
                    return result
                else:
                    # Still accumulating
                    return None
            else:
                # Silence detected
                self.silence_samples += len(samples)
                
                if self.is_speech and self.silence_samples > 12000:  # 500ms silence
                    # End of speech
                    self.is_speech = False
                    
                    if self.speech_buffer:
                        result = b''.join(self.speech_buffer)
                        self.speech_buffer = []
                        self.accumulated_samples = 0
                        
                        self.metrics.processed_chunks += 1
                        self.metrics.bytes_processed += len(result)
                        
                        return result
                
                return None  # Filter out silence
                
        except Exception as e:
            self.logger.error(f"VAD processing failed: {e}")
            self.metrics.errors += 1
            return audio  # Return original on error
            
        finally:
            self.metrics.processing_time_ms += (time.time() - start_time) * 1000

    def reset(self) -> None:
        """Reset VAD state"""
        self.is_speech = False
        self.speech_buffer = []
        self.silence_samples = 0
        self.accumulated_samples = 0

# ============== Audio Pipeline ==============

class AudioPipeline:
    """
    Composable audio processing pipeline.
    
    Chains multiple processors together for complex audio processing.
    """
    
    def __init__(
        self,
        config: AudioConfig,
        processors: Optional[List[AudioProcessor]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.processors: List[AudioProcessor] = processors or []
        
        # Sort by priority
        self._sort_processors()
        
        # Metrics
        self.total_chunks_processed = 0
        self.total_bytes_processed = 0
        self.total_processing_time_ms = 0.0
        
        self.logger.info(
            f"Audio pipeline initialized with {len(self.processors)} processors"
        )
    
    def _sort_processors(self):
        """Sort processors by priority"""
        self.processors.sort(key=lambda p: p.priority.value)
    
    def add_processor(self, processor: AudioProcessor):
        """Add processor to pipeline"""
        self.processors.append(processor)
        self._sort_processors()
        self.logger.info(f"Added processor: {processor.name}")
    
    def remove_processor(self, name: str) -> bool:
        """Remove processor by name"""
        for i, proc in enumerate(self.processors):
            if proc.name == name:
                self.processors.pop(i)
                self.logger.info(f"Removed processor: {name}")
                return True
        return False
    
    def set_processor_enabled(self, name: str, enabled: bool):
        """Enable/disable processor"""
        for proc in self.processors:
            if proc.name == name:
                proc.enabled = enabled
                self.logger.info(f"Processor {name} {'enabled' if enabled else 'disabled'}")
                break
    
    

    async def process(
        self,
        audio: AudioBytes,
        metadata: Optional[AudioMetadata] = None
    ) -> Optional[AudioBytes]:
        """
        Process audio through the pipeline.
        
        Returns:
            Processed audio or None if filtered out
        """
        start_time = time.time()
        current_audio = audio
        current_metadata = metadata
        
        # Update metrics at the start (count all attempts)
        self.total_chunks_processed += 1
        self.total_bytes_processed += len(audio)
        
        # Process through each enabled processor
        for processor in self.processors:
            if not processor.enabled:
                continue
            
            try:
                current_audio = await processor.process(current_audio, current_metadata)
                
                # If any processor returns None, stop pipeline
                if current_audio is None:
                    self.logger.debug(f"Audio filtered out by {processor.name}")
                    # Update timing even for filtered audio
                    self.total_processing_time_ms += (time.time() - start_time) * 1000
                    return None
                    
            except Exception as e:
                self.logger.error(f"Processor {processor.name} failed: {e}")
                # Continue with unprocessed audio
        
        # Update timing
        self.total_processing_time_ms += (time.time() - start_time) * 1000
        
        return current_audio
    
    def reset(self):
        """Reset all processors"""
        for processor in self.processors:
            try:
                processor.reset()
            except Exception as e:
                self.logger.error(f"Failed to reset {processor.name}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        processor_metrics = {}
        
        for proc in self.processors:
            processor_metrics[proc.name] = {
                "enabled": proc.enabled,
                "priority": proc.priority.value,
                "chunks": proc.metrics.processed_chunks,
                "time_ms": proc.metrics.processing_time_ms,
                "avg_time": proc.metrics.avg_time_per_chunk,
                "errors": proc.metrics.errors
            }
        
        return {
            "total_chunks": self.total_chunks_processed,
            "total_bytes": self.total_bytes_processed,
            "total_time_ms": self.total_processing_time_ms,
            "avg_time_per_chunk": (
                self.total_processing_time_ms / self.total_chunks_processed
                if self.total_chunks_processed > 0 else 0
            ),
            "processors": processor_metrics
        }
    
    def get_processor_chain(self) -> List[str]:
        """Get ordered list of enabled processors"""
        return [
            f"{p.name} (P:{p.priority.value})"
            for p in self.processors
            if p.enabled
        ]


# ============== Pipeline Presets ==============

class PipelinePresets:
    """Pre-configured pipelines for common use cases"""
    
    @staticmethod
    def create_basic_pipeline(config: AudioConfig) -> AudioPipeline:
        """Basic pipeline with validation only"""
        return AudioPipeline(
            config=config,
            processors=[
                AudioValidator(config)
            ]
        )
    
    @staticmethod
    def create_voice_pipeline(config: AudioConfig) -> AudioPipeline:
        """Pipeline optimized for voice"""
        return AudioPipeline(
            config=config,
            processors=[
                AudioValidator(config),
                NoiseReducer(noise_floor=0.02),
                VolumeNormalizer(target_level=0.3),
                VADProcessor(threshold=0.02)
            ]
        )
    
    @staticmethod
    def create_quality_pipeline(config: AudioConfig) -> AudioPipeline:
        """High quality pipeline with all enhancements"""
        return AudioPipeline(
            config=config,
            processors=[
                AudioValidator(config),
                NoiseReducer(noise_floor=0.01),
                EchoCanceller(),
                VolumeNormalizer(target_level=0.3),
                VADProcessor(threshold=0.015)
            ]
        )
    
    @staticmethod
    def create_realtime_pipeline(config: AudioConfig) -> AudioPipeline:
        """Minimal pipeline for real-time processing"""
        return AudioPipeline(
            config=config,
            processors=[
                AudioValidator(config),
                VolumeNormalizer(target_level=0.3)  # Fast processing only
            ]
        )