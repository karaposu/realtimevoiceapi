# here is realtimevoiceapi/fast_lane/fast_vad_detector.py


"""
Fast VAD Detector - Fast Lane

Lightweight Voice Activity Detection with minimal overhead.
Optimized for real-time performance with no allocations in hot path.

No allocations in hot path: All buffers pre-allocated
Pre-computed thresholds: Avoids calculations during processing
Simple state machine: Fast state transitions
Energy-based detection: Most efficient method
Optional adaptive mode: Can adjust to noise levels

are designed to work together in the fast lane with minimal latency, typically achieving <10ms detection latency on modern hardware.
"""

import numpy as np
import time
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .audio_types import AudioBytes, AudioConfig, VADConfig, VADType
import collections


class VADState(Enum):
    """VAD state machine states"""
    SILENCE = "silence"
    SPEECH_STARTING = "speech_starting"
    SPEECH = "speech"
    SPEECH_ENDING = "speech_ending"

@dataclass
class VADMetrics:
    """Metrics for VAD performance"""
    total_chunks: int = 0
    speech_chunks: int = 0
    silence_chunks: int = 0
    speech_starting_chunks: int = 0
    speech_ending_chunks: int = 0
    time_in_speech_ms: float = 0.0
    time_in_silence_ms: float = 0.0
    last_state_change: float = 0.0
    state_changes: int = 0
    transitions: int= 0
    speech_segments: int= 0
    total_speech_ms: float = 0.0  # Changed to float for consistency
    total_silence_ms: float = 0.0  # Changed to float for consistency



   


class FastVADDetector:
    """
    Fast Voice Activity Detection for real-time audio.
    
    Optimized for minimal latency - no allocations in hot path.
    """
    
    def __init__(
        self,
        config: Optional[VADConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None
    ):
        """
        Initialize VAD detector.
        
        Args:
            config: VAD configuration
            audio_config: Audio format configuration
            on_speech_start: Callback when speech starts (runs in hot path!)
            on_speech_end: Callback when speech ends (runs in hot path!)
        """
        self.config = config or VADConfig()
        self.audio_config = audio_config or AudioConfig()
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        
        # State machine
        self.state = VADState.SILENCE
        self.state_duration_ms = 0.0
        self.last_update_time = time.time()
        
        # Pre-computed values to avoid calculations in hot path
        self.samples_per_chunk = self.audio_config.chunk_size_bytes(
            self.audio_config.chunk_duration_ms
        ) // 2  # /2 for int16
        
        self.energy_threshold_squared = (
            self.config.energy_threshold * 32768
        ) ** 2  # Pre-square for faster comparison
        
        # Calculate bytes per millisecond
        self._bytes_per_ms = self.audio_config.bytes_per_second / 1000.0
        
        # Ring buffer for smoothing (pre-allocated)
        self.energy_history_size = 5
        self.energy_history = np.zeros(self.energy_history_size, dtype=np.float32)
        self.energy_history_pos = 0
        
        # Metrics
        self.metrics = VADMetrics()
        self.chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.transitions = 0
        self.speech_segments = 0

        self.speech_chunks = 0
        self.silence_chunks = 0


      
        # Pre-allocated work buffer
        self.work_buffer = np.zeros(self.samples_per_chunk, dtype=np.float32)

        # TO THIS:
        self.pre_buffer = collections.deque(maxlen=max(1, int(self.config.pre_buffer_ms / 20)))  # Assuming 20ms chunks
        self.pre_buffer_bytes = bytearray()
        self.max_prebuffer_size = int(self.audio_config.sample_rate * self.config.pre_buffer_ms / 1000 * 2)  # 16-bit samples
    
    def get_metrics(self):
        """Get VAD metrics as dictionary"""
        return {
            'total_chunks': self.metrics.total_chunks,
            'speech_chunks': self.metrics.speech_chunks,
            'silence_chunks': self.metrics.silence_chunks,
            'speech_starting_chunks': self.metrics.speech_starting_chunks,
            'speech_ending_chunks': self.metrics.speech_ending_chunks,
            'current_state': self.state.value,
            'time_in_speech_ms': self.metrics.time_in_speech_ms,
            'time_in_silence_ms': self.metrics.time_in_silence_ms,
            'last_state_change': self.metrics.last_state_change,
            'state_changes': self.metrics.state_changes,
            'transitions': self.metrics.transitions,
            'speech_segments': self.metrics.speech_segments,
            'total_speech_ms': self.metrics.total_speech_ms,
            'total_silence_ms': self.metrics.total_silence_ms,
            'chunks_processed': self.chunks_processed,
            'avg_processing_time_ms': self.total_processing_time_ms / self.chunks_processed if self.chunks_processed > 0 else 0
        }

    def get_pre_buffer(self) -> Optional[AudioBytes]:
        """Get the pre-buffer containing recent audio before speech detection"""
        if self.config.pre_buffer_ms <= 0:
            return None
        
        # Return concatenated prebuffer
        if self.pre_buffer:
            return bytes(b''.join(self.pre_buffer))
        return None


    def _calculate_energy(self, audio_chunk: AudioBytes) -> float:
        """
        Calculate energy of audio chunk.
        
        Returns energy value (0.0 to 1.0 range).
        """
        # Convert to numpy view (no copy)
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0  # Normalize to [-1, 1]
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(samples ** 2))
        
        return energy
    
    

    def process_chunk(self, audio_chunk: AudioBytes) -> VADState:
        """Process audio chunk and return current state"""
        if len(audio_chunk) == 0:
                return self.state
        

        self.metrics.total_chunks += 1


        if self.config.pre_buffer_ms > 0:
             self.pre_buffer.append(audio_chunk)
        start_time = time.perf_counter()
        
        # Calculate energy
        energy = self._calculate_energy(audio_chunk)
        chunk_duration_ms = len(audio_chunk) / self._bytes_per_ms
        
        old_state = self.state  # Track old state for transitions
        
        if energy > self.config.energy_threshold:
            self.speech_chunks += 1
            # Speech detected
            if self.state == VADState.SILENCE:
                self.state = VADState.SPEECH_STARTING
                self.state_duration_ms = chunk_duration_ms
                self.transitions += 1
                self.metrics.transitions += 1
                self.metrics.state_changes += 1
                self.metrics.last_state_change = time.time()
            elif self.state == VADState.SPEECH_STARTING:
                self.state_duration_ms += chunk_duration_ms
                # Need enough speech to confirm
                if self.state_duration_ms >= self.config.speech_start_ms:
                    self.state = VADState.SPEECH
                    self.speech_segments += 1
                    self.metrics.speech_segments += 1
                    self.transitions += 1
                    self.metrics.transitions += 1
                    if self.on_speech_start:
                        self.on_speech_start()
            elif self.state == VADState.SPEECH:
                self.state_duration_ms += chunk_duration_ms
            elif self.state == VADState.SPEECH_ENDING:
                # Speech resumed
                self.state = VADState.SPEECH
                self.state_duration_ms = chunk_duration_ms
                self.transitions += 1
                self.metrics.transitions += 1
                self.metrics.state_changes += 1
                self.metrics.last_state_change = time.time()
        else:
            # Silence detected
            self.silence_chunks += 1
            if self.state in [VADState.SPEECH, VADState.SPEECH_STARTING]:
                self.state = VADState.SPEECH_ENDING
                self.state_duration_ms = chunk_duration_ms
                self.transitions += 1
                self.metrics.transitions += 1
                self.metrics.state_changes += 1
                self.metrics.last_state_change = time.time()
            elif self.state == VADState.SPEECH_ENDING:
                self.state_duration_ms += chunk_duration_ms
                # Need enough silence to confirm end
                if self.state_duration_ms >= self.config.speech_end_ms:
                    self.state = VADState.SILENCE
                    self.transitions += 1
                    self.metrics.transitions += 1
                    if self.on_speech_end:
                        self.on_speech_end()
            else:  # SILENCE
                self.state_duration_ms += chunk_duration_ms
        
        # Update metrics based on current state
        if self.state == VADState.SILENCE:
            self.metrics.silence_chunks += 1
            self.metrics.time_in_silence_ms += chunk_duration_ms
            self.metrics.total_silence_ms += chunk_duration_ms
        elif self.state == VADState.SPEECH:
            self.metrics.speech_chunks += 1
            self.metrics.time_in_speech_ms += chunk_duration_ms
            self.metrics.total_speech_ms += chunk_duration_ms
        elif self.state == VADState.SPEECH_STARTING:
            self.metrics.speech_starting_chunks += 1
            # Count as speech time
            self.metrics.time_in_speech_ms += chunk_duration_ms
            self.metrics.total_speech_ms += chunk_duration_ms
        elif self.state == VADState.SPEECH_ENDING:
            self.metrics.speech_ending_chunks += 1
            # Count as speech time until confirmed silence
            self.metrics.time_in_speech_ms += chunk_duration_ms
            self.metrics.total_speech_ms += chunk_duration_ms
        
        # Update processing metrics
        self.chunks_processed += 1
        processing_time = (time.perf_counter() - start_time) * 1000
        self.total_processing_time_ms += processing_time
        
        return self.state
    
    def reset(self):
        """Reset VAD state"""
        self.state = VADState.SILENCE
        self.state_duration_ms = 0
        self.energy_history.fill(0)
        self.energy_history_pos = 0
        self.last_update_time = time.time()
        
        # Reset all metrics
        self.chunks_processed = 0
        self.total_processing_time_ms = 0.0
        self.speech_chunks = 0
        self.silence_chunks = 0
        self.transitions = 0
        self.speech_segments = 0
        
        # Reset metrics dataclass
        self.metrics = VADMetrics()
        
        if hasattr(self, 'pre_buffer'):
            self.pre_buffer.clear()
    
    
    def old_process_chunk(self, audio_chunk: AudioBytes) -> VADState:
        """
        Process audio chunk and return current state.
        
        CRITICAL: This runs in the audio callback thread!
        Must be extremely fast with no allocations.
        """
        # Convert to numpy view (no copy)
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Calculate energy (RMS squared to avoid sqrt)
        # Use pre-allocated work buffer
        np.square(samples, out=self.work_buffer[:len(samples)], dtype=np.float32)
        energy_squared = np.mean(self.work_buffer[:len(samples)])
        
        # Update energy history (circular buffer)
        self.energy_history[self.energy_history_pos] = energy_squared
        self.energy_history_pos = (self.energy_history_pos + 1) % self.energy_history_size
        
        # Get smoothed energy (simple moving average)
        smoothed_energy = np.mean(self.energy_history)
        
        # Determine if this is speech
        is_speech = smoothed_energy > self.energy_threshold_squared
        
        # Simple zero-crossing rate if enabled
        if self.config.type == VADType.COMBINED:
            # Count zero crossings (fast method)
            signs = np.sign(samples[:-1]) != np.sign(samples[1:])
            zcr = np.sum(signs) / len(samples)
            
            # Combine with energy
            is_speech = is_speech and (zcr > self.config.zcr_threshold)
        
        # Update state machine
        new_state = self._update_state(is_speech)
        
        return new_state
    
    

    
    def _update_state(self, is_speech: bool) -> VADState:
        """
        Update VAD state machine.
        
        Runs in hot path - must be fast!
        """
        current_time = time.time()
        time_delta_ms = (current_time - self.last_update_time) * 1000
        self.last_update_time = current_time
        
        self.state_duration_ms += time_delta_ms
        
        old_state = self.state
        
        # State transitions
        if self.state == VADState.SILENCE:
            if is_speech:
                self.state = VADState.SPEECH_STARTING
                self.state_duration_ms = 0
                
        elif self.state == VADState.SPEECH_STARTING:
            if not is_speech:
                # False start
                self.state = VADState.SILENCE
                self.state_duration_ms = 0
            elif self.state_duration_ms >= self.config.speech_start_ms:
                # Confirmed speech
                self.state = VADState.SPEECH
                self.state_duration_ms = 0
                self.metrics.speech_segments += 1
                
                # Trigger callback
                if self.on_speech_start:
                    self.on_speech_start()
                    
        elif self.state == VADState.SPEECH:
            if not is_speech:
                self.state = VADState.SPEECH_ENDING
                self.state_duration_ms = 0
                
        elif self.state == VADState.SPEECH_ENDING:
            if is_speech:
                # Speech continues
                self.state = VADState.SPEECH
                self.state_duration_ms = 0
            elif self.state_duration_ms >= self.config.speech_end_ms:
                # Confirmed end
                self.state = VADState.SILENCE
                self.state_duration_ms = 0
                
                # Trigger callback
                if self.on_speech_end:
                    self.on_speech_end()
        
        # Update metrics
        if old_state != self.state:
            self.metrics.transitions += 1
            self.metrics.last_transition_time = current_time
            
            # Track durations
            if old_state == VADState.SPEECH:
                self.metrics.total_speech_ms += time_delta_ms
            else:
                self.metrics.total_silence_ms += time_delta_ms
        
        return self.state
    
    # def reset(self):
    #     """Reset VAD state"""
    #     self.state = VADState.SILENCE
    #     self.state_duration_ms = 0
    #     self.energy_history.fill(0)
    #     self.energy_history_pos = 0
    #     self.last_update_time = time.time()
    
    # def get_metrics(self) -> dict:
    #     """Get VAD metrics"""
    #     total_time = self.metrics.total_speech_ms + self.metrics.total_silence_ms
        
    #     return {
    #         'state': self.state.value,
    #         'speech_segments': self.metrics.speech_segments,
    #         'total_speech_ms': self.metrics.total_speech_ms,
    #         'total_silence_ms': self.metrics.total_silence_ms,
    #         'speech_ratio': self.metrics.total_speech_ms / total_time if total_time > 0 else 0,
    #         'transitions': self.metrics.transitions
    #     }


class AdaptiveVAD(FastVADDetector):
    """
    Adaptive VAD that adjusts thresholds based on noise level.
    
    Still fast but with adaptive capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive parameters
        self.noise_floor = 0.0
        self.noise_samples = 0
        self.adaptation_rate = 0.001
        
        # Pre-allocated for noise estimation
        self.noise_estimation_buffer = np.zeros(
            self.audio_config.sample_rate,  # 1 second
            dtype=np.float32
        )
        self.noise_buffer_pos = 0
        self.is_calibrating = True
    
    

    def process_chunk(self, audio_chunk: AudioBytes) -> VADState:
        """Process with adaptive threshold"""
        
        # During calibration, just collect noise samples
        if self.is_calibrating:
            self._update_noise_floor(audio_chunk)
            
            # Calibrate for first second
            if self.noise_samples * self.audio_config.chunk_duration_ms >= 1000:
                self.is_calibrating = False
                # Set threshold above noise floor
                self.config.energy_threshold = max(
                    0.02,  # Minimum threshold
                    self.noise_floor * 3  # 3x noise floor
                )
                self.energy_threshold_squared = (
                    self.config.energy_threshold * 32768
                ) ** 2
        
        # Normal processing - use parent's process_chunk
        state = super().process_chunk(audio_chunk)
        
        # Adapt threshold during silence
        if state == VADState.SILENCE and not self.is_calibrating:
            self._adapt_threshold(audio_chunk)

        return state
        
    
    def _update_noise_floor(self, audio_chunk: AudioBytes):
        """Update noise floor estimate"""
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0  # Normalize
        
        # RMS energy
        energy = np.sqrt(np.mean(samples ** 2))
        
        # Update running average
        self.noise_samples += 1
        self.noise_floor += (energy - self.noise_floor) / self.noise_samples
    
    def _adapt_threshold(self, audio_chunk: AudioBytes):
        """Slowly adapt threshold during silence"""
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
        
        energy = np.sqrt(np.mean(samples ** 2))
        
        # Slowly track noise floor
        if energy < self.noise_floor:
            self.noise_floor -= self.adaptation_rate * (self.noise_floor - energy)
        else:
            self.noise_floor += self.adaptation_rate * (energy - self.noise_floor)
        
        # Update threshold
        new_threshold = max(0.02, self.noise_floor * 3)
        self.config.energy_threshold = (
            0.9 * self.config.energy_threshold + 
            0.1 * new_threshold
        )
        self.energy_threshold_squared = (self.config.energy_threshold * 32768) ** 2

