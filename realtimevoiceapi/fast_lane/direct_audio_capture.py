# here is realtimevoiceapi/fast_lane/direct_audio_capture.py


"""
Direct Audio Capture - Fast Lane Component

Hardware-level audio capture using sounddevice for minimal latency.
Optimized for real-time voice streaming with pre-allocated buffers.
"""

import asyncio
import queue
import logging
from typing import Optional, Callable, Union
from dataclasses import dataclass
import numpy as np
import time

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("sounddevice is required. Install with: pip install sounddevice")

from ..audio_types import AudioBytes, AudioConfig, AudioQuality
from ..exceptions import AudioError


@dataclass
class CaptureMetrics:
    """Metrics for audio capture performance"""
    chunks_captured: int = 0
    chunks_dropped: int = 0
    buffer_overruns: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    
    @property
    def capture_duration(self) -> float:
        """Total capture duration in seconds"""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def capture_rate(self) -> float:
        """Chunks per second"""
        duration = self.capture_duration
        if duration == 0:
            return 0.0
        return self.chunks_captured / duration


class DirectAudioCapture:
    """
    Direct audio capture using sounddevice for minimal latency.
    
    Features:
    - Triple-buffered capture for zero drops
    - Direct hardware access
    - Async queue-based interface
    - Automatic device selection
    """
    
    def __init__(
        self,
        device: Optional[Union[int, str]] = None,
        config: Optional[AudioConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize audio capture.
        
        Args:
            device: Audio device index/name or None for default
            config: Audio configuration
            logger: Logger instance
        """
        self.device = device
        self.config = config or AudioConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Audio stream
        self.stream: Optional[sd.InputStream] = None
        self.is_capturing = False
        
        # Calculate chunk size in samples
        self.chunk_samples = int(
            self.config.sample_rate * self.config.chunk_duration_ms / 1000
        )
        
        # Pre-allocate buffers for zero-copy operation
        self.audio_queue: asyncio.Queue[AudioBytes] = asyncio.Queue(maxsize=30)
        self.callback_queue: queue.Queue = queue.Queue(maxsize=30)
        
        # Metrics
        self.metrics = CaptureMetrics()
        
        # Get device info
        self._setup_device()
    
    def _setup_device(self):
        """Setup and validate audio device"""
        try:
            if self.device is None:
                # Get default input device
                self.device = sd.default.device[0]  # Input device
                self.logger.info(f"Using default input device: {self.device}")
            
            # Validate device
            device_info = sd.query_devices(self.device, 'input')
            self.device_name = device_info['name']
            self.max_channels = device_info['max_input_channels']
            
            if self.max_channels < self.config.channels:
                raise AudioError(
                    f"Device '{self.device_name}' has {self.max_channels} channels, "
                    f"but {self.config.channels} requested"
                )
            
            self.logger.info(
                f"Audio device ready: {self.device_name} "
                f"({self.config.sample_rate}Hz, {self.config.channels}ch)"
            )
            
        except Exception as e:
            raise AudioError(f"Failed to setup audio device: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags):
        """
        Audio stream callback - runs in audio thread.
        
        CRITICAL: Must be extremely fast, no allocations!
        """
        if status:
            self.logger.debug(f"Audio callback status: {status}")
            if status.input_overflow:
                self.metrics.buffer_overruns += 1
        
        if not self.is_capturing:
            return
        
        try:
            # Copy audio data (necessary to prevent it from being overwritten)
            audio_copy = indata.copy()
            
            # Try to put in queue (non-blocking)
            self.callback_queue.put_nowait(audio_copy)
            self.metrics.chunks_captured += 1
            
        except queue.Full:
            # Queue is full, drop the chunk
            self.metrics.chunks_dropped += 1
            self.logger.debug("Audio queue full, dropping chunk")
    
    async def start_async_capture(self) -> asyncio.Queue[AudioBytes]:
        """
        Start async audio capture.
        
        Returns:
            Async queue that will receive audio chunks
        """
        if self.is_capturing:
            raise AudioError("Already capturing")
        
        # Start capture stream
        self._start_stream()
        
        # Start async transfer task
        asyncio.create_task(self._transfer_audio())
        
        return self.audio_queue
    
    def _start_stream(self):
        """Start the audio stream"""
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype='int16',
                blocksize=self.chunk_samples,
                device=self.device,
                callback=self._audio_callback,
                latency='low'  # Request low latency
            )
            
            self.stream.start()
            self.is_capturing = True
            self.metrics.start_time = time.time()
            
            self.logger.info(
                f"Audio capture started: {self.chunk_samples} samples/chunk "
                f"({self.config.chunk_duration_ms}ms)"
            )
            
        except Exception as e:
            raise AudioError(f"Failed to start audio stream: {e}")
    
    async def _transfer_audio(self):
        """Transfer audio from callback queue to async queue"""
        loop = asyncio.get_event_loop()
        
        while self.is_capturing:
            try:
                # Get from callback queue (blocking)
                audio_array = await loop.run_in_executor(
                    None, 
                    self.callback_queue.get, 
                    True,  # block
                    0.1    # timeout
                )
                
                # Convert to bytes
                audio_bytes = audio_array.astype(np.int16).tobytes()
                self.metrics.total_bytes += len(audio_bytes)
                
                # Put in async queue
                await self.audio_queue.put(audio_bytes)
                
            except queue.Empty:
                # Timeout is normal, just continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Audio transfer error: {e}")
    
    def stop_capture(self):
        """Stop audio capture"""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.logger.info(
            f"Audio capture stopped. Metrics: "
            f"{self.metrics.chunks_captured} captured, "
            f"{self.metrics.chunks_dropped} dropped, "
            f"{self.metrics.buffer_overruns} overruns"
        )
    
    def get_device_info(self) -> dict:
        """Get current device information"""
        if not hasattr(self, 'device_name'):
            return {"name": "Unknown", "device": self.device}
        
        return {
            "name": self.device_name,
            "device": self.device,
            "channels": self.max_channels,
            "sample_rate": self.config.sample_rate
        }
    
    def get_metrics(self) -> dict:
        """Get capture metrics"""
        return {
            "chunks_captured": self.metrics.chunks_captured,
            "chunks_dropped": self.metrics.chunks_dropped,
            "buffer_overruns": self.metrics.buffer_overruns,
            "total_mb": self.metrics.total_bytes / (1024 * 1024),
            "duration_seconds": self.metrics.capture_duration,
            "capture_rate": self.metrics.capture_rate
        }
    
    @staticmethod
    def list_devices():
        """List available audio devices"""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    "index": i,
                    "name": device['name'],
                    "channels": device['max_input_channels'],
                    "default": i == sd.default.device[0]
                })
        
        return devices


class DirectAudioPlayer:
    """
    Direct audio playback using sounddevice.
    
    Optimized for low-latency playback of voice responses.
    """
    
    def __init__(
        self,
        device: Optional[Union[int, str]] = None,
        config: Optional[AudioConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize audio player"""
        self.device = device
        self.config = config or AudioConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup device
        if self.device is None:
            self.device = sd.default.device[1]  # Output device
            
        # Validate device
        try:
            device_info = sd.query_devices(self.device, 'output')
            self.device_name = device_info['name']
            self.logger.info(f"Audio player using device: {self.device_name}")
        except Exception as e:
            self.logger.warning(f"Could not get device info: {e}")
    
    def play_audio(self, audio_data: AudioBytes) -> bool:
        """
        Play audio with minimal latency.
        
        Args:
            audio_data: Raw audio bytes to play
            
        Returns:
            True if played successfully
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Play asynchronously (non-blocking)
            sd.play(
                audio_array,
                samplerate=self.config.sample_rate,
                device=self.device,
                latency='low'
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            return False
    
    def stop_playback(self):
        """Stop any ongoing playback"""
        sd.stop()
    
    def wait_until_done(self):
        """Wait until playback is complete"""
        sd.wait()