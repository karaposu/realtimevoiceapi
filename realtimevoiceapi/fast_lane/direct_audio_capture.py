#here is realtimevoiceapi/fast_lane/direct_audio_capture.py


"""
Direct Audio Capture - Fast Lane Component

Hardware-level audio capture using sounddevice for minimal latency.
Optimized for real-time voice streaming with pre-allocated buffers.
"""
"""
Direct Audio Capture - Fast Lane Component

Hardware-level audio capture using sounddevice for minimal latency.
Optimized for real-time voice streaming with pre-allocated buffers.
"""

import asyncio
import queue
import logging
import threading
import time
from typing import Optional, Callable, Union, Dict, Any
from dataclasses import dataclass
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    raise ImportError("sounddevice is required. Install with: pip install sounddevice")

from ..core.audio_types import AudioBytes, AudioConfig, AudioQuality
from ..core.exceptions import AudioError
from ..core.audio_interfaces import AudioPlayerInterface, AudioCaptureInterface


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


class DirectAudioCapture(AudioCaptureInterface):
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
        
        # Device info
        self.device_info = {}
        
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
            
            # Store device info
            self.device_info = {
                'name': device_info['name'],
                'channels': device_info['max_input_channels'],
                'sample_rate': device_info.get('default_samplerate', self.config.sample_rate),
                'index': self.device
            }
            
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
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get current device information"""
        return self.device_info.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
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


class DirectAudioPlayer(AudioPlayerInterface):
    """
    Direct audio playback using sounddevice.
    
    Optimized for ultra-low latency real-time voice.
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
        
        # Metrics tracking
        self._chunks_played = 0
        self._total_bytes_played = 0
        self._play_start_time: Optional[float] = None
        self._last_play_time: Optional[float] = None
        self.device_info = {}
        
        # Stream management
        self._stream: Optional[sd.OutputStream] = None
        self._is_active = False
        self._stream_lock = threading.Lock()
        
        # Small ring buffer for timing issues
        self._buffer = bytearray()
        self._buffer_lock = threading.Lock()
        
        # Setup device
        self._setup_device()
        
        # Pre-initialize stream for zero-latency start
        self._initialize_stream()
    
    def _setup_device(self):
        """Setup and validate audio device"""
        try:
            if self.device is None:
                self.device = sd.default.device[1]  # Output device
                
            # Get device info
            device_info = sd.query_devices(self.device, 'output')
            self.device_info = {
                'name': device_info['name'],
                'channels': device_info['max_output_channels'],
                'sample_rate': device_info.get('default_samplerate', self.config.sample_rate),
                'index': self.device
            }
            self.logger.info(f"Audio player using device: {self.device_info['name']}")
        except Exception as e:
            self.logger.warning(f"Could not get device info: {e}")
            self.device_info = {'name': 'Unknown', 'index': self.device}
    
    def _initialize_stream(self):
        """Pre-initialize the audio stream"""
        try:
            with self._stream_lock:
                if self._stream is None:
                    self._stream = sd.OutputStream(
                        samplerate=self.config.sample_rate,
                        channels=self.config.channels,
                        dtype='int16',
                        device=self.device,
                        blocksize=int(self.config.sample_rate * 0.01),  # 10ms blocks
                        latency='low',
                        callback=self._audio_callback
                    )
                    self._stream.start()
                    self._is_active = True
                    self.logger.debug("Audio stream pre-initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize stream: {e}")
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Stream callback - runs in audio thread.
        Pulls data from buffer with minimal latency.
        """
        if status:
            self.logger.debug(f"Stream status: {status}")
        
        with self._buffer_lock:
            # Calculate bytes needed
            bytes_needed = frames * self.config.channels * 2  # 16-bit = 2 bytes
            
            if len(self._buffer) >= bytes_needed:
                # We have enough data
                data = self._buffer[:bytes_needed]
                self._buffer = self._buffer[bytes_needed:]
                
                # Convert to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Reshape if multi-channel
                if self.config.channels > 1:
                    audio_array = audio_array.reshape(-1, self.config.channels)
                
                # Copy to output
                outdata[:] = audio_array.reshape(outdata.shape)
            else:
                # Not enough data - output silence
                outdata.fill(0)
    
    @property
    def is_playing(self) -> bool:
        """Check if currently playing audio"""
        with self._buffer_lock:
            return len(self._buffer) > 0 or self._is_active
    
    def play_audio(self, audio_data: AudioBytes) -> bool:
        """
        Play audio with minimal latency.
        
        Args:
            audio_data: Raw audio bytes to play
            
        Returns:
            True if played successfully
        """
        try:
            # Ensure stream is ready
            if self._stream is None:
                self._initialize_stream()
            
            # Add to buffer
            with self._buffer_lock:
                self._buffer.extend(audio_data)
            
            # Update metrics
            self._chunks_played += 1
            self._total_bytes_played += len(audio_data)
            
            current_time = time.time()
            if self._play_start_time is None:
                self._play_start_time = current_time
            self._last_play_time = current_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            return False
    
    def stop_playback(self):
        """Stop any ongoing playback"""
        with self._buffer_lock:
            self._buffer.clear()
        
        # Keep stream open for next audio
        # This maintains low latency for next playback
    
    def wait_until_done(self):
        """Wait until playback is complete"""
        while True:
            with self._buffer_lock:
                if len(self._buffer) == 0:
                    break
            time.sleep(0.01)  # 10ms check interval
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get player metrics"""
        with self._buffer_lock:
            buffer_ms = (len(self._buffer) / 2 / self.config.sample_rate) * 1000 if self._buffer else 0
        
        total_duration = 0.0
        if self._play_start_time and self._chunks_played > 0:
            if self.is_playing and self._last_play_time:
                total_duration = self._last_play_time - self._play_start_time
            else:
                total_samples = self._total_bytes_played / 2
                total_duration = total_samples / self.config.sample_rate
        
        return {
            "chunks_played": self._chunks_played,
            "total_mb_played": self._total_bytes_played / (1024 * 1024),
            "total_duration_seconds": total_duration,
            "is_playing": self.is_playing,
            "buffer_ms": buffer_ms,
            "device": self.device_info.get('name', 'Unknown'),
            "sample_rate": self.config.sample_rate
        }
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return self.device_info.copy()
    
    def cleanup(self):
        """Full cleanup (only on shutdown)"""
        with self._stream_lock:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._is_active = False
        
        with self._buffer_lock:
            self._buffer.clear()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except Exception:
            pass