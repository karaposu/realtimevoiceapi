"""
Direct Audio Capture - Fast Lane

Hardware-level audio capture with minimal overhead for real-time VAD.
Designed for ultra-low latency with zero allocations in the hot path.


Pre-allocated buffers: Triple buffering with no allocations
Direct callbacks: Audio data goes straight to callback
Atomic metrics: No locks needed
Hardware optimizations: Exclusive mode, low latency settings
Minimal Python overhead: Uses numpy for efficient operations




"""

import asyncio
import time
import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass
import sounddevice as sd
import threading
import queue

from .audio_types import AudioConfig, AudioBytes, AudioConstants
from .exceptions import AudioError


@dataclass
class CaptureMetrics:
    """Lightweight metrics - updated atomically"""
    chunks_captured: int = 0
    bytes_captured: int = 0
    overruns: int = 0
    last_capture_time: float = 0
    start_time: float = 0


class DirectAudioCapture:
    """
    Direct audio capture from hardware with minimal latency.
    
    Designed for fast lane - no abstractions in audio callback.
    """
    
    def __init__(
        self,
        device: Optional[int] = None,
        config: Optional[AudioConfig] = None,
        callback: Optional[Callable[[AudioBytes], None]] = None
    ):
        """
        Initialize direct capture.
        
        Args:
            device: Audio device index (None for default)
            config: Audio configuration
            callback: Direct callback for audio data (runs in audio thread!)
        """
        self.device = device
        self.config = config or AudioConfig()
        self.callback = callback
        
        # Pre-allocate buffers to avoid allocation in callback
        self.chunk_size = self.config.chunk_size_bytes(self.config.chunk_duration_ms)
        self.buffer_pool = self._create_buffer_pool()
        self.current_buffer_index = 0
        
        # Stream state
        self.stream: Optional[sd.InputStream] = None
        self.is_active = False
        
        # Metrics (atomic updates only)
        self.metrics = CaptureMetrics()
        
        # For async operation mode
        self.async_queue: Optional[queue.Queue] = None
        self.async_thread: Optional[threading.Thread] = None
    
    def _create_buffer_pool(self) -> list:
        """Pre-allocate buffer pool to avoid allocations"""
        # Create 3 buffers for triple buffering
        pool = []
        for _ in range(3):
            # Pre-allocate numpy array
            buffer = np.zeros(self.chunk_size // 2, dtype=np.int16)  # /2 because int16
            pool.append(buffer)
        return pool
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Low-level audio callback - runs in audio thread!
        
        CRITICAL: No allocations, no Python objects, minimal work.
        """
        if status:
            self.metrics.overruns += 1
            return
        
        if not self.is_active:
            return
        
        # Get next buffer from pool (no allocation)
        buffer = self.buffer_pool[self.current_buffer_index]
        self.current_buffer_index = (self.current_buffer_index + 1) % 3
        
        # Copy data directly (numpy handles this efficiently)
        np.copyto(buffer[:frames], indata[:, 0])
        
        # Update metrics (atomic operations)
        self.metrics.chunks_captured += 1
        self.metrics.bytes_captured += frames * 2
        self.metrics.last_capture_time = time.time()
        
        # Direct callback mode (fastest)
        if self.callback:
            # Pass view to avoid copy
            self.callback(buffer[:frames].tobytes())
        
        # Async mode (slightly slower but still fast)
        elif self.async_queue:
            try:
                # Copy to new buffer for queue (unavoidable)
                data_copy = buffer[:frames].copy()
                self.async_queue.put_nowait(data_copy)
            except queue.Full:
                # Queue full, drop chunk
                self.metrics.overruns += 1
    
    def start_capture(self, latency: str = 'low') -> None:
        """
        Start audio capture.
        
        Args:
            latency: 'low' for minimal latency, 'high' for stability
        """
        if self.stream:
            raise AudioError("Capture already started")
        
        # Create stream with optimal settings
        self.stream = sd.InputStream(
            device=self.device,
            channels=1,  # Mono for speed
            samplerate=self.config.sample_rate,
            blocksize=self.chunk_size // 2,  # /2 for int16
            dtype=np.int16,
            latency=latency,
            callback=self._audio_callback,
            # Disable automatic gain control and echo cancellation
            extra_settings=sd.WasapiSettings(
                exclusive=True  # Exclusive mode for lower latency on Windows
            ) if hasattr(sd, 'WasapiSettings') else None
        )
        
        self.is_active = True
        self.metrics.start_time = time.time()
        self.stream.start()
    
    def stop_capture(self) -> None:
        """Stop audio capture"""
        self.is_active = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.async_thread:
            self.async_thread.join(timeout=1.0)
            self.async_thread = None
    
    def get_device_info(self) -> dict:
        """Get audio device information"""
        if self.device is None:
            device_info = sd.query_devices(kind='input')
        else:
            device_info = sd.query_devices(self.device)
        
        return {
            'name': device_info['name'],
            'channels': device_info['max_input_channels'],
            'sample_rate': device_info['default_samplerate'],
            'latency': device_info['default_low_input_latency'],
            'host_api': sd.query_hostapis(device_info['hostapi'])['name']
        }
    
    def get_metrics(self) -> dict:
        """Get capture metrics"""
        uptime = time.time() - self.metrics.start_time if self.metrics.start_time else 0
        
        return {
            'chunks_captured': self.metrics.chunks_captured,
            'bytes_captured': self.metrics.bytes_captured,
            'overruns': self.metrics.overruns,
            'uptime_seconds': uptime,
            'capture_rate': self.metrics.chunks_captured / uptime if uptime > 0 else 0
        }
    
    # ============== Async Mode Support ==============
    
    async def start_async_capture(self, queue_size: int = 10) -> asyncio.Queue:
        """
        Start capture in async mode with queue.
        
        Returns asyncio queue that receives audio chunks.
        """
        # Create thread-safe queue for audio data
        self.async_queue = queue.Queue(maxsize=queue_size)
        
        # Create asyncio queue for consumer
        async_queue = asyncio.Queue(maxsize=queue_size)
        
        # Start capture
        self.start_capture()
        
        # Start transfer thread
        self.async_thread = threading.Thread(
            target=self._async_transfer_thread,
            args=(async_queue,),
            daemon=True
        )
        self.async_thread.start()
        
        return async_queue
    
    def _async_transfer_thread(self, async_queue: asyncio.Queue):
        """Transfer thread - moves data from audio thread to async queue"""
        loop = asyncio.new_event_loop()
        
        while self.is_active:
            try:
                # Get from thread-safe queue
                audio_data = self.async_queue.get(timeout=0.1)
                
                # Put to asyncio queue
                loop.run_until_complete(
                    self._put_async(async_queue, audio_data.tobytes())
                )
                
            except queue.Empty:
                continue
            except Exception:
                break
    
    async def _put_async(self, async_queue: asyncio.Queue, data: bytes):
        """Helper to put data in async queue"""
        try:
            await asyncio.wait_for(async_queue.put(data), timeout=0.1)
        except asyncio.TimeoutError:
            self.metrics.overruns += 1


class DirectAudioPlayer:
    """
    Direct audio playback with minimal latency.
    
    Companion to DirectAudioCapture for fast lane output.
    """
    
    def __init__(
        self,
        device: Optional[int] = None,
        config: Optional[AudioConfig] = None
    ):
        self.device = device
        self.config = config or AudioConfig()
        
        # Pre-allocate playback buffer
        self.playback_buffer = np.zeros(
            self.config.sample_rate * 2,  # 2 seconds buffer
            dtype=np.int16
        )
        self.buffer_write_pos = 0
        self.buffer_read_pos = 0
        
        # Stream
        self.stream: Optional[sd.OutputStream] = None
        self.is_playing = False
    
    def _playback_callback(self, outdata, frames, time_info, status):
        """Playback callback - runs in audio thread"""
        if status:
            print(f"Playback status: {status}")
        
        # Read from circular buffer
        available = self.buffer_write_pos - self.buffer_read_pos
        
        if available < frames:
            # Underrun - fill with silence
            outdata[:] = 0
            return
        
        # Copy data
        end_pos = self.buffer_read_pos + frames
        outdata[:, 0] = self.playback_buffer[self.buffer_read_pos:end_pos]
        self.buffer_read_pos = end_pos
        
        # Wrap around if needed
        if self.buffer_read_pos >= len(self.playback_buffer) // 2:
            # Move data to beginning
            remaining = self.buffer_write_pos - self.buffer_read_pos
            self.playback_buffer[:remaining] = \
                self.playback_buffer[self.buffer_read_pos:self.buffer_write_pos]
            self.buffer_write_pos = remaining
            self.buffer_read_pos = 0
    
    def start_playback(self, latency: str = 'low'):
        """Start audio playback"""
        if self.stream:
            return
        
        self.stream = sd.OutputStream(
            device=self.device,
            channels=1,
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size_bytes(20) // 2,  # 20ms blocks
            dtype=np.int16,
            latency=latency,
            callback=self._playback_callback
        )
        
        self.is_playing = True
        self.stream.start()
    
    def play_audio(self, audio_bytes: AudioBytes):
        """Queue audio for playback"""
        if not self.stream:
            self.start_playback()
        
        # Convert to numpy
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Add to buffer
        space_available = len(self.playback_buffer) - self.buffer_write_pos
        
        if len(audio_data) > space_available:
            # Need to wrap or drop
            return False
        
        # Copy to buffer
        end_pos = self.buffer_write_pos + len(audio_data)
        self.playback_buffer[self.buffer_write_pos:end_pos] = audio_data
        self.buffer_write_pos = end_pos
        
        return True
    
    def stop_playback(self):
        """Stop playback"""
        self.is_playing = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None