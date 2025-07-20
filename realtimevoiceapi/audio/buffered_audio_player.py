# realtimevoiceapi/audio/buffered_audio_player.py
import threading
import time
import logging
import numpy as np
import sounddevice as sd
from typing import List, Optional, Callable, Dict, Any
from ..core.audio_types import AudioConfig, AudioBytes


class BufferedAudioPlayer:
    """Enhanced audio player with buffering, metrics, and completion tracking"""
    
    def __init__(self, config: AudioConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Buffering
        self.buffer: List[AudioBytes] = []
        self.buffer_lock = threading.Lock()
        
        # State management
        self.is_playing = False
        self.is_complete = False
        self._playback_active = False  # Actually outputting audio
        self.play_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        
        # Metrics
        self.chunks_received = 0
        self.chunks_played = 0
        self.total_bytes_received = 0
        self.total_bytes_played = 0
        self.first_chunk_time: Optional[float] = None
        self.last_chunk_time: Optional[float] = None
        self.playback_start_time: Optional[float] = None
        self.playback_end_time: Optional[float] = None
        
        # Callbacks
        self.completion_callback: Optional[Callable] = None
        self.chunk_played_callback: Optional[Callable[[int], None]] = None
        
        # Configuration
        self.min_buffer_chunks = 2  # Start playback after this many chunks
        self.max_batch_chunks = 5   # Play up to this many chunks at once
        
    def play(self, audio_data: AudioBytes) -> None:
        """Add audio to buffer and start playback if needed"""
        with self.buffer_lock:
            # Track metrics
            if self.chunks_received == 0:
                self.first_chunk_time = time.time()
                self.logger.debug("First audio chunk received")
            
            self.last_chunk_time = time.time()
            self.chunks_received += 1
            self.total_bytes_received += len(audio_data)
            
            # Add to buffer
            self.buffer.append(audio_data)
            
            self.logger.debug(f"Chunk {self.chunks_received} added to buffer (size: {len(self.buffer)})")
        
        # Start playback if not already playing
        if not self.is_playing:
            self._start_playback()
    
    def mark_complete(self) -> None:
        """Mark that all audio has been received"""
        self.is_complete = True
        self.logger.info(f"Audio reception complete. Total chunks: {self.chunks_received}")
        
    def stop(self, force: bool = False) -> None:
        """Stop playback"""
        self.logger.debug(f"Stop requested (force={force})")
        
        self.is_playing = False
        self.stop_flag.set()
        
        if force:
            try:
                sd.stop()
                self.logger.debug("Force stopped sounddevice")
            except Exception as e:
                self.logger.error(f"Error force stopping: {e}")
        
        # Wait for thread to finish
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)
        
        # Clear buffer
        with self.buffer_lock:
            self.buffer.clear()
        
        self.logger.info("Playback stopped")
    
    def set_completion_callback(self, callback: Callable) -> None:
        """Set callback for when playback completes"""
        self.completion_callback = callback
        
    def set_chunk_played_callback(self, callback: Callable[[int], None]) -> None:
        """Set callback for when chunks are played"""
        self.chunk_played_callback = callback
    
    @property
    def is_actively_playing(self) -> bool:
        """Check if audio is actively being played (not just buffered)"""
        return self.is_playing and (len(self.buffer) > 0 or self._playback_active)
    
    @property
    def buffer_duration_ms(self) -> float:
        """Get current buffer duration in milliseconds"""
        with self.buffer_lock:
            total_bytes = sum(len(chunk) for chunk in self.buffer)
        
        # Calculate duration (16-bit = 2 bytes per sample)
        total_samples = total_bytes / 2
        duration_seconds = total_samples / self.config.sample_rate
        return duration_seconds * 1000
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed playback metrics"""
        metrics = {
            "chunks_received": self.chunks_received,
            "chunks_played": self.chunks_played,
            "chunks_buffered": len(self.buffer),
            "buffer_duration_ms": self.buffer_duration_ms,
            "bytes_received": self.total_bytes_received,
            "bytes_played": self.total_bytes_played,
            "is_playing": self.is_playing,
            "is_actively_playing": self.is_actively_playing,
            "is_complete": self.is_complete
        }
        
        # Timing metrics
        if self.first_chunk_time and self.last_chunk_time:
            metrics["reception_duration_s"] = self.last_chunk_time - self.first_chunk_time
            
        if self.playback_start_time:
            if self.playback_end_time:
                metrics["playback_duration_s"] = self.playback_end_time - self.playback_start_time
            else:
                metrics["playback_duration_s"] = time.time() - self.playback_start_time
        
        # Calculate latency
        if self.first_chunk_time and self.playback_start_time:
            metrics["initial_latency_ms"] = (self.playback_start_time - self.first_chunk_time) * 1000
        
        return metrics
    
    def _start_playback(self) -> None:
        """Start playback thread"""
        if self.is_playing:
            return
        
        self.is_playing = True
        self.is_complete = False
        self.stop_flag.clear()
        self._playback_active = False
        
        self.logger.info("Starting playback thread")
        
        self.play_thread = threading.Thread(target=self._playback_loop, name="AudioPlayback")
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def _playback_loop(self) -> None:
        """Buffered playback loop with enhanced control"""
        self.logger.debug("Playback loop started")
        
        try:
            while self.is_playing and not self.stop_flag.is_set():
                # Check buffer status
                with self.buffer_lock:
                    buffer_size = len(self.buffer)
                    can_play = buffer_size >= self.min_buffer_chunks or (self.is_complete and buffer_size > 0)
                
                if can_play:
                    # Determine how many chunks to play
                    with self.buffer_lock:
                        if self.is_complete:
                            # Play all remaining chunks
                            num_chunks = len(self.buffer)
                        else:
                            # Play up to max batch size
                            num_chunks = min(self.max_batch_chunks, len(self.buffer))
                        
                        # Extract chunks
                        chunks_to_play = []
                        for _ in range(num_chunks):
                            if self.buffer:
                                chunks_to_play.append(self.buffer.pop(0))
                    
                    if chunks_to_play:
                        # Mark first playback
                        if self.playback_start_time is None:
                            self.playback_start_time = time.time()
                            self.logger.info("Playback started")
                        
                        # Combine chunks
                        audio_data = b''.join(chunks_to_play)
                        audio_array = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Play audio
                        self._playback_active = True
                        try:
                            self.logger.debug(f"Playing {len(chunks_to_play)} chunks ({len(audio_data)} bytes)")
                            sd.play(audio_array, self.config.sample_rate, blocking=True)
                            
                            # Update metrics
                            self.chunks_played += len(chunks_to_play)
                            self.total_bytes_played += len(audio_data)
                            
                            # Notify chunk played
                            if self.chunk_played_callback:
                                self.chunk_played_callback(len(chunks_to_play))
                                
                        except Exception as e:
                            self.logger.error(f"Playback error: {e}")
                        finally:
                            self._playback_active = False
                
                # Check if we're done
                if self.is_complete and not self.buffer:
                    self.logger.info("All audio played, ending playback")
                    self.is_playing = False
                    self.playback_end_time = time.time()
                    
                    # Notify completion
                    if self.completion_callback:
                        try:
                            self.completion_callback()
                        except Exception as e:
                            self.logger.error(f"Completion callback error: {e}")
                    break
                
                # If not enough to play, wait a bit
                if not can_play:
                    time.sleep(0.02)  # 20ms wait
                    
        except Exception as e:
            self.logger.error(f"Playback loop error: {e}", exc_info=True)
        finally:
            self._playback_active = False
            self.is_playing = False
            self.logger.debug("Playback loop ended")