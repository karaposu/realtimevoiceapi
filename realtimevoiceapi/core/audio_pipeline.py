# here is realtimevoiceapi/core/audio_pipeline.py

"""
Audio Pipeline for BaseEngine

Lightweight audio streaming pipeline that extracts audio processing logic
from BaseEngine without adding overhead. Designed for zero-impact on fast lane.
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from .stream_protocol import StreamEvent, StreamEventType, StreamState
from audioengine.audioengine.audio_types import AudioBytes, AudioConfig
from .exceptions import AudioError
from audioengine.audioengine.audio_manager import AudioManager
from ..strategies.base_strategy import BaseStrategy


class PipelineState(Enum):
    """Pipeline states"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class PipelineMetrics:
    """Metrics for audio pipeline performance"""
    chunks_processed: int = 0
    chunks_sent: int = 0
    chunks_dropped: int = 0
    total_bytes_processed: int = 0
    processing_errors: int = 0
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Get pipeline run duration"""
        if not self.start_time:
            return 0.0
        end = self.stop_time or time.time()
        return end - self.start_time
    
    @property
    def throughput_bps(self) -> float:
        """Get throughput in bytes per second"""
        if self.duration > 0:
            return self.total_bytes_processed / self.duration
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_sent": self.chunks_sent,
            "chunks_dropped": self.chunks_dropped,
            "total_bytes_processed": self.total_bytes_processed,
            "processing_errors": self.processing_errors,
            "duration_seconds": self.duration,
            "throughput_bps": self.throughput_bps
        }


@dataclass
class AudioPipeline:
    """
    Lightweight audio pipeline for BaseEngine.
    
    Extracts audio streaming logic without adding overhead.
    Designed for zero performance impact on fast lane.
    """
    
    # Core components
    audio_manager: Optional[AudioManager] = None
    logger: Optional[logging.Logger] = None
    
    # Configuration
    timeout_ms: float = 100  # Queue timeout in milliseconds
    error_retry_delay: float = 0.1  # Delay after errors
    
    # State
    state: PipelineState = field(default=PipelineState.IDLE, init=False)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics, init=False)
    
    # Runtime
    _audio_queue: Optional[asyncio.Queue] = field(default=None, init=False)
    _vad_enabled: bool = field(default=False, init=False)
    _processing_task: Optional[asyncio.Task] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize logger if not provided"""
        if not self.logger:
            self.logger = logging.getLogger(__name__)
    
    # ============== Pipeline Control ==============
    
    async def start(
        self,
        strategy: BaseStrategy,
        state_checker: Callable[[], bool],
        stream_state_checker: Optional[Callable[[], StreamState]] = None,
        error_handler: Optional[Callable[[StreamEvent], None]] = None
    ) -> None:
        """
        Start the audio pipeline.
        
        Args:
            strategy: Strategy to send audio through
            state_checker: Function to check if pipeline should continue
            stream_state_checker: Optional function to check stream state
            error_handler: Optional error event handler
        """
        if self.state != PipelineState.IDLE:
            raise AudioError(f"Cannot start pipeline in state {self.state}")
        
        if not self.audio_manager:
            raise AudioError("No audio manager configured")
        
        self.state = PipelineState.STARTING
        self.metrics = PipelineMetrics()  # Reset metrics
        
        try:
            # Start audio capture
            self._audio_queue = await self.audio_manager.start_capture()
            self._vad_enabled = bool(self.audio_manager.config.vad_enabled)
            
            # Start processing task
            self._processing_task = asyncio.create_task(
                self._process_stream(
                    strategy,
                    state_checker,
                    stream_state_checker,
                    error_handler
                )
            )
            
            self.state = PipelineState.RUNNING
            self.metrics.start_time = time.time()
            
            self.logger.info("Audio pipeline started")
            
        except Exception as e:
            self.state = PipelineState.ERROR
            self.logger.error(f"Failed to start audio pipeline: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the audio pipeline gracefully"""
        if self.state not in [PipelineState.RUNNING, PipelineState.ERROR]:
            return
        
        self.state = PipelineState.STOPPING
        self.logger.info("Stopping audio pipeline")
        
        # Cancel processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await asyncio.wait_for(self._processing_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Stop audio capture
        if self.audio_manager:
            try:
                await self.audio_manager.stop_capture()
            except Exception as e:
                self.logger.error(f"Error stopping audio capture: {e}")
        
        self.state = PipelineState.STOPPED
        self.metrics.stop_time = time.time()
        
        self.logger.info(f"Audio pipeline stopped. Metrics: {self.metrics.to_dict()}")
    
    # ============== Core Processing ==============
    
    async def _process_stream(
        self,
        strategy: BaseStrategy,
        state_checker: Callable[[], bool],
        stream_state_checker: Optional[Callable[[], StreamState]],
        error_handler: Optional[Callable[[StreamEvent], None]]
    ) -> None:
        """
        Core audio processing loop.
        
        This is the hot path - optimized for minimal overhead.
        """
        if not self._audio_queue:
            self.logger.error("No audio queue available")
            return
        
        # Cache references for faster access in loop
        queue = self._audio_queue
        vad_enabled = self._vad_enabled
        audio_manager = self.audio_manager
        timeout = self.timeout_ms / 1000.0
        
        self.logger.debug(f"Starting audio processing (VAD: {vad_enabled})")
        
        try:
            while state_checker():
                try:
                    # Get audio chunk with timeout
                    audio_chunk = await asyncio.wait_for(
                        queue.get(),
                        timeout=timeout
                    )
                    
                    # Double-check state
                    if not state_checker():
                        break
                    
                    # Update metrics
                    self.metrics.chunks_processed += 1
                    self.metrics.total_bytes_processed += len(audio_chunk)
                    
                    # Check stream state if checker provided
                    if stream_state_checker:
                        stream_state = stream_state_checker()
                        if stream_state not in [StreamState.ACTIVE, StreamState.STARTING]:
                            self.metrics.chunks_dropped += 1
                            continue
                    
                    # Process based on VAD setting
                    should_send = True
                    
                    if vad_enabled and audio_manager:
                        vad_state = audio_manager.process_vad(audio_chunk)
                        should_send = not vad_state or vad_state in ["speech_starting", "speech"]
                    
                    # Send if appropriate
                    if should_send:
                        await strategy.send_audio(audio_chunk)
                        self.metrics.chunks_sent += 1
                    else:
                        self.metrics.chunks_dropped += 1
                    
                except asyncio.TimeoutError:
                    # Normal timeout, just continue
                    continue
                    
                except asyncio.CancelledError:
                    # Pipeline is being stopped
                    self.logger.debug("Audio processing cancelled")
                    break
                    
                except Exception as e:
                    # Other errors
                    self.metrics.processing_errors += 1
                    
                    # Only log if we're still supposed to be running
                    if state_checker():
                        self.logger.error(f"Audio processing error: {e}")
                        
                        # Call error handler if provided
                        if error_handler:
                            try:
                                error_event = StreamEvent(
                                    type=StreamEventType.STREAM_ERROR,
                                    stream_id="audio_pipeline",
                                    timestamp=time.time(),
                                    data={"error": str(e), "component": "audio_pipeline"}
                                )
                                error_handler(error_event)
                            except Exception as handler_error:
                                self.logger.error(f"Error handler failed: {handler_error}")
                        
                        # Brief delay before retrying
                        await asyncio.sleep(self.error_retry_delay)
                    
        except Exception as e:
            self.logger.error(f"Fatal audio pipeline error: {e}")
            self.state = PipelineState.ERROR
            raise
            
        finally:
            # Ensure audio capture is stopped
            if audio_manager:
                try:
                    await audio_manager.stop_capture()
                except Exception as e:
                    self.logger.error(f"Error in pipeline cleanup: {e}")
    
    # ============== Direct Audio Processing ==============
    
    async def process_single_chunk(
        self,
        audio_chunk: AudioBytes,
        strategy: BaseStrategy
    ) -> bool:
        """
        Process a single audio chunk.
        
        Useful for testing or manual control.
        
        Returns:
            True if chunk was sent, False if dropped
        """
        if self.state != PipelineState.RUNNING:
            raise AudioError(f"Pipeline not running (state: {self.state})")
        
        self.metrics.chunks_processed += 1
        self.metrics.total_bytes_processed += len(audio_chunk)
        
        # Apply VAD if enabled
        should_send = True
        
        if self._vad_enabled and self.audio_manager:
            vad_state = self.audio_manager.process_vad(audio_chunk)
            should_send = not vad_state or vad_state in ["speech_starting", "speech"]
        
        if should_send:
            await strategy.send_audio(audio_chunk)
            self.metrics.chunks_sent += 1
            return True
        else:
            self.metrics.chunks_dropped += 1
            return False
    
    # ============== Pipeline Status ==============
    
    @property
    def is_running(self) -> bool:
        """Check if pipeline is running"""
        return self.state == PipelineState.RUNNING
    
    @property
    def is_healthy(self) -> bool:
        """Check if pipeline is healthy"""
        return (
            self.state == PipelineState.RUNNING and
            self.metrics.processing_errors < 10 and  # Arbitrary threshold
            self._processing_task is not None and
            not self._processing_task.done()
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed pipeline status"""
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "is_healthy": self.is_healthy,
            "vad_enabled": self._vad_enabled,
            "has_audio_queue": self._audio_queue is not None,
            "metrics": self.metrics.to_dict()
        }
    
    # ============== Cleanup ==============
    
    async def cleanup(self) -> None:
        """Full cleanup of pipeline resources"""
        await self.stop()
        
        # Clear references
        self._audio_queue = None
        self._processing_task = None
        
        # Reset state
        self.state = PipelineState.IDLE


# ============== Specialized Pipelines ==============

class FastLaneAudioPipeline(AudioPipeline):
    """
    Optimized pipeline for fast lane.
    
    Pre-configures settings for minimal latency.
    """
    
    def __init__(self, audio_manager: Optional[AudioManager] = None):
        super().__init__(
            audio_manager=audio_manager,
            timeout_ms=50,  # Shorter timeout for lower latency
            error_retry_delay=0.05  # Faster retry
        )
    
    async def start_fast_lane(
        self,
        strategy: BaseStrategy,
        is_connected: Callable[[], bool],
        is_listening: Callable[[], bool]
    ) -> None:
        """
        Start with fast lane optimized settings.
        
        Simplified interface for fast lane usage.
        """
        await self.start(
            strategy=strategy,
            state_checker=lambda: is_connected() and is_listening(),
            stream_state_checker=lambda: strategy.get_state()
        )