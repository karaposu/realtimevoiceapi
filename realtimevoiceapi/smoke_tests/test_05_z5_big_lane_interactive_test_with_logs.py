#!/usr/bin/env python3
"""
Test 05.5: Big Lane Interactive Test with Comprehensive Logging

Interactive test showcasing big lane components:
- Composable audio pipeline with switchable presets
- Event-driven coordination
- Real-time metrics and monitoring
- Multi-processor audio enhancement

Controls:
- SPACE: Hold to record
- 1-4: Switch pipeline presets (Basic/Voice/Quality/Realtime)
- M: Show metrics
- E: Show event history
- ESC: Quit

python -m realtimevoiceapi.smoke_tests.test_05_z5_big_lane_interactive_test_with_logs
"""

import sys
import asyncio
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Callable
import threading
import json
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Create log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Setup comprehensive logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"big_lane_test_{timestamp}.log"

# Configure file handler with detailed format
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Console handler with simpler format
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)
logger.info(f"Starting Big Lane Interactive Test - Log file: {log_file}")

# Try different keyboard libraries
KEYBOARD_LIB = None
try:
    from pynput import keyboard as pynput_keyboard
    KEYBOARD_LIB = "pynput"
    logger.info("Using pynput for keyboard input")
except ImportError:
    logger.warning("pynput not available, will use simple input")

# Import fast lane components (we'll enhance with big lane)
from realtimevoiceapi.fast_lane.direct_audio_capture import DirectAudioCapture
from realtimevoiceapi.fast_lane.fast_stream_manager import FastStreamManager, FastStreamConfig
from realtimevoiceapi.core.audio_types import AudioConfig, AudioMetadata
from realtimevoiceapi.core.stream_protocol import StreamState
from realtimevoiceapi.core.message_protocol import MessageFactory

# Import big lane components
from realtimevoiceapi.big_lane.audio_pipeline import (
    AudioPipeline, PipelinePresets, AudioProcessor,
    ProcessorPriority, ProcessorMetrics
)
from realtimevoiceapi.big_lane.event_bus import EventBus, Event, EventPriority
from realtimevoiceapi.big_lane.response_aggregator import ResponseAggregator

try:
    import sounddevice as sd
    logger.info(f"sounddevice version: {sd.__version__}")
except ImportError:
    logger.error("sounddevice not found!")


class MetricsCollector:
    """Collects and displays metrics from all components"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics_history = deque(maxlen=100)
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        
        # Subscribe to metric events
        event_bus.subscribe("metrics.*", self._handle_metrics_event)
        event_bus.subscribe("pipeline.processed", self._handle_pipeline_event)
        event_bus.subscribe("audio.*", self._handle_audio_event)
        
        # Counters
        self.audio_chunks_processed = 0
        self.audio_chunks_filtered = 0
        self.total_audio_bytes = 0
        self.pipeline_switches = 0
        
    def _handle_metrics_event(self, event: Event):
        """Handle metrics events"""
        self.metrics_history.append({
            "timestamp": event.timestamp,
            "type": event.type,
            "data": event.data
        })
        
    def _handle_pipeline_event(self, event: Event):
        """Handle pipeline events"""
        if event.data.get("filtered", False):
            self.audio_chunks_filtered += 1
        else:
            self.audio_chunks_processed += 1
        
        self.total_audio_bytes += event.data.get("input_size", 0)
        
    def _handle_audio_event(self, event: Event):
        """Handle audio events"""
        if event.type == "audio.pipeline.switched":
            self.pipeline_switches += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "audio_chunks_processed": self.audio_chunks_processed,
            "audio_chunks_filtered": self.audio_chunks_filtered,
            "total_audio_mb": round(self.total_audio_bytes / 1024 / 1024, 2),
            "pipeline_switches": self.pipeline_switches,
            "filter_rate": round(
                self.audio_chunks_filtered / max(1, self.audio_chunks_processed + self.audio_chunks_filtered),
                2
            )
        }
    
    def display_metrics(self):
        """Display current metrics"""
        summary = self.get_summary()
        
        print("\n📊 === METRICS DASHBOARD ===")
        print(f"Audio Processed: {summary['audio_chunks_processed']} chunks")
        print(f"Audio Filtered: {summary['audio_chunks_filtered']} chunks")
        print(f"Filter Rate: {summary['filter_rate']:.1%}")
        print(f"Total Audio: {summary['total_audio_mb']} MB")
        print(f"Pipeline Switches: {summary['pipeline_switches']}")
        print("=" * 30)


class EnhancedAudioBuffer:
    """Audio buffer with event emission"""
    
    def __init__(self, event_bus: EventBus, sample_rate=24000):
        self.event_bus = event_bus
        self.sample_rate = sample_rate
        self.buffer = []
        self.is_playing = False
        self.logger = logging.getLogger(f"{__name__}.AudioBuffer")
        self.main_loop = None  # Will store the main event loop
        
    def set_event_loop(self, loop):
        """Set the main event loop for cross-thread operations"""
        self.main_loop = loop
        
    def add_chunk(self, audio_chunk: bytes):
        """Add audio chunk and emit event"""
        self.buffer.append(audio_chunk)
        
        # Emit event safely
        if self.main_loop and self.main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.event_bus.emit(Event(
                    type="audio.chunk.added",
                    data={
                        "size": len(audio_chunk),
                        "buffer_size": len(self.buffer)
                    }
                )),
                self.main_loop
            )
        
        if not self.is_playing:
            self._start_playback()
    
    def _start_playback(self):
        """Start playback in background"""
        if self.is_playing:
            return
        
        self.is_playing = True
        thread = threading.Thread(target=self._playback_loop, daemon=True)
        thread.start()
    
    def _playback_loop(self):
        """Playback loop"""
        while self.is_playing:
            if len(self.buffer) >= 3:
                chunks = []
                for _ in range(min(5, len(self.buffer))):
                    if self.buffer:
                        chunks.append(self.buffer.pop(0))
                
                if chunks:
                    audio = b''.join(chunks)
                    array = np.frombuffer(audio, dtype=np.int16)
                    
                    try:
                        sd.play(array, self.sample_rate, blocking=True)
                        
                        # Emit playback event safely
                        if self.main_loop and self.main_loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self.event_bus.emit(Event(
                                    type="audio.playback.chunk",
                                    data={"chunks": len(chunks), "bytes": len(audio)}
                                )),
                                self.main_loop
                            )
                    except Exception as e:
                        self.logger.error(f"Playback error: {e}")
            else:
                time.sleep(0.05)
                
            if not self.buffer:
                time.sleep(0.5)
                if not self.buffer:
                    self.is_playing = False
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.buffer.clear()


class BigLaneVoiceChat:
    """Voice chat showcasing big lane architecture"""
    
    def __init__(self, api_key: str, event_bus: EventBus):
        self.logger = logging.getLogger(f"{__name__}.VoiceChat")
        self.api_key = api_key
        self.event_bus = event_bus
        
        # Audio configuration
        self.audio_config = AudioConfig()
        self.capture = DirectAudioCapture(config=self.audio_config)
        self.audio_buffer = EnhancedAudioBuffer(event_bus)
        
        # Stream manager
        self.stream_config = FastStreamConfig(
            websocket_url="wss://api.openai.com/v1/realtime",
            api_key=api_key,
            voice="alloy",
            send_immediately=True
        )
        self.stream_manager = FastStreamManager(config=self.stream_config)
        
        # Audio pipelines
        self.pipelines = {
            "basic": PipelinePresets.create_basic_pipeline(self.audio_config),
            "voice": PipelinePresets.create_voice_pipeline(self.audio_config),
            "quality": PipelinePresets.create_quality_pipeline(self.audio_config),
            "realtime": PipelinePresets.create_realtime_pipeline(self.audio_config)
        }
        self.current_pipeline_name = "voice"
        self.current_pipeline = self.pipelines["voice"]
        
        # Response aggregator
        self.response_aggregator = ResponseAggregator(event_bus)
        
        # State
        self.is_recording = False
        self.recording_buffer = []
        self.interaction_count = 0
        
        # Metrics collector
        self.metrics_collector = MetricsCollector(event_bus)
        
        # Subscribe to events
        self._setup_event_subscriptions()
        
        self.logger.info("BigLaneVoiceChat initialized")

    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions"""
        # Subscribe to stream events
        self.event_bus.subscribe("stream.*", self._handle_stream_event)
        
        # Subscribe to errors
        self.event_bus.subscribe("*.error", self._handle_error_event)
        
        # Subscribe to pipeline events
        self.event_bus.subscribe("pipeline.*", self._handle_pipeline_event)
    
    async def initialize(self):
        """Initialize voice chat"""
        self.logger.info("=== INITIALIZATION START ===")
        print("🎤 Initializing big lane voice chat...")


        self.audio_buffer.set_event_loop(asyncio.get_event_loop())
        
        # Start event bus
        self.event_bus.start()
        
        # Start response aggregator
        await self.response_aggregator.start()
        
        # Connect to OpenAI
        await self.stream_manager.start()
        print("✅ Connected to OpenAI Realtime API")
        
        # Setup callbacks
        self.stream_manager.set_audio_callback(self._on_audio_response)
        self.stream_manager.set_text_callback(self._on_text_response)
        self.stream_manager.set_error_callback(self._on_error)
        
        # Configure session
        session_msg = MessageFactory.session_update(
            modalities=["text", "audio"],
            voice="alloy",
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            instructions="You are a helpful voice assistant. Always respond in English.",
            turn_detection={
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": False
            }
        )
        await self.stream_manager.connection.send(session_msg)
        
        await asyncio.sleep(0.5)
        
        # Emit initialization complete event
        await self.event_bus.emit(Event(
            type="system.initialized",
            data={"components": ["event_bus", "stream_manager", "pipelines"]}
        ))
        
        self.logger.info("=== INITIALIZATION COMPLETE ===")
        print("✅ Big lane voice chat ready!")
        print("\n🎛️  Pipeline: VOICE (Press 1-4 to switch)")
    
    def switch_pipeline(self, pipeline_name: str):
        """Switch audio pipeline"""
        if pipeline_name not in self.pipelines:
            return
        
        old_pipeline = self.current_pipeline_name
        self.current_pipeline_name = pipeline_name
        self.current_pipeline = self.pipelines[pipeline_name]
        
        # Reset the new pipeline
        self.current_pipeline.reset()
        
        print(f"\n🔄 Switched pipeline: {old_pipeline.upper()} → {pipeline_name.upper()}")
        
        # Log processor chain
        processors = self.current_pipeline.get_processor_chain()
        print(f"   Active processors: {', '.join(processors)}")
        
        # Emit event - use emit_sync for synchronous contexts
        self.event_bus.emit_sync(Event(
            type="audio.pipeline.switched",
            data={
                "from": old_pipeline,
                "to": pipeline_name,
                "processors": processors
            },
            priority=EventPriority.HIGH.value
        ))



    async def start_recording(self):
        """Start recording with pipeline processing"""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recording_buffer.clear()
        self.interaction_count += 1
        
        print(f"\n🔴 Recording #{self.interaction_count}... (Pipeline: {self.current_pipeline_name.upper()})")
        
        # Clear input buffer
        clear_msg = MessageFactory.input_audio_buffer_clear()
        await self.stream_manager.connection.send(clear_msg)
        
        # Start capture
        self.audio_queue = await self.capture.start_async_capture()
        
        # Emit recording started event
        await self.event_bus.emit(Event(
            type="recording.started",
            data={
                "interaction": self.interaction_count,
                "pipeline": self.current_pipeline_name
            }
        ))
        
        # Start processing loop
        asyncio.create_task(self._process_audio_loop())
    
    async def stop_recording(self):
        """Stop recording and send to AI"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.capture.stop_capture()
        
        print("⏹️  Processing with big lane pipeline...")
        
        # Emit recording stopped event
        await self.event_bus.emit(Event(
            type="recording.stopped",
            data={"chunks": len(self.recording_buffer)}
        ))
        
        # Send to AI if we have audio
        if self.recording_buffer:
            await self._send_processed_audio()
    
    async def _process_audio_loop(self):
        """Process audio through pipeline"""
        chunk_count = 0
        
        while self.is_recording and self.audio_queue:
            try:
                # Get raw audio chunk
                raw_chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.1)
                chunk_count += 1
                
                # Process through pipeline
                start_time = time.time()
                processed_chunk = await self.current_pipeline.process(raw_chunk)
                process_time_ms = (time.time() - start_time) * 1000
                
                # Emit processing event
                await self.event_bus.emit(Event(
                    type="pipeline.processed",
                    data={
                        "chunk_num": chunk_count,
                        "input_size": len(raw_chunk),
                        "output_size": len(processed_chunk) if processed_chunk else 0,
                        "filtered": processed_chunk is None,
                        "process_time_ms": process_time_ms,
                        "pipeline": self.current_pipeline_name
                    }
                ))
                
                # Add to buffer if not filtered
                if processed_chunk is not None:
                    self.recording_buffer.append(processed_chunk)
                
                # Log every 10th chunk
                if chunk_count % 10 == 0:
                    metrics = self.current_pipeline.get_metrics()
                    avg_time = metrics.get("avg_time_per_chunk", 0)
                    self.logger.debug(
                        f"Processed {chunk_count} chunks, "
                        f"avg pipeline time: {avg_time:.2f}ms"
                    )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
                break
    
    async def _send_processed_audio(self):
        """Send processed audio to AI"""
        if not self.recording_buffer:
            print("⚠️  No audio after pipeline processing")
            return
        
        # Combine chunks
        full_audio = b''.join(self.recording_buffer)
        duration_ms = len(full_audio) / (self.audio_config.sample_rate * 2) * 1000
        
        print(f"📤 Sending to AI: {duration_ms:.0f}ms of processed audio")
        
        # Get pipeline metrics for this interaction
        pipeline_metrics = self.current_pipeline.get_metrics()
        for proc_name, proc_metrics in pipeline_metrics["processors"].items():
            if proc_metrics["chunks"] > 0:
                print(f"   {proc_name}: {proc_metrics['chunks']} chunks, "
                      f"avg {proc_metrics['avg_time']:.1f}ms")
        
        # Send audio
        chunk_size = 24000
        for i in range(0, len(full_audio), chunk_size):
            chunk = full_audio[i:i+chunk_size]
            await self.stream_manager.send_audio(chunk)
        
        # Commit and create response
        await self.stream_manager.connection.send(MessageFactory.input_audio_buffer_commit())
        await self.stream_manager.connection.send(MessageFactory.response_create())
        
        print("⏳ Waiting for response...")
    
    def _on_audio_response(self, audio_chunk: bytes):
        """Handle audio response"""
        self.audio_buffer.add_chunk(audio_chunk)
    
    def _on_text_response(self, text: str):
        """Handle text response"""
        if text.strip():
            print(f"💬 {text}", end="", flush=True)
    
    def _on_error(self, error: Exception):
        """Handle errors"""
        print(f"\n❌ Error: {error}")
        
        # Emit error event
        asyncio.create_task(self.event_bus.emit(Event(
            type="stream.error",
            data={"error": str(error)},
            priority=EventPriority.HIGH.value
        )))
    
    async def _handle_stream_event(self, event: Event):
        """Handle stream events"""
        self.logger.debug(f"Stream event: {event.type}")
    
    async def _handle_error_event(self, event: Event):
        """Handle error events"""
        self.logger.error(f"Error event: {event.type} - {event.data}")
    
    async def _handle_pipeline_event(self, event: Event):
        """Handle pipeline events"""
        if event.type == "pipeline.processed" and event.data.get("filtered"):
            self.logger.debug(f"Chunk filtered by pipeline")
    
    def show_event_history(self):
        """Show recent events"""
        history = self.event_bus.get_history(limit=10)
        
        print("\n📜 === EVENT HISTORY (Last 10) ===")
        for event in history:
            timestamp = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            print(f"{timestamp} | {event.type} | {event.source or 'system'}")
        print("=" * 40)
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("=== CLEANUP START ===")
        
        self.audio_buffer.stop()
        
        if self.stream_manager.state == StreamState.ACTIVE:
            await self.stream_manager.stop()
        
        await self.response_aggregator.stop()
        await self.event_bus.stop()
        
        # Save metrics
        metrics_file = LOG_DIR / f"big_lane_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "summary": self.metrics_collector.get_summary(),
                "pipeline_metrics": {
                    name: pipeline.get_metrics()
                    for name, pipeline in self.pipelines.items()
                }
            }, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
        self.logger.info("=== CLEANUP COMPLETE ===")


async def run_with_pynput(chat: BigLaneVoiceChat):
    """Run with pynput keyboard controls"""
    print("\n🎮 Big Lane Controls:")
    print("  🎤 Hold SPACE = Record")
    print("  🔢 Press 1-4 = Switch Pipeline")
    print("     1: Basic (validation only)")
    print("     2: Voice (noise reduction + normalization + VAD)")
    print("     3: Quality (all enhancements)")
    print("     4: Realtime (minimal processing)")
    print("  📊 Press M = Show Metrics")
    print("  📜 Press E = Show Event History")
    print("  ❌ Press ESC = Quit")
    print("\n✨ Ready!\n")
    
    stop_event = asyncio.Event()
    is_recording = False
    loop = asyncio.get_event_loop()  # Get the event loop
    
    def on_press(key):
        nonlocal is_recording
        try:
            if key == pynput_keyboard.Key.space and not is_recording:
                is_recording = True
                # Use run_coroutine_threadsafe for cross-thread calls
                asyncio.run_coroutine_threadsafe(chat.start_recording(), loop)
            elif key == pynput_keyboard.Key.esc:
                loop.call_soon_threadsafe(stop_event.set)
                return False
            elif hasattr(key, 'char'):
                if key.char == '1':
                    loop.call_soon_threadsafe(chat.switch_pipeline, "basic")
                elif key.char == '2':
                    loop.call_soon_threadsafe(chat.switch_pipeline, "voice")
                elif key.char == '3':
                    loop.call_soon_threadsafe(chat.switch_pipeline, "quality")
                elif key.char == '4':
                    loop.call_soon_threadsafe(chat.switch_pipeline, "realtime")
                elif key.char == 'm':
                    loop.call_soon_threadsafe(chat.metrics_collector.display_metrics)
                elif key.char == 'e':
                    loop.call_soon_threadsafe(chat.show_event_history)
        except Exception as e:
            logger.error(f"Key press error: {e}")
    
    def on_release(key):
        nonlocal is_recording
        try:
            if key == pynput_keyboard.Key.space and is_recording:
                is_recording = False
                # Use run_coroutine_threadsafe for cross-thread calls
                asyncio.run_coroutine_threadsafe(chat.stop_recording(), loop)
        except Exception as e:
            logger.error(f"Key release error: {e}")
    
    listener = pynput_keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()
    
    try:
        await stop_event.wait()
    finally:
        listener.stop()


async def run_simple_controls(chat: BigLaneVoiceChat):
    """Simple keyboard controls without pynput"""
    print("\n📢 Commands:")
    print("  SPACE + ENTER = Start/stop recording")
    print("  1-4 + ENTER = Switch pipeline")
    print("  m + ENTER = Show metrics")
    print("  e + ENTER = Show events")
    print("  quit = Exit")
    print("\n🎯 Ready!\n")
    
    is_recording = False
    
    while True:
        try:
            command = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
            
            if command.lower() == 'quit':
                break
            elif command == ' ':
                if not is_recording:
                    is_recording = True
                    await chat.start_recording()
                else:
                    is_recording = False
                    await chat.stop_recording()
            elif command in ['1', '2', '3', '4']:
                pipelines = ['basic', 'voice', 'quality', 'realtime']
                chat.switch_pipeline(pipelines[int(command) - 1])
            elif command == 'm':
                chat.metrics_collector.display_metrics()
            elif command == 'e':
                chat.show_event_history()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Input error: {e}")


async def main():
    """Main entry point"""
    print("🎙️  RealtimeVoiceAPI - Big Lane Interactive Test")
    print("=" * 60)
    print(f"📝 Log file: {log_file}")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        return
    
    # Create event bus
    event_bus = EventBus(name="big_lane_voice", history_size=1000)
    
    # Create voice chat
    chat = BigLaneVoiceChat(api_key, event_bus)
    
    try:
        # Initialize
        await chat.initialize()
        
        # Choose control method
        if KEYBOARD_LIB == "pynput":
            await run_with_pynput(chat)
        else:
            await run_simple_controls(chat)
            
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
    finally:
        await chat.cleanup()
        print(f"\n📝 Logs saved to: {log_file}")


if __name__ == "__main__":
    asyncio.run(main())