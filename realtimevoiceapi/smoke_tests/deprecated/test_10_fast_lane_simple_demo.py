#!/usr/bin/env python3
"""
Test 08: Fast Lane Simple Demo

A minimal voice chat demo using the fast lane implementation.
Connects to OpenAI's Realtime API and enables voice conversation.

Requirements:
- Valid OpenAI API key with Realtime API access
- Microphone and speakers
- sounddevice installed

Usage:
    python -m realtimevoiceapi.smoke_tests.test_08_fast_lane_simple_demo
"""

import asyncio
import sys
import os
import termios
import tty
from pathlib import Path
from datetime import datetime
import logging
import traceback
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from realtimevoiceapi import VoiceEngine, VoiceEngineConfig

# Set up logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class SimpleVoiceDemo:
    """Minimal voice chat demo with verbose output"""
    
    def __init__(self, api_key: str, verbose: bool = True):
        self.api_key = api_key
        self.verbose = verbose
        self.engine = None
        self.running = True
        self.status = "OFFLINE"
        self.old_terminal_settings = None
        self.error_details = None
        
        # Stats
        self.audio_chunks_sent = 0
        self.audio_chunks_received = 0
        self.text_chunks_received = 0
        self.last_activity = time.time()
        
    async def start(self):
        """Start the voice chat demo"""
        try:
            # Configure engine for fast lane
            config = VoiceEngineConfig(
                api_key=self.api_key,
                mode="fast",
                latency_mode="ultra_low",
                vad_enabled=True,
                vad_threshold=0.02,
                voice="alloy",
                log_level="DEBUG" if self.verbose else "INFO"
            )
            
            # Create engine
            self._log("Creating VoiceEngine...")
            self.engine = VoiceEngine(config=config)
            
            # Set up callbacks with verbose output
            def on_audio(audio_data):
                self.audio_chunks_received += 1
                self._update_status(f"AUDIO RECEIVED ({len(audio_data)} bytes)")
                self._log(f"Received audio chunk #{self.audio_chunks_received}: {len(audio_data)} bytes")
            
            def on_text(text):
                self.text_chunks_received += 1
                self._update_status(f"TEXT: {text[:50]}...")
                self._log(f"Received text: {text}")
                # Print the actual response
                print(f"\n🤖 AI: {text}\n")
            
            def on_error(error):
                self._update_status(f"ERROR: {str(error)}")
                self._log(f"Engine error: {error}", level='error')
            
            self.engine.on_audio_response = on_audio
            self.engine.on_text_response = on_text
            self.engine.on_error = on_error
            
            # Connect
            self._update_status("CONNECTING...")
            self._log("Attempting to connect to OpenAI Realtime API...")
            await self.engine.connect()
            self._log("Connected successfully!")
            
            # Start listening
            self._update_status("STARTING MIC...")
            self._log("Starting audio capture...")
            await self.engine.start_listening()
            self._log("Audio capture started!")
            
            self.status = "LISTENING"
            
            # Hook into the audio processing to see what's happening
            if hasattr(self.engine, '_audio_processing_task'):
                self._log("Audio processing task is running")
            
            # Run UI
            await self._run_ui()
            
        except Exception as e:
            self.error_details = {
                'error': str(e),
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            self._log(f"Fatal error: {self.error_details}", level='error')
            self.status = "ERROR"
            await self._show_error()
        finally:
            await self.cleanup()
    
    async def _run_ui(self):
        """Run the minimal UI with activity monitoring"""
        # Set terminal to raw mode for key detection
        self.old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin)
        
        # Create tasks
        ui_task = asyncio.create_task(self._ui_loop())
        status_task = asyncio.create_task(self._status_loop())
        monitor_task = asyncio.create_task(self._monitor_activity())
        
        try:
            # Wait for tasks
            await asyncio.gather(ui_task, status_task, monitor_task)
        except asyncio.CancelledError:
            pass
    
    async def _ui_loop(self):
        """Handle keyboard input"""
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Check for key press (non-blocking)
                key = await loop.run_in_executor(None, self._get_key)
                
                if key:
                    if key.lower() == 'x':
                        self.running = False
                        break
                    elif key.lower() == 's':
                        # Show stats
                        await self._show_stats()
                    elif key.lower() == 't':
                        # Test by sending text
                        await self._send_test_message()
                
                await asyncio.sleep(0.1)
            except Exception as e:
                self._log(f"UI loop error: {e}", level='error')
    
    async def _status_loop(self):
        """Update status display"""
        while self.running:
            self._print_status(self.status)
            await asyncio.sleep(1)
    
    async def _monitor_activity(self):
        """Monitor for audio activity"""
        while self.running:
            try:
                # Check if we have access to internal components
                if hasattr(self.engine, '_strategy') and self.engine._strategy:
                    strategy = self.engine._strategy
                    
                    # Check audio capture
                    if hasattr(strategy, 'audio_capture') and strategy.audio_capture:
                        metrics = strategy.audio_capture.get_metrics()
                        if metrics['chunks_captured'] > self.audio_chunks_sent:
                            new_chunks = metrics['chunks_captured'] - self.audio_chunks_sent
                            self.audio_chunks_sent = metrics['chunks_captured']
                            self._update_status(f"MIC: {new_chunks} chunks")
                            self._log(f"Captured {new_chunks} new audio chunks")
                    
                    # Check VAD
                    if hasattr(strategy, 'vad_detector') and strategy.vad_detector:
                        vad_metrics = strategy.vad_detector.get_metrics()
                        state = vad_metrics.get('state', 'unknown')
                        if state == 'speech':
                            self._update_status("SPEAKING DETECTED")
                        elif state == 'silence':
                            self._update_status("SILENCE")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self._log(f"Monitor error: {e}", level='error')
                await asyncio.sleep(1)
    
    def _get_key(self):
        """Get a single key press (non-blocking)"""
        import select
        
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None
    
    def _print_status(self, status: str):
        """Print status line"""
        # Clear line and print status
        sys.stdout.write('\r\033[K')  # Clear line
        
        # Build status line with stats
        stats = f"[Sent: {self.audio_chunks_sent} | Rcvd: {self.audio_chunks_received}]"
        
        sys.stdout.write(f"Status: {status:<20} {stats} | S=Stats T=Test X=Quit")
        sys.stdout.flush()
    
    def _update_status(self, status: str):
        """Update status temporarily"""
        self.status = status
        self.last_activity = time.time()
        
        # Reset to LISTENING after a moment
        if status != "ERROR":
            asyncio.create_task(self._reset_status())
    
    async def _reset_status(self):
        """Reset status to LISTENING after a delay"""
        await asyncio.sleep(2)
        if self.status not in ["ERROR", "OFFLINE"]:
            self.status = "LISTENING"
    
    async def _show_stats(self):
        """Show detailed statistics"""
        # Restore terminal temporarily
        if self.old_terminal_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        
        print("\n\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        print(f"Audio chunks sent: {self.audio_chunks_sent}")
        print(f"Audio chunks received: {self.audio_chunks_received}")
        print(f"Text chunks received: {self.text_chunks_received}")
        
        if hasattr(self.engine, '_strategy') and self.engine._strategy:
            metrics = self.engine._strategy.get_metrics()
            print(f"\nEngine Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        print("="*60)
        print("\nPress Enter to continue...")
        input()
        
        # Restore raw mode
        tty.setraw(sys.stdin)
    
    async def _send_test_message(self):
        """Send a test text message"""
        # Restore terminal temporarily
        if self.old_terminal_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        
        print("\n\nSending test message...")
        await self.engine.send_text("Hello, can you hear me?")
        
        # Restore raw mode
        tty.setraw(sys.stdin)
    
    def _log(self, message: str, level: str = 'info'):
        """Log message"""
        logger = logging.getLogger('VoiceDemo')
        if level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    
    async def _show_error(self, detailed: bool = True):
        """Show error details"""
        if not self.error_details:
            return
        
        # Restore terminal
        if self.old_terminal_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        
        print("\n\n" + "="*60)
        print("ERROR DETAILS")
        print("="*60)
        print(f"Type: {self.error_details['type']}")
        print(f"Message: {self.error_details['error']}")
        print("\nFull Traceback:")
        print(self.error_details['traceback'])
        print("="*60)
    
    async def cleanup(self):
        """Cleanup resources"""
        # Restore terminal
        if self.old_terminal_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        
        # Clear line
        sys.stdout.write('\r\033[K')
        
        if self.error_details:
            print("Failed to start. See error details above.")
        else:
            print("Shutting down...")
        
        sys.stdout.flush()
        
        # Disconnect engine
        if self.engine and self.engine._is_connected:
            try:
                await self.engine.disconnect()
            except Exception as e:
                self._log(f"Cleanup error: {e}", level='error')


async def main():
    """Run the demo"""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("\nExample:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  python -m realtimevoiceapi.smoke_tests.test_08_fast_lane_simple_demo")
        return
    
    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Print header
    print("Fast Lane Voice Chat Demo")
    print("=" * 30)
    print("VERBOSE MODE - All activity will be logged")
    print("Controls:")
    print("  S - Show statistics")
    print("  T - Send test message")
    print("  X - Exit")
    print("=" * 30)
    print()
    
    # Create and run demo
    demo = SimpleVoiceDemo(api_key, verbose=True)
    
    try:
        await demo.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # For Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())