"""
VoxTerm Terminal Runner

Manages the terminal lifecycle and provides convenient running methods.
"""

import asyncio
import sys
import os
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import signal
import threading
import time

from .terminal import VoxTerminal
from ..config.settings import TerminalSettings, PRESETS
from ..integration.voice_engine import VoiceEngineAdapter


class VoxTermRunner:
    """
    Runner for VoxTerm terminals.
    
    Handles:
    - Event loop management
    - Thread coordination
    - Graceful shutdown
    - Integration helpers
    """
    
    def __init__(self, terminal: Optional[VoxTerminal] = None):
        self.terminal = terminal
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.runner_thread: Optional[threading.Thread] = None
        self._running = False
        self._shutdown_event = threading.Event()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for VoxTerm"""
        log_level = os.environ.get("VOXTERM_LOG_LEVEL", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr)
            ]
        )
        
        # Reduce noise from some modules
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    @classmethod
    def create_terminal(
        cls,
        title: str = "VoxTerm Voice Chat",
        mode: str = "push_to_talk",
        preset: Optional[str] = None,
        config: Optional[TerminalSettings] = None,
        **kwargs
    ) -> 'VoxTermRunner':
        """
        Create a terminal with runner.
        
        Args:
            title: Terminal window title
            mode: Initial interaction mode
            preset: Configuration preset name
            config: Custom configuration
            **kwargs: Additional terminal arguments
        """
        # Use preset if specified
        if preset and preset in PRESETS:
            config = PRESETS[preset]
        elif not config:
            config = TerminalSettings()
        
        # Create terminal
        terminal = VoxTerminal(
            title=title,
            mode=mode,
            config=config,
            **kwargs
        )
        
        return cls(terminal)
    
    def run(self) -> None:
        """
        Run the terminal synchronously.
        
        This is the main entry point for most applications.
        """
        if not self.terminal:
            raise RuntimeError("No terminal to run")
        
        try:
            # Create and run event loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run terminal
            self.loop.run_until_complete(self.terminal.run())
            
        except KeyboardInterrupt:
            pass  # Normal exit
        finally:
            # Cleanup
            if self.loop and not self.loop.is_closed():
                self.loop.close()
    
    async def run_async(self) -> None:
        """
        Run the terminal asynchronously.
        
        Use this when you need to run VoxTerm alongside other async code.
        """
        if not self.terminal:
            raise RuntimeError("No terminal to run")
        
        await self.terminal.run()
    
    def run_in_thread(self) -> threading.Thread:
        """
        Run the terminal in a separate thread.
        
        Returns the thread object for joining.
        """
        if not self.terminal:
            raise RuntimeError("No terminal to run")
        
        def thread_target():
            # Create new event loop for thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self.terminal.run())
            finally:
                loop.close()
        
        self.runner_thread = threading.Thread(
            target=thread_target,
            daemon=True,
            name="VoxTerm-Runner"
        )
        self.runner_thread.start()
        
        return self.runner_thread
    
    def stop(self) -> None:
        """Stop the terminal"""
        if self.terminal and self.loop:
            # Schedule shutdown in the event loop
            asyncio.run_coroutine_threadsafe(
                self.terminal.shutdown(),
                self.loop
            )
    
    # Integration helpers
    
    @classmethod
    def with_voice_engine(
        cls,
        engine: Any,
        title: str = "VoxTerm Voice Chat",
        mode: str = "push_to_talk",
        adapter_class: Optional[type] = None,
        **kwargs
    ) -> 'VoxTermRunner':
        """
        Create a runner with voice engine already bound.
        
        Args:
            engine: Voice engine instance
            title: Terminal title
            mode: Initial mode
            adapter_class: Custom adapter class (optional)
            **kwargs: Additional terminal arguments
        """
        # Create runner
        runner = cls.create_terminal(title=title, mode=mode, **kwargs)
        
        # Bind voice engine
        if adapter_class:
            adapter = adapter_class()
            adapter.connect_to_terminal(runner.terminal, engine)
        else:
            # Direct binding
            runner.terminal.bind_voice_engine(engine)
        
        return runner
    
    @classmethod
    def quick_start(
        cls,
        voice_engine: Any,
        mode: str = "push_to_talk"
    ) -> None:
        """
        Quick start helper for simple applications.
        
        Example:
            engine = VoiceEngine(api_key="...")
            VoxTermRunner.quick_start(engine)
        """
        runner = cls.with_voice_engine(
            engine,
            title="Voice Chat",
            mode=mode
        )
        
        # Add shutdown hook
        def on_ready():
            print("\n🎤 Voice chat ready! Press 'h' for help.\n")
        
        runner.terminal.on_ready(on_ready)
        
        # Run
        runner.run()


# Convenience functions

def run_terminal(
    title: str = "VoxTerm Voice Chat",
    mode: str = "push_to_talk",
    config: Optional[TerminalSettings] = None,
    voice_engine: Optional[Any] = None
) -> None:
    """
    Run a VoxTerm terminal.
    
    This is the simplest way to use VoxTerm:
    
    ```python
    from voxterm import run_terminal
    from my_voice_api import VoiceEngine
    
    engine = VoiceEngine(api_key="...")
    run_terminal(voice_engine=engine)
    ```
    """
    runner = VoxTermRunner.create_terminal(
        title=title,
        mode=mode,
        config=config
    )
    
    if voice_engine:
        runner.terminal.bind_voice_engine(voice_engine)
    
    runner.run()


async def run_terminal_async(
    title: str = "VoxTerm Voice Chat",
    mode: str = "push_to_talk",
    config: Optional[TerminalSettings] = None,
    voice_engine: Optional[Any] = None
) -> VoxTerminal:
    """
    Run a terminal asynchronously.
    
    Returns the terminal instance for further interaction.
    """
    terminal = VoxTerminal(
        title=title,
        mode=mode,
        config=config
    )
    
    if voice_engine:
        terminal.bind_voice_engine(voice_engine)
    
    # Start terminal in background
    task = asyncio.create_task(terminal.run())
    
    # Wait for it to be ready
    while terminal.state != ComponentState.RUNNING:
        await asyncio.sleep(0.1)
    
    return terminal


class VoxTermContext:
    """
    Context manager for running VoxTerm.
    
    Example:
        async with VoxTermContext(voice_engine=engine) as terminal:
            # Terminal is running
            await do_something()
        # Terminal automatically shuts down
    """
    
    def __init__(
        self,
        title: str = "VoxTerm Voice Chat",
        mode: str = "push_to_talk",
        config: Optional[TerminalSettings] = None,
        voice_engine: Optional[Any] = None
    ):
        self.title = title
        self.mode = mode
        self.config = config
        self.voice_engine = voice_engine
        self.terminal: Optional[VoxTerminal] = None
        self._task: Optional[asyncio.Task] = None
    
    async def __aenter__(self) -> VoxTerminal:
        """Enter context - start terminal"""
        self.terminal = VoxTerminal(
            title=self.title,
            mode=self.mode,
            config=self.config
        )
        
        if self.voice_engine:
            self.terminal.bind_voice_engine(self.voice_engine)
        
        # Initialize and start
        await self.terminal.initialize()
        await self.terminal.start()
        
        # Run in background
        self._task = asyncio.create_task(self._run_terminal())
        
        return self.terminal
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - stop terminal"""
        if self.terminal:
            await self.terminal.shutdown()
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _run_terminal(self):
        """Run terminal main loop"""
        try:
            await self.terminal._main_loop()
        except asyncio.CancelledError:
            pass


# Example usage patterns

def example_basic():
    """Basic usage example"""
    from realtimevoiceapi import VoiceEngine
    
    # Create voice engine
    engine = VoiceEngine(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Run terminal
    run_terminal(voice_engine=engine)


def example_custom():
    """Custom configuration example"""
    from realtimevoiceapi import VoiceEngine
    
    # Custom settings
    config = TerminalSettings()
    config.display.theme = "light"
    config.key_bindings.push_to_talk = "ctrl"
    config.behavior.auto_scroll = True
    
    # Create and run
    engine = VoiceEngine(api_key=os.getenv("OPENAI_API_KEY"))
    
    runner = VoxTermRunner.create_terminal(
        title="My Voice Assistant",
        mode="always_on",
        config=config
    )
    runner.terminal.bind_voice_engine(engine)
    runner.run()


async def example_async():
    """Async usage example"""
    from realtimevoiceapi import VoiceEngine
    
    # Create engine
    engine = VoiceEngine(api_key=os.getenv("OPENAI_API_KEY"))
    await engine.connect()
    
    # Run terminal alongside other async code
    async with VoxTermContext(voice_engine=engine) as terminal:
        # Terminal is running
        
        # Do other async work
        await asyncio.sleep(60)  # Run for 1 minute
        
        # Terminal automatically shuts down


def example_threaded():
    """Threaded usage example"""
    from realtimevoiceapi import VoiceEngine
    
    # Create components
    engine = VoiceEngine(api_key=os.getenv("OPENAI_API_KEY"))
    
    runner = VoxTermRunner.with_voice_engine(
        engine,
        title="Background Voice Chat"
    )
    
    # Run in thread
    thread = runner.run_in_thread()
    
    # Do other work in main thread
    try:
        while True:
            # Your application logic here
            time.sleep(1)
    except KeyboardInterrupt:
        runner.stop()
        thread.join()


if __name__ == "__main__":
    # Run basic example if executed directly
    example_basic()