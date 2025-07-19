#keyboard.py


"""
Simple keyboard handler for VoxTerm

Provides basic keyboard input without complex abstractions.
"""

import asyncio
import threading
from typing import Optional, Callable, Dict
import sys
import platform

# Try to import keyboard library
try:
    from pynput import keyboard as pynput_keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False


class KeyboardHandler:
    """Simple keyboard input handler"""
    
    def __init__(self):
        self.callbacks = {
            'press': {},
            'release': {}
        }
        self.listener = None
        self.running = False
        self.pressed_keys = set()
        
        # Common key mappings
        self.key_map = {
            'space': ' ',
            'enter': '\n',
            'return': '\n',
            'esc': 'escape',
            'ctrl': 'control'
        }
        
    def on_press(self, key: str, callback: Callable):
        """Register a key press callback"""
        self.callbacks['press'][key.lower()] = callback
        
    def on_release(self, key: str, callback: Callable):
        """Register a key release callback"""
        self.callbacks['release'][key.lower()] = callback
        
    def start(self):
        """Start listening for keyboard input"""
        self.running = True
        
        if HAS_PYNPUT:
            self._start_pynput()
        elif HAS_KEYBOARD:
            self._start_keyboard()
        else:
            print("⚠️  No keyboard library found. Install with: pip install pynput")
            self._start_manual()
            
    def stop(self):
        """Stop listening"""
        self.running = False
        if self.listener:
            if HAS_PYNPUT and hasattr(self.listener, 'stop'):
                self.listener.stop()
                
    def _start_pynput(self):
        """Use pynput library (best for macOS)"""
        def on_press(key):
            if not self.running:
                return False
                
            key_str = self._pynput_key_to_string(key)
            if key_str and key_str not in self.pressed_keys:
                self.pressed_keys.add(key_str)
                self._handle_key_event('press', key_str)
                
        def on_release(key):
            if not self.running:
                return False
                
            key_str = self._pynput_key_to_string(key)
            if key_str:
                self.pressed_keys.discard(key_str)
                self._handle_key_event('release', key_str)
                
        self.listener = pynput_keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.listener.start()
        
    def _start_keyboard(self):
        """Use keyboard library (best for Windows/Linux)"""
        def handle_event(event):
            if not self.running:
                return
                
            key_str = event.name.lower()
            if event.event_type == 'down':
                if key_str not in self.pressed_keys:
                    self.pressed_keys.add(key_str)
                    self._handle_key_event('press', key_str)
            else:  # up
                self.pressed_keys.discard(key_str)
                self._handle_key_event('release', key_str)
                
        keyboard.hook(handle_event)
        
    def _start_manual(self):
        """Fallback: manual input mode"""
        print("Running in manual mode. Type commands:")
        print("  'space' - Start recording")
        print("  'stop' - Stop recording")
        print("  'quit' - Exit")
        
        def input_loop():
            while self.running:
                try:
                    cmd = input("> ").lower().strip()
                    if cmd == 'space':
                        self._handle_key_event('press', 'space')
                    elif cmd == 'stop':
                        self._handle_key_event('release', 'space')
                    elif cmd in ['q', 'quit', 'exit']:
                        self._handle_key_event('press', 'q')
                    elif cmd == 'm':
                        self._handle_key_event('press', 'm')
                except:
                    break
                    
        threading.Thread(target=input_loop, daemon=True).start()
        
    def _pynput_key_to_string(self, key) -> Optional[str]:
        """Convert pynput key to string"""
        if hasattr(key, 'char') and key.char:
            return key.char.lower()
        elif hasattr(key, 'name'):
            return key.name.lower()
        elif key == pynput_keyboard.Key.space:
            return 'space'
        elif key == pynput_keyboard.Key.enter:
            return 'enter'
        elif key == pynput_keyboard.Key.esc:
            return 'escape'
        return None
        
    def _handle_key_event(self, event_type: str, key: str):
        """Handle a keyboard event"""
        # Normalize key
        key = self.key_map.get(key, key)
        
        # Call registered callback
        if key in self.callbacks[event_type]:
            callback = self.callbacks[event_type][key]
            
            # Run async callbacks in the event loop
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback())
            else:
                # Run sync callbacks in thread to avoid blocking
                threading.Thread(target=callback, daemon=True).start()


class SimpleKeyboard:
    """Even simpler keyboard helper for basic use cases"""
    
    def __init__(self):
        self.handler = KeyboardHandler()
        
    def on_space(self, on_press: Callable, on_release: Callable):
        """Register space bar handlers"""
        self.handler.on_press('space', on_press)
        self.handler.on_release('space', on_release)
        
    def on_key(self, key: str, callback: Callable):
        """Register a key press handler"""
        self.handler.on_press(key, callback)
        
    def start(self):
        """Start listening"""
        self.handler.start()
        
    def stop(self):
        """Stop listening"""
        self.handler.stop()


# Utility functions for common patterns

def wait_for_key(key: str = 'enter') -> asyncio.Future:
    """Wait for a specific key press"""
    future = asyncio.Future()
    
    def on_key():
        if not future.done():
            future.set_result(True)
            
    handler = KeyboardHandler()
    handler.on_press(key, on_key)
    handler.start()
    
    async def cleanup():
        await future
        handler.stop()
        
    asyncio.create_task(cleanup())
    return future


async def get_confirmation(prompt: str = "Press ENTER to continue...") -> bool:
    """Simple confirmation prompt"""
    print(prompt, end="", flush=True)
    await wait_for_key('enter')
    print()
    return True