#
import time
import json
from .client import RealtimeClient
#  New file: recording.py
class ConversationRecorder:
    """Record and replay conversations for testing/debugging"""
    
    def __init__(self, client: RealtimeClient):
        self.client = client
        self.events = []
        self.start_time = None
    
    def start_recording(self):
        """Start recording all events"""
        self.start_time = time.time()
        self.client.on_event("*", self._record_event)
    
    async def _record_event(self, event_data):
        self.events.append({
            "timestamp": time.time() - self.start_time,
            "event": event_data
        })
    
    def save_recording(self, filepath: str):
        """Save recording to file"""
        with open(filepath, 'w') as f:
            json.dump({
                "version": "1.0",
                "duration": time.time() - self.start_time,
                "events": self.events
            }, f, indent=2)