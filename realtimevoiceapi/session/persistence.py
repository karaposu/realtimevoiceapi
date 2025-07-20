#  file: session/persistence.py


class SessionStateManager:
    """Persist and restore session state"""
    
    async def save_state(self, client: RealtimeClient, filepath: str):
        """Save current session state"""
        state = {
            "session_config": client.session_config.to_dict() if client.session_config else None,
            "conversation_items": [item.__dict__ for item in client.conversation_items],
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    async def restore_state(self, client: RealtimeClient, filepath: str):
        """Restore session state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore configuration
        if state["session_config"]:
            config = SessionConfig.from_dict(state["session_config"])
            await client.configure_session(config)