"""
Integration of Unified Message System with Layered Architecture

This shows how the RealtimeMessage dataclass integrates with our layered
WebSocket architecture to provide maximum flexibility and configuration.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Import the layered architecture
from websocket_handler import WebSocketHandler, WebSocketMessage, ConnectionState
from realtime_llm_handler import RealtimeLLMHandler, LLMEventHandler  
from realtime_generation_engine import RealtimeGenerationEngine

# Import the unified message system
from unified_message_dataclass import RealtimeMessage, RealtimeResponse, ModelConfig, AudioConfig


# ============================================================================
# ENHANCED GENERATION ENGINE WITH MESSAGE SUPPORT
# ============================================================================

class MessageAwareGenerationEngine(RealtimeGenerationEngine):
    """
    Enhanced generation engine that works with RealtimeMessage dataclass
    """
    
    async def process_message(self, message: RealtimeMessage) -> RealtimeResponse:
        """
        Process a unified RealtimeMessage and return RealtimeResponse
        
        Args:
            message: Unified message with all configuration
            
        Returns:
            Complete response with all requested data
        """
        # Validate message first
        issues = message.validate()
        if issues:
            return RealtimeResponse(
                request_id=message.request_id,
                success=False,
                error=f"Message validation failed: {'; '.join(issues)}"
            )
        
        start_time = time.time()
        retries = 0
        
        while retries <= message.retry_config.max_retries:
            try:
                # Apply per-message session configuration if needed
                if await self._should_update_session(message):
                    await self._apply_message_session_config(message)
                
                # Process based on message content
                if message.has_tool_call:
                    response = await self._process_function_call_message(message)
                elif message.has_audio_input:
                    response = await self._process_audio_message(message)  
                elif message.has_text_input:
                    response = await self._process_text_message(message)
                else:
                    response = RealtimeResponse(
                        request_id=message.request_id,
                        success=False,
                        error="No processable input found in message"
                    )
                
                # Add processing metadata
                response.processing_time_ms = (time.time() - start_time) * 1000
                response.retries_attempted = retries
                response.model_used = message.model_config.model
                
                if message.audio_config and response.has_audio:
                    response.voice_used = message.audio_config.voice
                
                return response
                
            except Exception as e:
                retries += 1
                error_str = str(e)
                
                # Check if this error should trigger a retry
                should_retry = (
                    retries <= message.retry_config.max_retries and
                    any(retry_error in error_str.lower() 
                        for retry_error in message.retry_config.retry_on_errors)
                )
                
                if should_retry:
                    # Calculate delay with exponential backoff
                    if message.retry_config.exponential_backoff:
                        delay = min(
                            message.retry_config.base_delay * (2 ** retries),
                            message.retry_config.max_delay
                        )
                    else:
                        delay = message.retry_config.base_delay
                    
                    self.logger.warning(f"Request {message.request_id} failed, retrying in {delay}s: {error_str}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # No more retries or non-retryable error
                    return RealtimeResponse(
                        request_id=message.request_id,
                        success=False,
                        error=error_str,
                        retries_attempted=retries,
                        processing_time_ms=(time.time() - start_time) * 1000
                    )
        
        # If we get here, all retries exhausted
        return RealtimeResponse(
            request_id=message.request_id,
            success=False,
            error="Maximum retries exceeded",
            retries_attempted=retries,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _should_update_session(self, message: RealtimeMessage) -> bool:
        """Check if session config needs updating for this message"""
        # Update if model, voice, or temperature differs from current session
        current_config = getattr(self, '_current_session_config', {})
        
        return (
            current_config.get('model') != message.model_config.model or
            current_config.get('voice') != message.audio_config.voice or
            abs(current_config.get('temperature', 0.8) - message.model_config.temperature) > 0.01
        )
    
    async def _apply_message_session_config(self, message: RealtimeMessage):
        """Apply message-specific session configuration"""
        session_config = message.to_session_config()
        
        # Apply any session overrides from the message
        if message.session_overrides:
            session_config.update(message.session_overrides)
        
        await self.llm_handler.configure_session(session_config)
        
        # Cache current config to avoid unnecessary updates
        self._current_session_config = session_config
        
        self.logger.info(f"Updated session config for message {message.request_id}")
    
    async def _process_text_message(self, message: RealtimeMessage) -> RealtimeResponse:
        """Process text-based message"""
        # Send conversation items to LLM
        items = message.to_conversation_items()
        
        if items:
            # For now, send the last user message (simplified)
            user_content = None
            for item in items:
                if item.get('role') == 'user':
                    for content in item.get('content', []):
                        if content.get('type') == 'input_text':
                            user_content = content.get('text')
                            break
            
            if user_content:
                await self.llm_handler.send_text_message(user_content)
        
        # Wait for response based on timeout
        return await self._wait_for_message_response(message)
    
    async def _process_audio_message(self, message: RealtimeMessage) -> RealtimeResponse:
        """Process audio-based message"""
        if message.input_audio_b64:
            await self.llm_handler.send_audio_data(message.input_audio_b64)
            await self.llm_handler.commit_audio_input()
        
        return await self._wait_for_message_response(message)
    
    async def _process_function_call_message(self, message: RealtimeMessage) -> RealtimeResponse:
        """Process function call message"""
        if message.tool_call:
            # Execute function directly if we have it
            if message.tool_call.name in self._function_registry:
                try:
                    result = await self._execute_function(
                        message.tool_call.name,
                        message.tool_call.arguments
                    )
                    
                    return RealtimeResponse(
                        request_id=message.request_id,
                        text=str(result),
                        function_calls=[{
                            'name': message.tool_call.name,
                            'arguments': message.tool_call.arguments,
                            'result': result,
                            'success': True
                        }]
                    )
                except Exception as e:
                    return RealtimeResponse(
                        request_id=message.request_id,
                        success=False,
                        error=f"Function execution failed: {e}",
                        function_calls=[{
                            'name': message.tool_call.name,
                            'arguments': message.tool_call.arguments,
                            'error': str(e),
                            'success': False
                        }]
                    )
            else:
                return RealtimeResponse(
                    request_id=message.request_id,
                    success=False,
                    error=f"Function {message.tool_call.name} not found"
                )
        
        return RealtimeResponse(
            request_id=message.request_id,
            success=False,
            error="No tool call found in function call message"
        )
    
    async def _wait_for_message_response(self, message: RealtimeMessage) -> RealtimeResponse:
        """Wait for LLM response to message"""
        # This is simplified - in real implementation you'd track the specific response
        # for this message and handle streaming if requested
        
        timeout = message.timeout
        start_time = time.time()
        
        # Create response tracking
        response = RealtimeResponse(request_id=message.request_id)
        
        # Wait for response completion (simplified)
        while (time.time() - start_time) < timeout:
            # Check if we have responses from LLM handler
            current_responses = self.llm_handler.get_current_responses()
            
            if current_responses:
                # Take the most recent completed response
                for resp_id, resp_data in current_responses.items():
                    if resp_data.get('status') == 'completed':
                        # Build response from LLM data
                        response.text = resp_data.get('text', '').strip() or None
                        
                        # Process audio if requested
                        if message.wants_audio_output:
                            audio_chunks = resp_data.get('audio_chunks', [])
                            if audio_chunks:
                                response.audio_b64 = ''.join(audio_chunks)
                        
                        response.duration_ms = resp_data.get('completed_at', 0) - resp_data.get('started_at', 0)
                        response.raw_response = resp_data
                        
                        return response
            
            await asyncio.sleep(0.1)
        
        # Timeout
        response.success = False
        response.error = "Response timeout"
        return response


# ============================================================================
# ENHANCED SERVICE WITH MESSAGE SUPPORT  
# ============================================================================

class MessageAwareRealtimeService:
    """
    Enhanced service that uses RealtimeMessage for all communication
    """
    
    def __init__(
        self, 
        api_key: str,
        default_model: str = "gpt-4o-realtime-preview",
        default_voice: str = "alloy",
        logger: Optional[logging.Logger] = None
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.default_voice = default_voice
        self.logger = logger or logging.getLogger(__name__)
        
        # Create layered architecture
        self.websocket_handler = WebSocketHandler(api_key, logger)
        self.generation_engine = MessageAwareGenerationEngine(
            self.websocket_handler,
            logger=logger
        )
        
        # Service state
        self._is_connected = False
        self._message_history: List[RealtimeMessage] = []
        self._response_history: List[RealtimeResponse] = []
    
    async def connect(self) -> bool:
        """Connect to the service"""
        if self._is_connected:
            return True
        
        success = await self.websocket_handler.connect()
        if success:
            self._is_connected = True
            self.logger.info("Message-aware service connected")
        
        return success
    
    async def disconnect(self):
        """Disconnect from the service"""
        await self.websocket_handler.disconnect()
        self._is_connected = False
        self.logger.info("Message-aware service disconnected")
    
    async def send_message(self, message: RealtimeMessage) -> RealtimeResponse:
        """
        Send a RealtimeMessage and get RealtimeResponse
        
        Args:
            message: Complete message with all configuration
            
        Returns:
            Complete response with requested data
        """
        if not self._is_connected:
            raise RuntimeError("Service not connected. Call connect() first.")
        
        # Add to message history
        self._message_history.append(message)
        
        # Process through generation engine
        response = await self.generation_engine.process_message(message)
        
        # Add to response history
        self._response_history.append(response)
        
        self.logger.info(
            f"Message {message.request_id} processed: "
            f"success={response.success}, "
            f"has_text={response.has_text}, "
            f"has_audio={response.has_audio}, "
            f"duration={response.processing_time_ms:.0f}ms"
        )
        
        return response
    
    # ========================================
    # Convenience Methods (using dataclass)
    # ========================================
    
    async def send_text(
        self, 
        prompt: str, 
        *,
        system_prompt: Optional[str] = None,
        output_format: str = "text",
        voice: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> RealtimeResponse:
        """Send text message with optional configuration"""
        message = RealtimeMessage.text(
            prompt,
            system_prompt=system_prompt,
            output_format=output_format,
            model=self.default_model,
            voice=voice or self.default_voice,
            temperature=temperature,
            **kwargs
        )
        return await self.send_message(message)
    
    async def send_audio(
        self,
        audio_input: Union[str, Path, bytes],
        *,
        output_format: str = "both",
        voice: Optional[str] = None,
        **kwargs
    ) -> RealtimeResponse:
        """Send audio message with optional configuration"""
        message = RealtimeMessage.audio(
            audio_input,
            output_format=output_format,
            voice=voice or self.default_voice,
            **kwargs
        )
        return await self.send_message(message)
    
    async def send_multimodal(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Union[str, Path, bytes]] = None,
        images: Optional[List[str]] = None,
        output_format: str = "both",
        **kwargs
    ) -> RealtimeResponse:
        """Send multimodal message"""
        message = RealtimeMessage.multimodal(
            text=text,
            audio=audio,
            images=images,
            output_format=output_format,
            **kwargs
        )
        return await self.send_message(message)
    
    async def call_function(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        *,
        output_format: str = "text",
        **kwargs
    ) -> RealtimeResponse:
        """Call a function"""
        message = RealtimeMessage.function_call(
            function_name,
            arguments,
            output_format=output_format,
            **kwargs
        )
        return await self.send_message(message)
    
    # ========================================
    # Advanced Configuration Methods
    # ========================================
    
    def create_custom_message(
        self,
        operation_name: str,
        **message_kwargs
    ) -> RealtimeMessage:
        """Create a custom message with default configurations"""
        return RealtimeMessage(
            operation_name=operation_name,
            model_config=ModelConfig(model=self.default_model),
            audio_config=AudioConfig(voice=self.default_voice),
            **message_kwargs
        )
    
    async def send_with_custom_config(
        self,
        content: Union[str, bytes, Path],
        *,
        model_config: Optional[ModelConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> RealtimeResponse:
        """Send message with completely custom configuration"""
        
        # Auto-detect content type and create appropriate message
        if isinstance(content, str):
            if Path(content).exists():
                message = RealtimeMessage.audio(content, **kwargs)
            else:
                message = RealtimeMessage.text(content, **kwargs)
        elif isinstance(content, bytes):
            message = RealtimeMessage.audio(content, **kwargs)
        elif isinstance(content, Path):
            message = RealtimeMessage.audio(content, **kwargs)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
        
        # Apply custom configurations
        if model_config:
            message.model_config = model_config
        if audio_config:
            message.audio_config = audio_config
        if retry_config:
            message.retry_config = retry_config
        
        return await self.send_message(message)
    
    # ========================================
    # Context Manager Support
    # ========================================
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    # ========================================
    # History and Analytics
    # ========================================
    
    def get_message_history(self) -> List[RealtimeMessage]:
        """Get all sent messages"""
        return self._message_history.copy()
    
    def get_response_history(self) -> List[RealtimeResponse]:
        """Get all received responses"""
        return self._response_history.copy()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self._response_history:
            return {}
        
        successful_responses = [r for r in self._response_history if r.success]
        
        return {
            "total_messages": len(self._message_history),
            "total_responses": len(self._response_history),
            "success_rate": len(successful_responses) / len(self._response_history),
            "avg_processing_time_ms": sum(r.processing_time_ms for r in successful_responses) / len(successful_responses) if successful_responses else 0,
            "total_retries": sum(r.retries_attempted for r in self._response_history),
            "models_used": list(set(r.model_used for r in successful_responses if r.model_used)),
            "voices_used": list(set(r.voice_used for r in successful_responses if r.voice_used))
        }


# ============================================================================
# USAGE EXAMPLES WITH UNIFIED MESSAGES
# ============================================================================

async def demonstrate_message_system():
    """Demonstrate the power of the unified message system"""
    
    print("🚀 Unified Message System Demonstration")
    print("=" * 60)
    
    service = MessageAwareRealtimeService("your-api-key")
    
    async with service:
        # Example 1: Simple text with custom config
        print("\n1️⃣ Text with custom voice and temperature")
        response = await service.send_text(
            "Tell me a joke",
            output_format="both",  # Want both text and audio
            voice="echo",
            temperature=0.9,
            timeout=15.0
        )
        print(f"   Response: {response.text}")
        print(f"   Has audio: {response.has_audio}")
        print(f"   Processing time: {response.processing_time_ms:.0f}ms")
        
        # Example 2: Audio with retry configuration
        print("\n2️⃣ Audio with custom retry policy")
        custom_retry = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            exponential_backoff=True
        )
        
        if Path("test_audio.wav").exists():
            response = await service.send_audio(
                "test_audio.wav",
                output_format="text",
                retry_config=custom_retry,
                priority="high"
            )
            print(f"   Transcription: {response.text}")
            print(f"   Retries used: {response.retries_attempted}")
        
        # Example 3: Completely custom message
        print("\n3️⃣ Completely custom message configuration")
        custom_message = service.create_custom_message(
            operation_name="storytelling",
            user_prompt="Tell me a short story about AI",
            system_prompt="You are a creative storyteller",
            output_data_format="both",
            model_config=ModelConfig(
                temperature=0.95,
                max_tokens=500
            ),
            audio_config=AudioConfig(
                voice="ballad",
                speed=0.8
            ),
            timeout=45.0,
            priority="high"
        )
        
        response = await service.send_message(custom_message)
        print(f"   Story: {response.text[:100]}...")
        print(f"   Voice used: {response.voice_used}")
        print(f"   Audio duration: {response.audio_duration_seconds:.1f}s")
        
        # Example 4: Function calling with different configs
        print("\n4️⃣ Function calling with configuration")
        
        # Register a function
        def get_weather(location: str, units: str = "fahrenheit") -> str:
            return f"Weather in {location}: 72°{units[0].upper()}, sunny"
        
        service.generation_engine.register_function(
            "get_weather", get_weather, 
            "Get weather for location",
            {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["fahrenheit", "celsius"]}
                },
                "required": ["location"]
            }
        )
        
        weather_response = await service.call_function(
            "get_weather",
            {"location": "Tokyo", "units": "celsius"},
            output_format="audio",  # Want spoken result
            voice="shimmer"
        )
        
        print(f"   Function result: {weather_response.text}")
        print(f"   Function calls: {len(weather_response.function_calls)}")
        
        # Example 5: Conversation with context
        print("\n5️⃣ Conversation statistics")
        stats = service.get_conversation_stats()