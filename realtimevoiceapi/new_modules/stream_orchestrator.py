"""
Stream Orchestrator - Big Lane Component

Coordinates multiple streams, providers, and complex workflows.
Handles failover, load balancing, and stream composition.



stream_orchestrator.py:

Multi-Provider: Manages streams across different providers
Automatic Failover: Detects failures and switches streams
Load Balancing: Multiple strategies for distributing load
Workflow Engine: Execute complex multi-step workflows
Health Monitoring: Continuous health checks
Capability Routing: Route based on provider features
Event-Driven: Fully integrated with event bus



"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

from ..event_bus import EventBus, Event
from ..stream_protocol import (
    IStreamManager, StreamState, StreamConfig,
    StreamCapabilities, StreamEvent, StreamEventType
)
from ..provider_protocol import (
    IVoiceProvider, ProviderRegistry, ProviderFeature,
    ProviderEvent, ProviderEventType
)
from ..session_manager import SessionManager, Session, SessionState


class StreamRole(Enum):
    """Role of a stream in orchestration"""
    PRIMARY = "primary"
    BACKUP = "backup"
    AUXILIARY = "auxiliary"
    MONITOR = "monitor"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for multiple streams"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LATENCY_BASED = "latency_based"
    COST_OPTIMIZED = "cost_optimized"
    FEATURE_BASED = "feature_based"


@dataclass
class StreamInstance:
    """Active stream instance"""
    id: str
    manager: IStreamManager
    provider: str
    role: StreamRole
    config: StreamConfig
    created_at: float = field(default_factory=time.time)
    
    # Metrics
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    latency_ms: float = 0.0
    cost_estimate: float = 0.0
    
    # State
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)
    capabilities: Optional[StreamCapabilities] = None


@dataclass
class WorkflowStep:
    """Step in a stream workflow"""
    name: str
    provider: Optional[str] = None  # None = any available
    required_features: List[ProviderFeature] = field(default_factory=list)
    timeout_ms: int = 30000
    retry_count: int = 3
    fallback_step: Optional[str] = None
    
    # Processing function
    process_func: Optional[Callable] = None
    
    # Conditions
    condition_func: Optional[Callable] = None


class StreamOrchestrator:
    """
    Orchestrates multiple streams for complex scenarios.
    
    Features:
    - Multi-provider coordination
    - Automatic failover
    - Load balancing
    - Stream composition
    - Workflow management
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        provider_registry: Optional[ProviderRegistry] = None,
        session_manager: Optional[SessionManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.event_bus = event_bus
        self.provider_registry = provider_registry or ProviderRegistry()
        self.session_manager = session_manager or SessionManager()
        self.logger = logger or logging.getLogger(__name__)
        
        # Active streams
        self.streams: Dict[str, StreamInstance] = {}
        self.provider_streams: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.max_streams_per_provider = 5
        self.health_check_interval = 30.0  # seconds
        self.failover_timeout = 5.0  # seconds
        
        # Load balancing
        self.load_strategy = LoadBalancingStrategy.LATENCY_BASED
        self.round_robin_index: Dict[str, int] = defaultdict(int)
        
        # Workflows
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Health monitoring
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Subscribe to events
        self._setup_subscriptions()
    
    def _setup_subscriptions(self):
        """Setup event subscriptions"""
        # Stream events
        self.event_bus.subscribe("stream.*", self._handle_stream_event)
        
        # Provider events
        self.event_bus.subscribe("provider.*", self._handle_provider_event)
        
        # Error events for failover
        self.event_bus.subscribe("*.error", self._handle_error_event)
    
    async def start(self):
        """Start orchestrator"""
        if self._is_running:
            return
        
        self._is_running = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        self.logger.info("Stream orchestrator started")
    
    async def stop(self):
        """Stop orchestrator"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Stop health monitor
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close all streams
        for stream_id in list(self.streams.keys()):
            await self.close_stream(stream_id)
        
        self.logger.info("Stream orchestrator stopped")
    
    async def create_stream(
        self,
        provider: Optional[str] = None,
        role: StreamRole = StreamRole.PRIMARY,
        config: Optional[StreamConfig] = None,
        required_features: Optional[List[ProviderFeature]] = None
    ) -> str:
        """
        Create a new stream with automatic provider selection.
        
        Returns:
            Stream ID
        """
        # Select provider if not specified
        if not provider:
            provider = await self._select_provider(required_features)
            if not provider:
                raise ValueError("No suitable provider found")
        
        # Check provider limits
        if len(self.provider_streams[provider]) >= self.max_streams_per_provider:
            raise ValueError(f"Provider {provider} has reached stream limit")
        
        # Get provider instance
        provider_instance = self.provider_registry.get(provider)
        
        # Create stream config
        if not config:
            config = StreamConfig(
                provider=provider,
                mode="both",
                metadata={}
            )
        
        # Create stream manager through provider
        # This is simplified - in reality would create through provider
        stream_manager = await self._create_stream_manager(
            provider_instance,
            config
        )
        
        # Create stream instance
        stream_id = f"stream_{uuid.uuid4().hex[:8]}"
        stream = StreamInstance(
            id=stream_id,
            manager=stream_manager,
            provider=provider,
            role=role,
            config=config,
            capabilities=provider_instance.get_capabilities()
        )
        
        # Register stream
        self.streams[stream_id] = stream
        self.provider_streams[provider].add(stream_id)
        
        # Start stream
        await stream_manager.start()
        
        # Emit event
        await self.event_bus.emit(Event(
            type="orchestrator.stream.created",
            data={
                "stream_id": stream_id,
                "provider": provider,
                "role": role.value
            }
        ))
        
        self.logger.info(f"Created stream {stream_id} with provider {provider}")
        
        return stream_id
    
    async def close_stream(self, stream_id: str):
        """Close a stream"""
        if stream_id not in self.streams:
            return
        
        stream = self.streams[stream_id]
        
        try:
            # Stop stream manager
            await stream.manager.stop()
            
            # Remove from registry
            del self.streams[stream_id]
            self.provider_streams[stream.provider].discard(stream_id)
            
            # Emit event
            await self.event_bus.emit(Event(
                type="orchestrator.stream.closed",
                data={
                    "stream_id": stream_id,
                    "provider": stream.provider
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Error closing stream {stream_id}: {e}")
    
    async def send_to_stream(
        self,
        stream_id: str,
        data: Any,
        data_type: str = "audio"
    ):
        """Send data to specific stream"""
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        stream = self.streams[stream_id]
        
        try:
            if data_type == "audio":
                await stream.manager.send_audio(data)
            elif data_type == "text":
                await stream.manager.send_text(data)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # Update metrics
            stream.messages_sent += 1
            
        except Exception as e:
            stream.errors += 1
            stream.is_healthy = False
            raise
    
    async def broadcast(
        self,
        data: Any,
        data_type: str = "audio",
        role_filter: Optional[StreamRole] = None
    ):
        """Broadcast data to multiple streams"""
        target_streams = [
            s for s in self.streams.values()
            if not role_filter or s.role == role_filter
        ]
        
        results = []
        for stream in target_streams:
            try:
                await self.send_to_stream(stream.id, data, data_type)
                results.append((stream.id, True, None))
            except Exception as e:
                results.append((stream.id, False, str(e)))
        
        return results
    
    async def route_by_capability(
        self,
        data: Any,
        required_feature: ProviderFeature
    ) -> Optional[str]:
        """Route data to stream with specific capability"""
        # Find capable streams
        capable_streams = [
            s for s in self.streams.values()
            if s.capabilities and required_feature in s.capabilities.features
            and s.is_healthy
        ]
        
        if not capable_streams:
            return None
        
        # Select based on load balancing strategy
        selected = self._select_stream_by_strategy(capable_streams)
        
        if selected:
            await self.send_to_stream(selected.id, data)
            return selected.id
        
        return None
    
    def _select_stream_by_strategy(
        self,
        streams: List[StreamInstance]
    ) -> Optional[StreamInstance]:
        """Select stream based on load balancing strategy"""
        if not streams:
            return None
        
        if self.load_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round-robin
            key = "global"
            index = self.round_robin_index[key] % len(streams)
            self.round_robin_index[key] += 1
            return streams[index]
        
        elif self.load_strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select stream with fewest messages
            return min(streams, key=lambda s: s.messages_sent)
        
        elif self.load_strategy == LoadBalancingStrategy.LATENCY_BASED:
            # Select stream with lowest latency
            healthy_streams = [s for s in streams if s.is_healthy]
            if healthy_streams:
                return min(healthy_streams, key=lambda s: s.latency_ms)
            return streams[0]
        
        elif self.load_strategy == LoadBalancingStrategy.COST_OPTIMIZED:
            # Select cheapest stream
            return min(streams, key=lambda s: s.cost_estimate)
        
        else:
            # Default to first available
            return streams[0]
    
    async def _select_provider(
        self,
        required_features: Optional[List[ProviderFeature]] = None
    ) -> Optional[str]:
        """Select best provider based on requirements"""
        if not required_features:
            # Return default provider
            providers = self.provider_registry.list_providers()
            return providers[0] if providers else None
        
        # Find providers with all required features
        suitable_providers = []
        
        for provider_name in self.provider_registry.list_providers():
            provider = self.provider_registry.get(provider_name)
            capabilities = provider.get_capabilities()
            
            if all(f in capabilities.features for f in required_features):
                suitable_providers.append(provider_name)
        
        if not suitable_providers:
            return None
        
        # Select based on current load
        min_load = float('inf')
        selected = None
        
        for provider in suitable_providers:
            load = len(self.provider_streams[provider])
            if load < min_load:
                min_load = load
                selected = provider
        
        return selected
    
    async def failover_stream(
        self,
        failed_stream_id: str,
        reason: str = "unknown"
    ) -> Optional[str]:
        """Failover from failed stream to backup"""
        if failed_stream_id not in self.streams:
            return None
        
        failed_stream = self.streams[failed_stream_id]
        
        # Emit failover event
        await self.event_bus.emit(Event(
            type="orchestrator.failover.started",
            data={
                "failed_stream": failed_stream_id,
                "provider": failed_stream.provider,
                "reason": reason
            }
        ))
        
        # Find backup stream
        backup_streams = [
            s for s in self.streams.values()
            if s.role == StreamRole.BACKUP
            and s.provider != failed_stream.provider
            and s.is_healthy
        ]
        
        if backup_streams:
            # Use existing backup
            backup = backup_streams[0]
            backup.role = StreamRole.PRIMARY
            
            self.logger.info(
                f"Failover: Using existing backup {backup.id} "
                f"for failed stream {failed_stream_id}"
            )
            
            return backup.id
        
        # Create new stream with different provider
        try:
            # Get alternative providers
            all_providers = set(self.provider_registry.list_providers())
            alternative_providers = all_providers - {failed_stream.provider}
            
            if not alternative_providers:
                self.logger.error("No alternative providers for failover")
                return None
            
            # Create new stream
            new_stream_id = await self.create_stream(
                provider=list(alternative_providers)[0],
                role=StreamRole.PRIMARY,
                config=failed_stream.config
            )
            
            self.logger.info(
                f"Failover: Created new stream {new_stream_id} "
                f"to replace {failed_stream_id}"
            )
            
            # Close failed stream
            await self.close_stream(failed_stream_id)
            
            return new_stream_id
            
        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            return None
    
    # ============== Workflow Management ==============
    
    def register_workflow(
        self,
        name: str,
        steps: List[WorkflowStep]
    ):
        """Register a workflow"""
        self.workflows[name] = steps
        self.logger.info(f"Registered workflow '{name}' with {len(steps)} steps")
    
    async def execute_workflow(
        self,
        workflow_name: str,
        initial_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        steps = self.workflows[workflow_name]
        
        # Initialize workflow state
        workflow_state = {
            "id": workflow_id,
            "name": workflow_name,
            "data": initial_data,
            "context": context or {},
            "current_step": 0,
            "results": {},
            "errors": []
        }
        
        self.active_workflows[workflow_id] = workflow_state
        
        # Emit start event
        await self.event_bus.emit(Event(
            type="orchestrator.workflow.started",
            data={"workflow_id": workflow_id, "name": workflow_name}
        ))
        
        try:
            # Execute steps
            for i, step in enumerate(steps):
                workflow_state["current_step"] = i
                
                # Check condition
                if step.condition_func:
                    if not await step.condition_func(workflow_state):
                        self.logger.debug(f"Skipping step {step.name} - condition not met")
                        continue
                
                # Execute step
                try:
                    result = await self._execute_workflow_step(
                        step,
                        workflow_state
                    )
                    workflow_state["results"][step.name] = result
                    
                except Exception as e:
                    workflow_state["errors"].append({
                        "step": step.name,
                        "error": str(e)
                    })
                    
                    # Try fallback
                    if step.fallback_step:
                        self.logger.warning(
                            f"Step {step.name} failed, trying fallback {step.fallback_step}"
                        )
                        # Find and execute fallback step
                        fallback = next(
                            (s for s in steps if s.name == step.fallback_step),
                            None
                        )
                        if fallback:
                            result = await self._execute_workflow_step(
                                fallback,
                                workflow_state
                            )
                            workflow_state["results"][fallback.name] = result
            
            # Emit completion event
            await self.event_bus.emit(Event(
                type="orchestrator.workflow.completed",
                data={
                    "workflow_id": workflow_id,
                    "results": workflow_state["results"]
                }
            ))
            
            return workflow_state
            
        except Exception as e:
            # Emit error event
            await self.event_bus.emit(Event(
                type="orchestrator.workflow.error",
                data={
                    "workflow_id": workflow_id,
                    "error": str(e)
                }
            ))
            raise
        
        finally:
            # Cleanup
            del self.active_workflows[workflow_id]
    
    async def _execute_workflow_step(
        self,
        step: WorkflowStep,
        workflow_state: Dict[str, Any]
    ) -> Any:
        """Execute a single workflow step"""
        self.logger.debug(f"Executing workflow step: {step.name}")
        
        # Select stream based on requirements
        stream_id = None
        
        if step.provider:
            # Find stream for specific provider
            provider_streams = [
                s for s in self.streams.values()
                if s.provider == step.provider and s.is_healthy
            ]
            if provider_streams:
                stream_id = provider_streams[0].id
        
        elif step.required_features:
            # Find stream with required features
            for feature in step.required_features:
                stream_id = await self.route_by_capability(
                    workflow_state["data"],
                    feature
                )
                if stream_id:
                    break
        
        if not stream_id:
            # Use any available stream
            healthy_streams = [s for s in self.streams.values() if s.is_healthy]
            if healthy_streams:
                stream_id = healthy_streams[0].id
        
        if not stream_id:
            raise RuntimeError(f"No suitable stream for step {step.name}")
        
        # Execute step function
        if step.process_func:
            result = await step.process_func(
                stream_id=stream_id,
                data=workflow_state["data"],
                context=workflow_state["context"],
                orchestrator=self
            )
            return result
        
        return None
    
    # ============== Health Monitoring ==============
    
    async def _health_monitor_loop(self):
        """Monitor stream health"""
        while self._is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check each stream
                for stream in list(self.streams.values()):
                    try:
                        await self._check_stream_health(stream)
                    except Exception as e:
                        self.logger.error(
                            f"Health check failed for stream {stream.id}: {e}"
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
    
    async def _check_stream_health(self, stream: StreamInstance):
        """Check health of a single stream"""
        # Update last check time
        stream.last_health_check = time.time()
        
        # Check stream state
        state = stream.manager.get_state()
        
        if state == StreamState.ERROR:
            stream.is_healthy = False
            
            # Trigger failover
            await self.failover_stream(stream.id, "stream_error_state")
            
        elif state == StreamState.ACTIVE:
            # Additional health checks
            metrics = stream.manager.get_metrics()
            
            # Check error rate
            if metrics.get("error_rate", 0) > 0.1:  # 10% error rate
                stream.is_healthy = False
                
            # Update latency
            stream.latency_ms = metrics.get("latency_ms", 0)
            
        else:
            # Stream not active
            stream.is_healthy = False
    
    # ============== Event Handlers ==============
    
    async def _handle_stream_event(self, event: Event):
        """Handle stream events"""
        # Update metrics based on events
        stream_id = event.data.get("stream_id")
        
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            
            if event.type == "stream.message.sent":
                stream.messages_sent += 1
            elif event.type == "stream.message.received":
                stream.messages_received += 1
    
    async def _handle_provider_event(self, event: Event):
        """Handle provider events"""
        if event.type == ProviderEventType.RATE_LIMIT_EXCEEDED:
            # Handle rate limiting
            provider = event.data.get("provider")
            
            # Find alternative streams
            for stream in self.streams.values():
                if stream.provider == provider:
                    await self.failover_stream(
                        stream.id,
                        "rate_limit_exceeded"
                    )
    
    async def _handle_error_event(self, event: Event):
        """Handle error events"""
        # Check if it's a stream error
        stream_id = event.data.get("stream_id")
        
        if stream_id in self.streams:
            stream = self.streams[stream_id]
            stream.errors += 1
            
            # Mark unhealthy if too many errors
            if stream.errors > 10:
                stream.is_healthy = False
                await self.failover_stream(stream_id, "too_many_errors")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            "total_streams": len(self.streams),
            "healthy_streams": sum(1 for s in self.streams.values() if s.is_healthy),
            "providers_in_use": list(self.provider_streams.keys()),
            "active_workflows": len(self.active_workflows),
            "load_strategy": self.load_strategy.value,
            "stream_metrics": {
                s.id: {
                    "provider": s.provider,
                    "role": s.role.value,
                    "is_healthy": s.is_healthy,
                    "messages_sent": s.messages_sent,
                    "messages_received": s.messages_received,
                    "errors": s.errors,
                    "latency_ms": s.latency_ms
                }
                for s in self.streams.values()
            }
        }
    
    async def _create_stream_manager(
        self,
        provider: IVoiceProvider,
        config: StreamConfig
    ) -> IStreamManager:
        """Create stream manager for provider (simplified)"""
        # This would actually create the appropriate stream manager
        # based on the provider type
        # For now, return a mock
        
        class MockStreamManager:
            def __init__(self):
                self.state = StreamState.IDLE
                
            async def start(self):
                self.state = StreamState.ACTIVE
                
            async def stop(self):
                self.state = StreamState.ENDED
                
            async def send_audio(self, data):
                pass
                
            async def send_text(self, data):
                pass
                
            def get_state(self):
                return self.state
                
            def get_metrics(self):
                return {"error_rate": 0.0, "latency_ms": 50}
        
        return MockStreamManager()