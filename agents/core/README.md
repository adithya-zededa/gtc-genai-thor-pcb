# Core Agent Modules

This directory contains the core agent implementations for the camera monitoring system.

## Modules

### `proactive_agent.py` - Intelligent Proactive Monitoring Agent

**Purpose:** LLM-driven, context-aware camera monitoring that decides autonomously when and how to analyze frames.

**Key Features:**
- ✅ No hardcoded rules or SSIM thresholds
- ✅ LLM reasoning drives all decisions
- ✅ Temporal awareness across frames
- ✅ Natural language instruction parsing
- ✅ Intelligent action selection (wait, quick_check, full_inspection)
- ✅ Scene signature tracking to avoid redundancy

**Classes:**

#### `ProactiveMonitoringAgent`

The main intelligent monitoring agent.

**Initialization:**
```python
from agents.core.proactive_agent import ProactiveMonitoringAgent
from agents.vlm.client import UnifiedVLMClient
from agents.core.camera_agent import StreamlinedAgent

agent = ProactiveMonitoringAgent(
    instruction="Monitor the conveyor for PCB defects",
    vlm_client=vlm_client,
    detection_agent=detection_agent,
    publisher_getter=get_camera_publisher,
    event_callback=handle_detection_event,
    config={
        "frame_interval_seconds": 1.5,
        "decision_temperature": 0.2,
    }
)

agent.start()  # Start monitoring loop
```

**Configuration:**
- `frame_interval_seconds`: How often to process frames (guideline)
- `stability_frame_count`: Frames hint for scene stability (not a threshold)
- `observation_temperature`: LLM temp for observations (0.1 = consistent)
- `decision_temperature`: LLM temp for decisions (0.2 = balanced)
- `quick_check_temperature`: LLM temp for quick checks (0.15 = fast)
- `inspection_ttl_seconds`: How long to remember inspected scenes
- `max_idle_seconds`: Context hint for idle time awareness

#### `ActionType` Enum

Available actions the LLM can choose:

```python
class ActionType(str, Enum):
    WAIT = "wait"  # Continue monitoring
    QUICK_CHECK = "quick_check"  # Lightweight verification
    FULL_INSPECTION = "full_inspection"  # Comprehensive analysis
```

#### `ObservationResult` DataClass

Output from the LLM observation stage:

```python
@dataclass
class ObservationResult:
    frame_number: int
    timestamp: str
    scene_summary: str  # Natural language description
    primary_objects: List[str]  # Key objects in scene
    target_present: bool  # Is target visible?
    target_state: str  # entering/moving/stopped/stable/gone
    scene_changed: bool  # Meaningful change from last frame?
    scene_signature: str  # Consistent identifier for this scene
    target_ready: bool  # Ready for inspection?
    notes: str  # Additional context
    confidence: float  # 0.0 to 1.0
    raw_response: str  # Full LLM response
```

#### `DecisionResult` DataClass

Output from the LLM decision stage:

```python
@dataclass
class DecisionResult:
    action: ActionType  # wait/quick_check/full_inspection
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Why this action?
    analysis_plan: Dict[str, Any]  # How to execute if inspection
    scene_signature: str  # Scene identifier
    should_emit_event: bool  # Emit to subscribers?
    raw_response: str  # Full LLM response
```

#### `MonitoringContext` DataClass

Rich contextual state maintained across frames:

```python
@dataclass
class MonitoringContext:
    instruction: str  # User's objective
    frames_processed: int  # Total frames
    frames_since_scene_change: int  # Temporal stability
    last_action: ActionType  # Previous action
    last_action_time: float  # When
    last_observation: Optional[ObservationResult]  # Previous observations
    last_decision: Optional[DecisionResult]  # Previous decision
    last_quick_check: Optional[QuickCheckResult]  # Previous quick check
    inspected_signatures: Dict[str, float]  # Scene memory
    quick_check_count: int  # Statistics
    full_inspection_count: int  # Statistics
    target_ready_frames: int  # Stability tracking
```

**Methods:**

- `start()` - Start the monitoring loop
- `stop()` - Stop the monitoring loop
- `update_instruction(instruction: str)` - Update monitoring objective
- `snapshot() -> Dict` - Get current state snapshot
- `get_performance_metrics() -> Dict` - Get performance statistics

**Example Usage:**

```python
# Initialize agent
agent = ProactiveMonitoringAgent(
    instruction="Watch for packages without shipping labels",
    vlm_client=vlm_client,
    detection_agent=detection_agent,
    publisher_getter=get_camera_publisher,
    config={
        "frame_interval_seconds": 2.0,
        "decision_temperature": 0.25,
    }
)

# Start monitoring
agent.start()

# Check status
status = agent.snapshot()
print(f"Frames processed: {status['context']['frames_processed']}")
print(f"Inspections run: {status['context']['full_inspection_count']}")

# Get performance metrics
metrics = agent.get_performance_metrics()
print(f"Action rate: {metrics['action_rate']:.2%}")

# Update instruction on the fly
agent.update_instruction("Monitor for PCB defects instead")

# Stop when done
agent.stop()
```

### `camera_agent.py` - Streamlined Detection Agent

**Purpose:** Executes domain-specific vision analysis tasks (PCB inspection, package detection, PPE compliance, etc.)

Used by the ProactiveMonitoringAgent to run full inspections when the LLM decides conditions are optimal.

**Key Classes:**
- `StreamlinedAgent` - Main detection agent
- `CircuitBreaker` - Resilience pattern for VLM failures

### `state.py` - State Management

**Purpose:** State tracking and event definitions

**Key Classes:**
- `DetectionEvent` - Represents a detection event
- `AgentMemory` - Thread-safe ring buffer for event history
- `AgentSnapshot` - State snapshot for UIs

### `alerting.py` - Alert Management

**Purpose:** Alert dispatching and notification handling

**Key Classes:**
- `AlertManager` - Manages alert channels (email, webhook, etc.)

## Architecture Principles

### 1. LLM-Driven Decision Making

**The ProactiveMonitoringAgent follows a strict principle:**

```
🚫 NO hardcoded rules
🚫 NO SSIM thresholds  
🚫 NO fixed task switches

✅ LLM reasoning drives ALL decisions
```

### 2. Two-Stage Reasoning

The agent uses two distinct LLM stages:

1. **Observation** (cheap, fast) - Perceive scene state
2. **Decision** (intelligent) - Reason about optimal action

This separation enables:
- Efficient resource usage
- Clear reasoning traces
- Modular prompt engineering

### 3. Temporal Awareness

The agent maintains context across frames:
- Scene change detection
- Target state tracking (entering → moving → stable)
- Inspection memory (avoid redundancy)
- Action history

### 4. Natural Language Interface

Instructions are natural language, not code:

```python
# ✅ Good
"Monitor the conveyor for PCB defects"
"Watch for packages without shipping labels"
"Check PPE compliance for workers"

# ❌ Bad (trying to impose rules)
"If SSIM > 0.95 then skip"
"Analyze every 5th frame"
```

## Integration Example

Full integration with monitoring service:

```python
from services.core.monitoring import StreamlinedMonitoringService
from services.core.camera import get_camera_publisher

# Create monitoring service
service = StreamlinedMonitoringService()
service.initialize()

# Start proactive monitoring
success = service.start_proactive_monitoring(
    instruction="Monitor for PCB defects and inspect when boards stop",
    config={
        "frame_interval_seconds": 1.5,
        "stability_frame_count": 6,
        "decision_temperature": 0.2,
    }
)

if success:
    print("Proactive monitoring started")
    
    # Check status anytime
    status = service.get_proactive_snapshot()
    print(f"Running: {status['running']}")
    print(f"Frames: {status['context']['frames_processed']}")
    
# Stop when done
service.stop_proactive_monitoring()
```

## Testing

Unit tests for proactive agent:

```bash
pytest tests/unit/agents/test_proactive_agent.py -v
```

Integration tests:

```bash
pytest tests/integration/test_proactive_monitoring.py -v
```

## Further Reading

- [Proactive Monitoring Architecture](../../docs/architecture/proactive-monitoring.md)
- [Quick Start Guide](../../docs/development/proactive-monitoring-quickstart.md)
- [API Documentation](../../docs/api/endpoints.md)
- [VLM Prompts](../vlm/prompts.py) - Prompt engineering for observations and decisions

## Key Takeaways

1. **Intelligence First:** The LLM makes decisions, not hardcoded rules
2. **Context Matters:** Rich temporal state enables smart decisions
3. **Efficiency:** Proactive agents minimize unnecessary work
4. **Adaptability:** Natural language instructions = flexible behavior
5. **Observability:** Clear reasoning traces for debugging

The proactive agent represents a paradigm shift from reactive, rule-based monitoring to intelligent, context-aware analysis.
