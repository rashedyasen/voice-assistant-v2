from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class ActionType(Enum):
    CHAT = "chat"  # Just talking
    TOOL_USE = "tool_use"  # Needs to execute a function
    IGNORE = "ignore"  # Noise / incoherent


@dataclass
class ToolCall:
    tool: str
    params: Dict[str, Any]


@dataclass
class IntentResult:
    action_type: ActionType
    refined_query: str
    thought: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    confidence: float = 1.0
