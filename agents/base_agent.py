"""Abstract base class for the 4-agent defense pipeline."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Decision:
    agent_name: str
    action: str
    reasoning: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base_agent"

    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """Consume the previous agent's output, produce this agent's output."""
        raise NotImplementedError
