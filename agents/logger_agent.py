"""Agent 4 — Logger: append-only audit trail of every agent decision."""
import json
import os
import time
from typing import Any, Dict, List, Optional

import config
from agents.base_agent import BaseAgent, Decision


def _json_safe(obj: Any) -> Any:
    """Strip non-JSON-serializable payloads (e.g. raw numpy update vectors)
    out of decision metadata before logging."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if k != "update"}
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


class LoggerAgent(BaseAgent):
    name = "logger_agent"

    def __init__(self, log_path: str = config.AUDIT_LOG_PATH):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def process(self, decision: Decision, round_number: int = 0, client_id: Optional[int] = None, **kwargs) -> None:
        self.log_decision(decision, round_number, client_id)

    def log_decision(self, decision: Decision, round_number: int, client_id: Optional[int] = None) -> None:
        entry = {
            "timestamp": time.time(),
            "round_number": round_number,
            "agent_name": decision.agent_name,
            "client_id": client_id,
            "action": decision.action,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "metadata": _json_safe(decision.metadata),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_logs(self, round_number: Optional[int] = None, agent_name: Optional[str] = None) -> List[Dict]:
        entries = self.get_all_logs()
        if round_number is not None:
            entries = [e for e in entries if e["round_number"] == round_number]
        if agent_name is not None:
            entries = [e for e in entries if e["agent_name"] == agent_name]
        return entries

    def get_all_logs(self) -> List[Dict]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]
