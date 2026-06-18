from dataclasses import dataclass
from typing import Optional


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    source: str                        # "rule", "bert", "llm"
    confidence: float = 1.0
    generalized: Optional[str] = None  # e.g. "45 år" -> "40-50 år"
    risk: str = "low"                  # "low", "medium", "high"
