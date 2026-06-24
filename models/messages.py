from pydantic import BaseModel, Field
from typing import Any, Literal
from datetime import datetime
import uuid

class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    from_agent: str
    to_agent: str
    task: str
    payload: dict[str, Any] = {}
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class AgentResult(BaseModel):
    agent_name: str
    task: str
    result: str
    success: bool
    error: str | None = None

class SupervisorPlan(BaseModel):
    """Structured output from supervisor's planning step."""
    reasoning: str
    agents_needed: list[Literal["research", "analysis", "writer"]]
    tasks: dict[str, str]  # agent_name -> specific task instruction
    final_synthesis_needed: bool = True