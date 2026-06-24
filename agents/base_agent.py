from abc import ABC, abstractmethod
from models.messages import AgentMessage, AgentResult
import ollama
from rich.console import Console

console = Console()

class BaseAgent(ABC):
    def __init__(self, name: str, model: str = "llama3.1"):
        self.name = name
        self.model = model
        self.system_prompt = self._build_system_prompt()

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Each agent defines its own identity and rules."""
        ...

    @abstractmethod
    async def run(self, message: AgentMessage) -> AgentResult:
        """Execute the task in the message."""
        ...

    def _llm(self, prompt: str, system: str | None = None) -> str:
        """Synchronous Ollama call — wrap in asyncio.to_thread if needed."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system or self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        return response["message"]["content"]

    def log(self, msg: str):
        console.print(f"[bold cyan][{self.name}][/bold cyan] {msg}")