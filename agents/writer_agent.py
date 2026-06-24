import asyncio
from agents.base_agent import BaseAgent
from models.messages import AgentMessage, AgentResult

class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="writer")

    def _build_system_prompt(self) -> str:
        return """You are a writer agent. You take research findings and analysis 
and produce a clear, well-structured final response for the user. 
Write in plain English. Use headers and bullets where helpful. Be complete but not verbose.
Be concise. Max 1 short paragraph (5 sentences max). No markdown headers. No bold text."""

    async def run(self, message: AgentMessage) -> AgentResult:
        self.log(f"Writing response for: {message.task}")
        try:
            inputs = message.payload.get("all_results", "")
            prompt = f"""Original user question: {message.task}

Agent findings:
{inputs}

Write a clear, complete final answer for the user based on these findings."""

            result = await asyncio.to_thread(self._llm, prompt)
            return AgentResult(agent_name=self.name, task=message.task,
                               result=result, success=True)
        except Exception as e:
            return AgentResult(agent_name=self.name, task=message.task,
                               result="", success=False, error=str(e))