import asyncio
from agents.base_agent import BaseAgent
from models.messages import AgentMessage, AgentResult

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="analysis")

    def _build_system_prompt(self) -> str:
        return """You are an analysis agent. You receive information and a question.
Your job: identify patterns, assess credibility, highlight gaps, and give a 
structured critique or breakdown. Think step by step. Be critical but fair.Be concise. Max 4 bullet points. No markdown headers. No bold text. Plain text only."""

    async def run(self, message: AgentMessage) -> AgentResult:
        self.log(f"Analyzing: {message.task}")
        try:
            context = message.payload.get("context", "")
            prompt = f"""Task: {message.task}

Context/data to analyze:
{context}

Provide a structured analysis. Include: key insights, limitations, and your confidence level."""

            result = await asyncio.to_thread(self._llm, prompt)
            return AgentResult(agent_name=self.name, task=message.task,
                               result=result, success=True)
        except Exception as e:
            return AgentResult(agent_name=self.name, task=message.task,
                               result="", success=False, error=str(e))