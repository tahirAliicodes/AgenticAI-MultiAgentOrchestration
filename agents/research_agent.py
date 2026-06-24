import asyncio
from ddgs import DDGS
from agents.base_agent import BaseAgent
from models.messages import AgentMessage, AgentResult


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="research")

    def _build_system_prompt(self) -> str:
        return """You are a research agent. You receive raw search results and extract 
the most relevant facts. Be concise. Return only verified, useful information. 
Do not hallucinate — if search results don't cover something, say so.
Be concise. Return max 3 bullet points. No markdown headers. Plain text only."""

    async def run(self, message: AgentMessage) -> AgentResult:
        self.log(f"Researching: {message.task}")
        try:
            # Web search
            results = await asyncio.to_thread(self._search, message.task)

            # LLM distills the raw results
            prompt = f"""Task: {message.task}

Search results:
{results}

Extract the key facts relevant to the task. Be factual and concise."""

            summary = await asyncio.to_thread(self._llm, prompt)
            return AgentResult(agent_name=self.name, task=message.task,
                               result=summary, success=True)
        except Exception as e:
            return AgentResult(agent_name=self.name, task=message.task,
                               result="", success=False, error=str(e))

    def _search(self, query: str, max_results: int = 5) -> str:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"[{r['title']}]\n{r['body']}" for r in results
        )