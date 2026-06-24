import asyncio
import json
from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.writer_agent import WriterAgent
from models.messages import AgentMessage, AgentResult, SupervisorPlan
from rich.console import Console
from rich.panel import Panel

console = Console()


class Supervisor(BaseAgent):
    def __init__(self):
        super().__init__(name="supervisor")
        self.agents = {
            "research": ResearchAgent(),
            "analysis": AnalysisAgent(),
            "writer": WriterAgent(),
        }

    def _build_system_prompt(self) -> str:
        return """You are a supervisor agent that coordinates a team of specialists.
Available agents:
- research: searches the web and finds facts
- analysis: critiques, compares, and reasons about information
- writer: produces polished final responses

Your job: given a user query, decide which agents are needed and in what order.
Respond ONLY with valid JSON matching this schema:
{
  "reasoning": "why you chose these agents",
  "agents_needed": ["research", "analysis", "writer"],
  "tasks": {
    "research": "specific instruction for research agent",
    "analysis": "specific instruction for analysis agent",
    "writer": "specific instruction for writer agent"
  },
  "final_synthesis_needed": true
}

Be minimal. Only add agents that are strictly necessary.
- Simple creative tasks (poems, jokes, rewrites) → writer only
- Factual questions → research + writer
- Complex analysis → research + analysis + writer

Only include agents that are actually needed."""

    async def run(self, message: AgentMessage) -> AgentResult:
        user_query = message.task
        console.print(Panel(f"[bold yellow]Supervisor received:[/bold yellow] {user_query}"))

        plan = await self._plan(user_query)
        console.print(f"[bold yellow][Supervisor][/bold yellow] Plan: {plan.agents_needed}")
        console.print(f"[dim]Reasoning: {plan.reasoning}[/dim]")

        results: dict[str, AgentResult] = {}
        accumulated_context = ""

        for agent_name in plan.agents_needed:
            agent = self.agents[agent_name]
            task_instruction = plan.tasks.get(agent_name, user_query)

            msg = AgentMessage(
                from_agent="supervisor",
                to_agent=agent_name,
                task=task_instruction,
                payload={
                    "context": accumulated_context,
                    "all_results": accumulated_context,
                    "original_query": user_query
                }
            )

            result = await agent.run(msg)
            results[agent_name] = result

            if result.success:
                accumulated_context += f"\n\n[{agent_name.upper()} AGENT]:\n{result.result}"
            else:
                console.print(f"[red][{agent_name}] failed: {result.error}[/red]")

        final_result = (
            results.get("writer") or
            next((r for r in reversed(list(results.values())) if r.success), None)
        )

        if not final_result:
            return AgentResult(agent_name="supervisor", task=user_query,
                               result="All agents failed to produce a result.", success=False)

        return AgentResult(agent_name="supervisor", task=user_query,
                           result=final_result.result, success=True)

    async def stream_orchestrate(self, query: str):
        """SSE generator — yields events as each agent runs."""

        yield f"data: {json.dumps({'agent': 'supervisor', 'status': 'planning', 'msg': f'Planning for: {query}'})}\n\n"

        plan = await self._plan(query)

        yield f"data: {json.dumps({'agent': 'supervisor', 'status': 'plan_ready', 'msg': f'Agents: {plan.agents_needed}', 'reasoning': plan.reasoning})}\n\n"

        results: dict[str, AgentResult] = {}
        accumulated_context = ""

        for agent_name in plan.agents_needed:
            agent = self.agents[agent_name]
            task_instruction = plan.tasks.get(agent_name, query)

            yield f"data: {json.dumps({'agent': agent_name, 'status': 'started', 'msg': task_instruction})}\n\n"

            msg = AgentMessage(
                from_agent="supervisor",
                to_agent=agent_name,
                task=task_instruction,
                payload={
                    "context": accumulated_context,
                    "all_results": accumulated_context,
                    "original_query": query
                }
            )

            result = await agent.run(msg)
            results[agent_name] = result

            if result.success:
                accumulated_context += f"\n\n[{agent_name.upper()} AGENT]:\n{result.result}"
                yield f"data: {json.dumps({'agent': agent_name, 'status': 'done', 'msg': result.result[:150]})}\n\n"
            else:
                yield f"data: {json.dumps({'agent': agent_name, 'status': 'error', 'msg': result.error})}\n\n"

        final_result = (
            results.get("writer") or
            next((r for r in reversed(list(results.values())) if r.success), None)
        )

        if final_result:
            yield f"data: {json.dumps({'agent': 'supervisor', 'status': 'final', 'msg': final_result.result})}\n\n"
        else:
            yield f"data: {json.dumps({'agent': 'supervisor', 'status': 'error', 'msg': 'All agents failed'})}\n\n"

    async def _plan(self, query: str) -> SupervisorPlan:
        raw = await asyncio.to_thread(self._llm, query)

        try:
            clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(clean)
            return SupervisorPlan(**data)
        except Exception:
            console.print("[yellow][Supervisor] Plan parsing failed, using default.[/yellow]")
            return SupervisorPlan(
                reasoning="Fallback: couldn't parse LLM plan",
                agents_needed=["research", "writer"],
                tasks={"research": query, "writer": query}
            )