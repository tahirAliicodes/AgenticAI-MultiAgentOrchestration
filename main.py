import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.thesupervisor_agent import Supervisor
from models.messages import AgentMessage
from rich.console import Console

app = FastAPI(title="Stage 6 — Multi-Agent Orchestration")
console = Console()

supervisor = Supervisor()

class UserQuery(BaseModel):
    query: str

@app.post("/orchestrate")
async def orchestrate(request: UserQuery):
    msg = AgentMessage(
        from_agent="user",
        to_agent="supervisor",
        task=request.query
    )
    result = await supervisor.run(msg)
    return {
        "query": request.query,
        "success": result.success,
        "result": result.result,
        "agent": result.agent_name
    }

@app.get("/orchestrate/stream")
async def orchestrate_stream(query: str):
    return StreamingResponse(
        supervisor.stream_orchestrate(query),
        media_type="text/event-stream"
    )

@app.get("/agents")
async def list_agents():
    return {"agents": list(supervisor.agents.keys()), "supervisor": "active"}