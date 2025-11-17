from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import os
import uuid
from contextlib import asynccontextmanager

from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient
from my_tools import deploy_bedrock_flow_stack, invoke_bedrock_flow

from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

ACTOR_ID = "bedrock-flow-api"
AGENTCORE_MEMORY_ID = "strands_agent_memory-GI7k3lEiET"
AGENTCORE_REGION = "us-west-2"
AGENTCORE_NAMESPACE = f"/bedrock-flow/actors/{ACTOR_ID}/errors"

# Global state
agent = None
mcp_clients = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global agent, mcp_clients
    
    SESSION_ID = f"api-{uuid.uuid4()}"
    
    memory_provider = AgentCoreMemoryToolProvider(
        memory_id=AGENTCORE_MEMORY_ID,
        actor_id=ACTOR_ID,
        session_id=SESSION_ID,
        namespace=AGENTCORE_NAMESPACE,
        region=AGENTCORE_REGION,
    )
    
    memory_config = AgentCoreMemoryConfig(
        memory_id=AGENTCORE_MEMORY_ID,
        session_id=SESSION_ID,
        actor_id=ACTOR_ID,
        retrieval_config={
            AGENTCORE_NAMESPACE: RetrievalConfig(top_k=5, relevance_score=0.5),
        },
    )
    
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config,
        region_name=AGENTCORE_REGION,
    )
    
    # Initialize MCP clients (REMOVED AWS_PROFILE)
    stdio_documentation_client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx",
                args=[
                    "--from",
                    "awslabs.aws-documentation-mcp-server@latest",
                    "awslabs.aws-documentation-mcp-server",
                ],
            )
        )
    )
    
    stdio_cdk_client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx",
                args=[
                    "--from",
                    "awslabs.cdk-mcp-server@latest",
                    "awslabs.cdk-mcp-server",
                ],
            )
        )
    )
    
    stdio_cfn_client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx",
                args=[
                    "--from",
                    "awslabs.cfn-mcp-server@latest",
                    "awslabs.cfn-mcp-server",
                ],
            )
        )
    )
    
    mcp_clients = [stdio_documentation_client, stdio_cdk_client, stdio_cfn_client]
    
    # Enter context managers
    for client in mcp_clients:
        client.__enter__()
    
    tools = (
        stdio_documentation_client.list_tools_sync()
        + stdio_cdk_client.list_tools_sync()
        + stdio_cfn_client.list_tools_sync()
        + [deploy_bedrock_flow_stack, invoke_bedrock_flow]
        + memory_provider.tools
    )
    
    agent = Agent(
        tools=tools,
        callback_handler=None,
        session_manager=session_manager,
        system_prompt="""[Your system prompt here]"""
    )
    
    print("✓ Agent initialized")
    
    yield
    
    # Shutdown
    for client in mcp_clients:
        try:
            client.__exit__(None, None, None)
        except:
            pass

app = FastAPI(lifespan=lifespan)

class InvokeRequest(BaseModel):
    prompt: str
    session_id: str | None = None

class InvokeResponse(BaseModel):
    response: str
    session_id: str

@app.post("/invoke", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    session_id = request.session_id or f"api-{uuid.uuid4()}"
    
    full_text_chunks = []
    invocation_state = {
        "user_id": ACTOR_ID,
        "confirm_tool_calls": False,
    }
    
    async for event in agent.stream_async(request.prompt, invocation_state=invocation_state):
        if "data" in event:
            full_text_chunks.append(event["data"])
    
    return InvokeResponse(
        response="".join(full_text_chunks),
        session_id=session_id
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "agent_ready": agent is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)