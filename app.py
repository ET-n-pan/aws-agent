from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import asyncio
import os
import uuid
import json
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    # Initialize MCP clients
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
        system_prompt="""
            You are a Bedrock Flow and CloudFormation expert that helps the user design, deploy, and debug Bedrock Flow stacks.
        """
    )
    
    logger.info("✓ Agent initialized")
    
    yield
    
    # Shutdown
    for client in mcp_clients:
        try:
            client.__exit__(None, None, None)
        except:
            pass

app = FastAPI(lifespan=lifespan)

# Log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    logger.info(f"Headers: {dict(request.headers)}")
    
    # Read body
    body = await request.body()
    logger.info(f"Body: {body.decode() if body else 'empty'}")
    
    # Create new request with body
    async def receive():
        return {"type": "http.request", "body": body}
    
    request._receive = receive
    
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/invocations")
async def invocations(request: Request):
    """Handle invocations - accepts multiple input formats"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Parse request body
        body = await request.body()
        logger.info(f"Raw body: {body}")
        
        data = json.loads(body)
        logger.info(f"Parsed data: {data}")
        
        # Extract prompt from various possible formats
        prompt = None
        session_id = None
        
        # Try different field names
        if isinstance(data, dict):
            prompt = (
                data.get("prompt") or 
                data.get("inputText") or 
                data.get("input") or 
                data.get("text") or
                data.get("query")
            )
            session_id = data.get("session_id") or data.get("sessionId")
        elif isinstance(data, str):
            prompt = data
        
        if not prompt:
            logger.error(f"Could not extract prompt from: {data}")
            raise HTTPException(
                status_code=422, 
                detail=f"Missing prompt field. Received: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )
        
        logger.info(f"Processing prompt: {prompt[:100]}...")
        
        session_id = session_id or f"api-{uuid.uuid4()}"
        
        # Run agent
        full_text_chunks = []
        invocation_state = {
            "user_id": ACTOR_ID,
            "confirm_tool_calls": False,
        }
        
        async for event in agent.stream_async(prompt, invocation_state=invocation_state):
            if "data" in event:
                full_text_chunks.append(event["data"])
        
        response_text = "".join(full_text_chunks)
        logger.info(f"Response generated: {len(response_text)} chars")
        
        # Return in multiple formats for compatibility
        return {
            "response": response_text,
            "completion": response_text,
            "output": response_text,
            "session_id": session_id,
            "sessionId": session_id
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent_ready": agent is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)