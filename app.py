from fastapi import FastAPI, HTTPException, Request
import uuid
import json
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
    
    # stdio_cdk_client = MCPClient(
    #     lambda: stdio_client(
    #         StdioServerParameters(
    #             command="uvx",
    #             args=[
    #                 "--from",
    #                 "awslabs.cdk-mcp-server@latest",
    #                 "awslabs.cdk-mcp-server",
    #             ],
    #         )
    #     )
    # )
    
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
    
    mcp_clients = [stdio_documentation_client, stdio_cfn_client]
    
    # Enter context managers
    for client in mcp_clients:
        client.__enter__()
    
    tools = (
        stdio_documentation_client.list_tools_sync()
        + stdio_cfn_client.list_tools_sync()
        + [deploy_bedrock_flow_stack, invoke_bedrock_flow]
        + memory_provider.tools
    )
    
    agent = Agent(
        tools=tools,
        callback_handler=None,
        session_manager=session_manager,
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        system_prompt="""You are a Bedrock Flow expert.

            TOOLS YOU HAVE:
            - MCP documentation tools for AWS docs.
            - `agent_core_memory`: store / retrieve structured notes in Amazon Bedrock AgentCore Memory.

            AT THE START OF EACH CONVERSATION:
            1. If the user's request is about Bedrock Flow or CFN templates, first call:
            - `agent_core_memory(action="retrieve", query="<task>", top_k=5)`
            to see if relevant notes / patterns already exist.

            ERROR HANDLING & MEMORY:
            1. When you see a deployment or validation error:
            - Summarize it into a short error key (e.g. "CFN ROLLBACK: Parameter X missing").
            - Call `agent_core_memory(action="retrieve", query="<that key>", top_k=5)`
                to see if we've solved it before.

            2. If you solve a NEW error:
            - Create a concise JSON note including:
                - error_message
                - root_cause
                - final_fix (code snippet / template fragment)
                - related_service (e.g. "bedrock-flow", "cloudformation")
            - Call:
                `agent_core_memory(
                    action="record",
                    content="<that JSON as a string>",
                    label="<short error key>"
                )`

            DOCS USAGE:
            - For up-to-date API shapes and behaviors, always trust MCP AWS documentation servers first.
            - Use AgentCore Memory mainly for:
            - recurring errors and their fixes
            - stable patterns / best practices, not entire docs dumps.
        """
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

@app.post("/invocations")
async def invocations(request: Request):
    """Handle invocations - accepts multiple input formats"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Parse request body
        body = await request.body()
        print(f"Raw body: {body}")
        
        data = json.loads(body)
        print(f"Parsed data: {data}")
        
        # Extract prompt from various possible formats
        prompt = None
        session_id = None
        
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
            print(f"Could not extract prompt from: {data}")
            raise HTTPException(
                status_code=422, 
                detail=f"Missing prompt field. Received: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )
        
        print(f"Processing prompt: {prompt[:100]}...")
        
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
        print(f"Response generated: {len(response_text)} chars")
        
        # Return in multiple formats for compatibility
        return {
            "response": response_text,
            "sessionId": session_id
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        print(f"Error processing request: {e}")
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