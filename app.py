import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
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
AGENTCORE_MEMORY_ID = "bedrock_flow_gen_tool_memory-DZI1UvB5oo"
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
    """Streaming with comprehensive metrics logging"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        body = await request.body()
        data = json.loads(body)
        
        # Extract prompt
        prompt = extract_prompt(data)  # Your existing logic
        
        print(f"Streaming: {prompt[:100]}...")
        
        # Collect metrics
        request_metrics = {
            "prompt_length": len(prompt),
            "start_time": time.time(),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "tools_used": [],
        }
        
        seen_tools = set()
        
        async def generate():
            invocation_state = {
                "user_id": ACTOR_ID,
                "confirm_tool_calls": False,
            }
            
            try:
                async for event in agent.stream_async(prompt, invocation_state=invocation_state):
                    # Tool usage
                    if "current_tool_use" in event:
                        tool_name = event["current_tool_use"].get("name")
                        tool_id = event["current_tool_use"].get("toolUseId")
                        
                        if tool_name and tool_id and tool_id not in seen_tools:
                            seen_tools.add(tool_id)
                            request_metrics["tools_used"].append(tool_name)
                            yield f"\nTool: {tool_name}\n"
                    
                    # Text output
                    elif "data" in event:
                        yield event["data"]
                    
                    # Final metrics
                    elif "result" in event:
                        result = event["result"]
                        if hasattr(result, 'metrics'):
                            metrics = result.metrics.accumulated_usage
                            request_metrics["input_tokens"] = metrics.get('inputTokens', 0)
                            request_metrics["output_tokens"] = metrics.get('outputTokens', 0)
                            request_metrics["total_tokens"] = metrics.get('totalTokens', 0)
                            request_metrics["duration"] = time.time() - request_metrics["start_time"]
                            
                            # Log to CloudWatch / your monitoring system
                            print(f"METRICS: {json.dumps(request_metrics)}")
                        
            except Exception as e:
                yield f"\n\nError: {str(e)}"
                request_metrics["error"] = str(e)
            finally:
                # Always log metrics
                print(f"Final Metrics: {json.dumps(request_metrics, indent=2)}")
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_prompt(data):
    """Helper to extract prompt"""
    if isinstance(data, dict):
        if "input" in data and isinstance(data["input"], dict):
            return data["input"].get("prompt")
        else:
            return (
                data.get("prompt") or 
                data.get("inputText") or 
                data.get("input") or 
                data.get("text") or
                data.get("query")
            )
    return data

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent_ready": agent is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
