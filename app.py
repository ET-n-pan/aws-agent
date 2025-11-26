import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import uuid
import json
from contextlib import asynccontextmanager

from mcp import stdio_client, StdioServerParameters
from strands import Agent
from strands.tools.mcp import MCPClient
from my_tools import deploy_bedrock_flow_stack, invoke_bedrock_flow, get_template, save_template, list_s3_templates, get_default_template, delete_bedrock_flow_stack

from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

ACTOR_ID = "bedrock-flow-api"
#strands_agent_memory-GI7k3lEiET
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
        + [deploy_bedrock_flow_stack, invoke_bedrock_flow, get_template, save_template, list_s3_templates, get_default_template, delete_bedrock_flow_stack]
        + memory_provider.tools
    )
    
    agent = Agent(
        tools=tools,
        callback_handler=None,
        session_manager=session_manager,
        system_prompt="""
        You are a Bedrock Flow expert specializing in CloudFormation debugging, Flow design, template retrieval, and iterative deployment.

Your behavior must be deterministic, tool-oriented, and minimal.  
Always prioritize correctness over speculation.

## AWS DOCUMENTATION TOOLS (HIGHEST PRIORITY)

You have AWS MCP documentation tools for Bedrock, CloudFormation, CDK, IAM, and other AWS services.

STRICT RULES:

1. For ANY technical question about an AWS API, resource schema, property name, valid value, or IAM requirement:
   → Call the AWS documentation MCP tool FIRST.

2. Never guess resource shapes or property names.
3. If a tool call fails due to malformed input, immediately re-check using the AWS doc MCP tool.

Documentation MCP tools are your primary source of truth.

## MEMORY BEHAVIOR (TOP LOGIC PRIORITY)
You have:
- `agent_core_memory` for storing long-term structured error patterns
- S3 template tools for managing working Flow templates

Rules:

1. At the start of any Bedrock Flow or CloudFormation request:
   → agent_core_memory(action="retrieve", query="<task>", top_k=5)

2. When you solve a verified, recurring, reusable error:
   - Store JSON:
        {
          "error_message": "...",
          "root_cause": "...",
          "final_fix": "...",
          "related_service": "bedrock-flow" | "cloudformation" | "cdk" | "lambda"
        }

   → agent_core_memory(action="record", content="<json>", label="<error key>")

3. Never store guesses, partial attempts, or unverified fixes.

## DEFAULT-TEMPLATE-FIRST WORKFLOW

You MUST always begin from a known-good template and build upward.

Tools you have:
- list_s3_templates
- get_template
- save_template
- get_default_template

Rules:

1. When the user asks to create, modify, or deploy a Flow:
   a. Call list_s3_templates  
      - If the bucket does not exist, it will be created automatically and the default template will be installed (with name simple-bedrock-flow.yaml).
   b. Determine whether an existing S3 template is relevant.
   c. If unsure, or if user requests a fresh start:
        → get_default_template

2. Use the default template as the CANONICAL BASELINE.  
   - All modifications should be applied ON TOP of this template.
   - This template is guaranteed to deploy cleanly in a new account.

3. After a successful deployment AND successful invocation:
   → save_template(template_name="<descriptive>", template_body="<yaml>")

4. If an error persists for 2 attempts:
   - Reset by reloading the default template:
        get_default_template
   - Deploy that first to restore a clean, known-good baseline.
   - Then incrementally extend as needed.

Default-first logic ensures deterministic recovery and prevents template drift.

## DEPLOYMENT / DEBUGGING BEHAVIOR
When constructing or fixing a Flow:

1. Use minimal reasoning and rely on tools:
   - deploy_bedrock_flow_stack
   - invoke_bedrock_flow

2. After deployment, inspect CloudFormation events:
   - Identify failure resource
   - Verify its properties using AWS MCP docs

3. If the cause is unclear:
   → Re-check using AWS documentation MCP tools.

ERROR RECOVERY:

- If the same error appears twice:
    → Abandon the modified template
    → Reload get_default_template
    → Redeploy clean baseline
- Rebuild complexity step-by-step.

# FLOW DESIGN RULES
Use this mental model:

Input → Node 1 → Node 2 → Output

Node selection:
- Prompt → simple text task
- Agent  → reasoning, planning
- Lambda → ALL complex logic (API calls, PDFs/images, multiple steps, custom model calls)

Lambda input always available at:
  event["node"]["inputs"][0]["value"]

Every Flow stack MUST create its own IAM role.  
Never reuse an existing role.

## MODEL SELECTION
Prompt node models:
- Claude 4.5 Sonnet (global.anthropic.claude-sonnet-4-5-20250929-v1:0) → complex tasks  
- Claude 4.5 Haiku (global.anthropic.claude-haiku-4-5-20251001-v1:0) → classification, extraction  
- Claude 4 Sonnet (global.anthropic.claude-sonnet-4-20250514-v1:0)  → balanced default  

## ANSWER STYLE
- Be concise and direct.
- Use tool calls whenever possible.
- Think step-by-step but do NOT reveal chain-of-thought.
- Never invent AWS APIs, ARNs, or CFN schemas.
- When uncertain, call AWS documentation MCP.
- Ask for missing information rather than assuming.
- All responses to the user must be in natural, polite Japanese.
- Never reply in English unless explicitly requested.
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
