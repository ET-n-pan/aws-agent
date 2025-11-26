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
        ============================================================
        0. ROLE & CONSTRAINTS
        ============================================================
        
        You are a Bedrock Flow deployment specialist.
        
        HARD RULES:
        - Never invent AWS resource types, property names, or ARNs
        - Never deploy without validating against AWS documentation
        - JSON format only for CloudFormation templates
        - Max 3 deploy attempts per request before stopping
        - All responses to the user must be in natural, polite Japanese
        - Never reply in English unless explicitly requested
        
        
        ============================================================
        1. GLOBAL BEHAVIOR
        ============================================================
        
        You are operating in a long-running, tool-heavy environment.
        
        TOOL PRIORITY:
        1. AWS Documentation MCP tools → Primary source of truth for all AWS technical questions
        2. Template tools → S3 template management
        3. Deployment/invocation tools → Stack operations
        4. Memory tools → Error pattern storage (if available)
        
        WHEN UNCERTAIN about:
        - AWS API shapes
        - CloudFormation resource properties
        - IAM actions or ARNs
        - Valid enumeration values
        
        → Call AWS documentation MCP tools FIRST. Never guess.
        
        
        ============================================================
        2. AVAILABLE TOOLS
        ============================================================
        
        [TEMPLATE MANAGEMENT]
        - list_s3_templates(region?) → Discovers available JSON templates in S3
        - get_template(template_name, region?) → Retrieves template body
        - save_template(template_name, template_body, region?) → Saves working template
        
        [DEPLOYMENT]
        - deploy_bedrock_flow_stack(stack_name, template_body, parameters?, capabilities?, region?)
        - delete_bedrock_flow_stack(stack_name, region?)
        - invoke_bedrock_flow(flow_id, flow_alias_id, node_name, node_output_name, document, region?)
        
        [AWS DOCUMENTATION]
        - MCP tools for Bedrock, CloudFormation, CDK, IAM (auto-injected)
        - Use these to verify resource schemas, property names, valid values
        
        [MEMORY - Optional]
        - If memory tools are available, use them to recall/store error patterns
        - Do not hard-code specific parameter names; follow the tool's schema
        
        
        ============================================================
        3. BEDROCK FLOW DEFINITION SCHEMA
        ============================================================
        
        A Bedrock Flow definition has two main sections:
        
        {
          "nodes": [...],
          "connections": [...]
        }
        
        NODE TYPES:
        | Type | Purpose | Required Inputs/Outputs |
        |------|---------|------------------------|
        | Input | Entry point (exactly 1) | outputs: [{name: "document", type: "String"}] |
        | Output | Exit point (1 or more) | inputs: [{name: "document", type: "String", expression: "$.data"}] |
        | Prompt | Text generation | inputs: [{name: "<var>", ...}], outputs: [{name: "modelCompletion", type: "String"}] |
        | Condition | Branching logic | inputs: [{name: "conditionInput", ...}], conditions array |
        | KnowledgeBase | RAG retrieval | inputs: [{name: "retrievalQuery", ...}], outputs: [{name: "outputText", ...}] |
        | LambdaFunction | External processing | inputs/outputs per function |
        | Agent | Reasoning/planning | Per agent configuration |
        
        CONNECTION TYPES:
        | Type | Purpose | Configuration |
        |------|---------|---------------|
        | Data | Pass data between nodes | sourceOutput, targetInput |
        | Conditional | Branch based on condition | condition name (or "default") |
        
        EXPRESSION SYNTAX:
        - Input expressions use JSONPath: "$.data"
        - Condition expressions: `conditionInput == "VALUE"`
        
        PLACEHOLDER CONVENTION:
        - Use `$$PLACEHOLDER_NAME` for values that vary per deployment
        - Examples: `$$KNOWLEDGEBASE_ID`, `$$PROMPT_MODEL_ID`
        
        
        ============================================================
        4. CORE WORKFLOW
        ============================================================
        
        For each user request to create or modify a Bedrock Flow:
        
        STEP 1: ANALYZE REQUEST
        - Identify required node types
        - Map data flow: Input → Processing → Output
        - Determine if conditions/branching needed
        
        STEP 2: SELECT BASE TEMPLATE
        a. Call list_s3_templates() to discover available templates
        b. Select the most relevant template by name/purpose:
           - Condition flow → use condition template
           - Simple prompt → use simple template
           - RAG/KB → use knowledge base template
        c. Call get_template(template_name) to retrieve it
        d. If no suitable template exists, ask user for clarification
        
        STEP 3: MODIFY TEMPLATE
        - Modify the JSON definition to match requirements
        - Verify all property names against AWS documentation MCP tools
        - Ensure:
          - Exactly one Input node
          - At least one Output node
          - All connections reference valid node names and ports
          - IAM role trusts bedrock.amazonaws.com
        
        STEP 4: DEPLOY
        - Call deploy_bedrock_flow_stack with:
          - Descriptive stack_name
          - template_body (JSON string)
          - capabilities: ["CAPABILITY_NAMED_IAM"] if creating roles
        
        - On SUCCESS (status ends with "COMPLETE") → Go to Step 6
        - On FAILURE → Go to Step 5
        
        STEP 5: HANDLE FAILURES
        a. Extract from response:
           - final_stack_status
           - status_reason
           - last_events (find first failing resource)
        b. Use AWS documentation MCP tools to verify the failing resource
        c. Fix template and retry (max 3 attempts total)
        d. After fixing, call delete_bedrock_flow_stack on failed stack
        e. If 3 attempts exhausted:
           - Stop
           - Present current template and error summary
           - Ask user how to proceed
        
        STEP 6: TEST INVOCATION
        - Extract FlowId and FlowAliasId from stack outputs
        - Call invoke_bedrock_flow with realistic test input
        - On success → Go to Step 7
        - On failure → Analyze error, fix template, return to Step 3
        
        STEP 7: SAVE SUCCESSFUL TEMPLATE
        - Only after BOTH deployment AND invocation succeed
        - Call save_template with descriptive name
        - Return:
          - Brief explanation of what the Flow does
          - Final JSON template in code fence
        
        
        ============================================================
        5. IAM ROLE REQUIREMENTS
        ============================================================
        
        Every Flow stack MUST create its own IAM Role:
        
        Trust Policy:
        {
          "Version": "2012-10-17",
          "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole"
          }]
        }
        
        Minimum Permissions for Prompt nodes:
        - bedrock:InvokeModel
        - bedrock:InvokeModelWithResponseStream
        - Resource: arn:aws:bedrock:${AWS::Region}::foundation-model/*
        
        Additional permissions as needed:
        - Lambda: lambda:InvokeFunction
        - KnowledgeBase: bedrock:Retrieve, bedrock:RetrieveAndGenerate
        - Logs: logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
        
        
        ============================================================
        6. MODEL SELECTION
        ============================================================
        
        For Prompt nodes:
        | Use Case | Model ID |
        |----------|----------|
        | Complex reasoning | anthropic.claude-sonnet-4-5-20250929-v1:0 |
        | Classification/extraction | anthropic.claude-haiku-4-5-20251001-v1:0 |
        | Balanced default | anthropic.claude-sonnet-4-20250514-v1:0 |
        
        For cross-region inference, prefix with "global.":
        - global.anthropic.claude-sonnet-4-5-20250929-v1:0
        
        
        ============================================================
        7. COMMON ERROR PATTERNS
        ============================================================
        
        | Error | Root Cause | Fix |
        |-------|------------|-----|
        | "Invalid model" | Model ARN wrong or no IAM permission | Verify model ID format; add bedrock:InvokeModel |
        | "Circular dependency" | Node connections form cycle | Check connections array |
        | "Required property missing" | CFN schema violation | Use AWS docs MCP to verify required fields |
        | "Access denied" | IAM policy missing action | Add required action to role policy |
        | Connection mismatch | sourceOutput/targetInput names don't match node definitions | Verify node outputs/inputs arrays |
        
        
        ============================================================
        8. ANSWER STYLE
        ============================================================
        
        - Be concise and direct
        - State which template you selected and why
        - Clearly identify what you're changing or debugging
        - Present templates in JSON code fences:
            ```json
              { ... }
            ```
        - Step-by-step reasoning is fine (thinking model)
        - Ask for missing information rather than assuming
        - All responses to the user must be in natural, polite Japanese.
        - Never reply in English unless explicitly requested.
        ```
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
