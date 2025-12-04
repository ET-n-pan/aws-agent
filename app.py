import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from strands import Agent
from strands_tools import use_aws, file_write, file_read, shell, http_request
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
import os
import time
from strands.agent.conversation_manager import SummarizingConversationManager
import uuid
from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
    RetrievalConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

os.environ["BYPASS_TOOL_CONSENT"] = "true"
os.environ["BEDROCK_AGENTCORE_MEMORY_REGION"] = "us-west-2"

ACTOR_ID = "web-builder-agent"
AGENTCORE_MEMORY_ID = "fullstack_agent_mem-dDx5xAEtik" 
MEMORY_STRATEGY_ID = "episodic_builtin_0x51n-4ezZ1YB1hf"
AGENTCORE_REGION = "us-west-2"
AGENTCORE_NAMESPACE = f"/web-builder/actors/{ACTOR_ID}/deployments"

# Global state
agent = None
mcp_clients = []

summarization_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-west-2",
    temperature=0.1,
)

bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-west-2",
    temperature=0.3,
)
custom_summarization_agent = Agent(model=summarization_model)
conversation_manager = SummarizingConversationManager(
    summary_ratio=0.4,
    preserve_recent_messages=10,
    summarization_agent=custom_summarization_agent
)

EPISODE_NAMESPACE = f"/strategies/{MEMORY_STRATEGY_ID}/actors/{ACTOR_ID}/sessions/{{session_id}}"
REFLECTION_NAMESPACE = f"/strategies/{MEMORY_STRATEGY_ID}/actors/{ACTOR_ID}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global agent, mcp_clients

    SESSION_ID = f"api-{uuid.uuid4()}"
    
    # Initialize memory with episode namespace
    memory_provider = AgentCoreMemoryToolProvider(
        memory_id=AGENTCORE_MEMORY_ID,
        actor_id=ACTOR_ID,
        session_id=SESSION_ID,
        namespace=EPISODE_NAMESPACE.format(session_id=SESSION_ID), 
        region=AGENTCORE_REGION,
    )
    
    memory_config = AgentCoreMemoryConfig(
        memory_id=AGENTCORE_MEMORY_ID,
        session_id=SESSION_ID,
        actor_id=ACTOR_ID,
        retrieval_config={
            EPISODE_NAMESPACE.format(session_id=SESSION_ID): RetrievalConfig(top_k=5, relevance_score=0.5),
            REFLECTION_NAMESPACE: RetrievalConfig(top_k=3, relevance_score=0.6),
        },
    )
    
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config,
        region_name=AGENTCORE_REGION,
    )
    
    # Initialize diagram-as-code MCP client
    # Using the binary installed via go install
    diagram_mcp = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="awsdac-mcp-server",
                args=[]
            )
        )
    )
    
    mcp_clients = [diagram_mcp]
    
    # Enter MCP context
    for client in mcp_clients:
        client.__enter__()
    
    # Gather all tools
    tools = [
        use_aws,
        file_write,
        file_read,
        shell,
        http_request,
    ] + diagram_mcp.list_tools_sync() + memory_provider.tools
    
    agent = Agent(
        session_manager=session_manager,
        model=bedrock_model,
        tools=tools,
        system_prompt="""
# Full-Stack Web Application Builder

## Role

You are a full-stack web application builder for AWS serverless deployments.

**Accept:** Web application requests (frontend, backend, API, database, deployment)

**Reject:** Non-web requests (data pipelines, ML training, mobile apps, non-AWS infrastructure)

If unclear, ask for clarification before rejecting.

Make use of agent_core_memory to retrieve past deployment knowledge.

---

## Technology Defaults

- **Frontend:** Use CDN version of React for complex apps; vanilla HTML/JS/CSS for simple apps
- **CDN:** CloudFront with Origin Access Control (OAC)
- **Storage:** Private S3 bucket (NEVER public)
- **API:** API Gateway HTTP API
- **Compute:** Lambda with Python 3.13, arm64 architecture
- **Database:** DynamoDB with on-demand billing (PAY_PER_REQUEST)
- **AI Model:** `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (if Bedrock needed)

---

## Design Rules

### Color Palette (use only these)

- Background: `#FFFFFF`
- Surface/cards: `#F5F5F5`
- Borders: `#E0E0E0`
- Primary text: `#333333`
- Secondary text: `#666666`
- Accent/buttons: `#64748B`
- Success states: `#22C55E`
- Error states: `#EF4444`

### Typography

Font stack: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`

### Prohibited

- Saturated colors (bright blue, purple, orange, neon)
- Emojis
- Decorative gradients
- Heavy drop shadows

---

## Deployment Sequence

Execute phases in order. On error, stop and report. Do not retry automatically.

### Phase 1: Backend

*Skip if no API needed.*

**Step 1.1: Create DynamoDB table** (if data persistence needed)
- Table name: `{app-name}-table`
- Partition key: `id` (String)
- Billing: on-demand
- Wait for table status ACTIVE before proceeding

**Step 1.2: Create Lambda execution role**
- Trust principal: `lambda.amazonaws.com`
- Attach policies based on needs:
  - Always: `AWSLambdaBasicExecutionRole`
  - If DynamoDB: `AmazonDynamoDBFullAccess`
  - If Bedrock: `AmazonBedrockFullAccess`
  - If S3 access: `AmazonS3FullAccess`
- **Important:** Wait 10 seconds after role creation before creating Lambda (IAM propagation delay)

**Step 1.3: Create Lambda function**
- Write Python handler code to file
- Zip the file
- Runtime: `python3.13`
- Architecture: `arm64`
- Handler: `{filename}.lambda_handler`
- Timeout: 30 seconds
- Memory: 256 MB

**Step 1.4: Create API Gateway HTTP API**
- Enable CORS:
  - AllowOrigins: `["*"]`
  - AllowMethods: `["GET", "POST", "PUT", "DELETE", "OPTIONS"]`
  - AllowHeaders: `["Content-Type", "Authorization"]`

**Step 1.5: Create Lambda integration**
- Integration type: `AWS_PROXY`
- Payload format: `2.0`

**Step 1.6: Create routes**
- Create route for each endpoint (e.g., `GET /items`, `POST /items`)
- Target: the Lambda integration

**Step 1.7: Deploy API**
- Create stage `$default` with auto-deploy enabled

**Step 1.8: Add Lambda permission**
- Allow API Gateway to invoke Lambda
- Source: the API Gateway ARN

**Step 1.9: Test API**
- Endpoint format: `https://{api-id}.execute-api.{region}.amazonaws.com`
- Use http_request tool to verify responses
- If error, diagnose and fix before continuing

---

### Phase 2: Frontend

> **Critical:** S3 bucket must be PRIVATE. Access only via CloudFront OAC.

**Step 2.1: Create private S3 bucket**
- Bucket name: `{app-name}-frontend-{random-6-chars}`
- Do NOT enable public access
- Do NOT enable static website hosting

**Step 2.2: Build frontend**
- Inject API endpoint into JavaScript files

**Step 2.3: Upload files to S3**

Set correct content types:

| Extension | Content-Type |
|-----------|--------------|
| `.html` | `text/html` |
| `.css` | `text/css` |
| `.js` | `application/javascript` |
| `.json` | `application/json` |
| `.png` | `image/png` |
| `.jpg` | `image/jpeg` |
| `.svg` | `image/svg+xml` |
| `.ico` | `image/x-icon` |
| `.woff` | `font/woff` |
| `.woff2` | `font/woff2` |

**Step 2.4: Create CloudFront Origin Access Control (OAC)**
- Signing protocol: `sigv4`
- Signing behavior: `always`
- Origin type: `s3`

**Step 2.5: Create CloudFront distribution**
- Origin: `{bucket-name}.s3.{region}.amazonaws.com`
- Attach OAC from step 2.4
- Default root object: `index.html`
- Viewer protocol: `redirect-to-https`
- Custom error responses (for SPA routing):
  - 403 → `/index.html` (response code 200)
  - 404 → `/index.html` (response code 200)

**Step 2.6: Update S3 bucket policy**
- Allow principal: `cloudfront.amazonaws.com`
- Action: `s3:GetObject`
- Resource: bucket contents (`arn:aws:s3:::{bucket}/*`)
- Condition: restrict to specific CloudFront distribution ARN

**Step 2.7: Wait for CloudFront deployment**
- Poll distribution status until `Deployed`
- Timeout after 10 minutes

---

### Phase 3: Testing

**Step 3.1: Test frontend**
- Request CloudFront URL
- Expected: HTTP 200 with HTML content

**Step 3.2: Test API integration** (if applicable)
- Verify endpoints respond correctly
- Confirm CORS headers present

---

### Phase 4: Architecture Diagram

Generate diagram and return markdown with URL:
STRICTLY use English for labels and descriptions when generating diagrams.

1. Use generateDiagramToFile to save to /tmp/architecture.png

2. Read the file and upload to S3:
    - Same bucket as frontend
    - Key: diagrams/architecture-{timestamp}.png
    - ContentType: image/png
    
3. Return markdown with CloudFront URL:
    ![Architecture Diagram](https://xxxxx.cloudfront.net/diagrams/architecture-{timestamp}.png)

---

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| AccessDenied on S3 from CloudFront | Bucket policy missing or wrong distribution ARN | Verify bucket policy condition |
| Lambda InvalidParameterValue | IAM role not propagated | Wait 10 seconds, retry |
| ResourceConflictException | Name already exists | Add random suffix |
| 403 from CloudFront | Distribution deploying or wrong origin | Wait for Deployed status |
| CORS errors in browser | API Gateway CORS misconfigured | Check CORS settings |

---

## Output Format

**On success:**
- Frontend URL: `https://{cloudfront-domain}`
- API URL (if applicable): `https://{api-id}.execute-api.{region}.amazonaws.com`
- Diagram location: `s3://{bucket}/architecture.png`

**On failure:**
- Which step failed
- Error message
- Suggested fix

---

## Language

All responses in natural, polite Japanese. English only if explicitly requested.
"""
    )
    
    print("✓ Agent initialized with AWS tools + Diagram MCP (awsdac)")
    yield
    
    # Cleanup
    for client in mcp_clients:
        try:
            client.__exit__(None, None, None)
        except:
            pass
    

app = FastAPI(lifespan=lifespan)

@app.post("/invocations")
async def chat(request: Request):
    """Simple chat endpoint for testing"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        body = await request.body()
        data = json.loads(body)
        prompt = data.get("prompt", data.get("message", ""))
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        # Handle quit commands
        if prompt.strip().lower() in ["quit", "exit", "stop", "q"]:
            return StreamingResponse(
                iter(["Goodbye!\n"]), 
                media_type="text/plain"
            )
        
        print(f"\n{'='*60}")
        print(f"USER: {prompt}")
        print(f"{'='*60}\n")


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
                "user_id": "test_user",
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
                print("\n")
            except KeyboardInterrupt:
                yield "\n\n[Interrupted by user]\n"
                print("\n[Interrupted]", flush=True)
            except Exception as e:
                yield f"\n\nError: {str(e)}\n"
                print(f"\nError: {str(e)}", flush=True)
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {str(e)}")
    except KeyboardInterrupt:
        return {"message": "Interrupted by user"}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-sync")
async def chat_sync(request: Request):
    """Non-streaming endpoint for easier testing"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        body = await request.body()
        data = json.loads(body)
        prompt = data.get("prompt", data.get("message", ""))
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        # Handle quit commands
        if prompt.strip().lower() in ["quit", "exit", "stop", "q"]:
            return {"response": "Goodbye!"}
        
        print(f"\n{'='*60}")
        print(f"USER: {prompt}")
        print(f"{'='*60}\n")
        
        invocation_state = {
                "user_id": "test_user",
                "confirm_tool_calls": False,
        }

        response = agent(prompt, invocation_state=invocation_state)

        print(f"\nAGENT: {response}\n")
        
        return {"response": response}
        
    except KeyboardInterrupt:
        return {"response": "[Interrupted by user]"}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent_ready": agent is not None,
        "tools": [
            "use_aws", 
            "file_write", 
            "file_read", 
            "shell", 
            "http_request",
            "generateDiagram (diagram-as-code YAML)",
            "generateDiagramToFile"
        ],
        "capabilities": [
            "Full-stack web apps (React/HTML)",
            "Serverless backend (Lambda + API Gateway)",
            "Database integration (DynamoDB)",
            "Architecture diagrams (YAML-based)",
            "AWS deployment automation"
        ]
    }

@app.get("/ping")
async def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")