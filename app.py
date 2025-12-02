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

os.environ["BYPASS_TOOL_CONSENT"] = "true"

# Global state
agent = None
mcp_clients = []

bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name="us-west-2",
    temperature=0.3,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global agent, mcp_clients
    
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
    ] + diagram_mcp.list_tools_sync()
    
    agent = Agent(
        model=bedrock_model,
        tools=tools,
        system_prompt="""
You are a full-stack web development specialist with AWS deployment capabilities.

AVAILABLE TOOLS:
- use_aws: All AWS service operations (S3, CloudFormation, Lambda, API Gateway, DynamoDB, etc)
- file_write: Create files (HTML, JS, Python, etc)
- file_read: Read existing files
- shell: Execute commands (npm, zip, aws cli, etc)
- http_request: Test HTTP endpoints
- generateDiagram: Generate AWS architecture diagrams from YAML
- generateDiagramToFile: Save diagrams directly to file

ARCHITECTURE PATTERNS:

1. FRONTEND: React (Vite) or HTML → S3 static hosting
2. BACKEND: Lambda + API Gateway (Python 3.13)
3. DATABASE: DynamoDB (serverless, simple, cost-effective)
4. DIAGRAM: Generate architecture diagram after deployment


DESIGN RULES:

- Use clean, professional styling
- CSS: prefer neutral tones (grays, whites, light blues)
- Fonts: sans-serif, clean and modern
- Layouts: simple, intuitive, user-friendly
- NO blue/purple (or other high saturation color) themes
- NO emojis


DEPLOYMENT WORKFLOW:

1. BACKEND:
   - Create DynamoDB table if needed (use_aws)
   - Generate Lambda function code
   - Deploy via CloudFormation or direct use_aws calls
   - Extract API endpoint from outputs

2. FRONTEND:
   - Generate HTML/React with above CSS
   - Create S3 bucket
   - Upload files with proper content types
   - Enable static website hosting

3. TESTING:
   - Test API with http_request tool
   - Verify DynamoDB operations
   - Return frontend URL

4. DIAGRAM:
   Generate simple YAML diagram:
   ```yaml
   resources:
     - id: api
       type: apigateway
       name: API
     - id: lambda
       type: lambda
       name: Function
     - id: db
       type: dynamodb
       name: Database
     - id: web
       type: s3
       name: Frontend
   connections:
     - from: api
       to: lambda
     - from: lambda
       to: db
   ```

- Be direct and practical. Use the professional CSS for all frontends. Generate architecture diagrams.
- All responses to the user must be in natural, polite Japanese
- Never reply in English unless explicitly requested
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

@app.post("/chat")
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
        
        async def generate():
            try:
                invocation_state = {
                    "user_id": "test_user",
                    "confirm_tool_calls": False,
                }
                async for event in agent.stream_async(prompt, invocation_state=invocation_state):
                    if "data" in event:
                        yield event["data"]
                    elif "current_tool_use" in event:
                        tool_name = event["current_tool_use"].get("name", "unknown")
                        yield f"\n[Using tool: {tool_name}]\n"
                        print(f"\n[Tool: {tool_name}]", flush=True)
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

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped by user")