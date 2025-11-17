import json
import time
from typing import Dict, List, Optional, Any

import boto3
from strands import tool

def get_cfn_client(region=None):
    """Get CloudFormation client with optional region."""
    if region:
        return boto3.client("cloudformation", region_name=region)
    return boto3.client("cloudformation")

TERMINAL_STATUSES = {
    "CREATE_COMPLETE",
    "CREATE_FAILED",
    "ROLLBACK_COMPLETE",
    "ROLLBACK_FAILED",
    "UPDATE_COMPLETE",
    "UPDATE_ROLLBACK_COMPLETE",
    "UPDATE_ROLLBACK_FAILED",
}


@tool
def deploy_bedrock_flow_stack(
    stack_name: str,
    template_body: str,
    parameters: Optional[Dict[str, str]] = None,
    capabilities: Optional[List[str]] = None,
    region: Optional[str] = None,
    poll_interval_seconds: int = 10,
    timeout_seconds: int = 1800,
) -> str:
    """
    Create or update a CloudFormation stack and wait until it reaches a terminal state.

    Returns:
        JSON string with a "progress_event"-style summary:
        {
          "status": "SUCCESS" | "FAILED" | "TIMEOUT",
          "stack_name": "...",
          "stack_id": "...",
          "final_stack_status": "...",
          "status_reason": "...",
          "last_events": [...],
        }
    """
    client = get_cfn_client(region)
    start = time.time()

    # Check if stack exists
    exists = False
    try:
        client.describe_stacks(StackName=stack_name)
        exists = True
    except client.exceptions.ClientError:
        exists = False

    kwargs = {
        "StackName": stack_name,
        "TemplateBody": template_body,
    }
    if parameters:
        kwargs["Parameters"] = [
            {"ParameterKey": k, "ParameterValue": v}
            for k, v in parameters.items()
        ]
    if capabilities:
        kwargs["Capabilities"] = capabilities

    if not exists:
        resp = client.create_stack(**kwargs)
        action = "CREATE"
    else:
        resp = client.update_stack(**kwargs)
        action = "UPDATE"

    stack_id = resp["StackId"]

    last_events = []
    status_reason = None
    final_stack_status = None

    while True:
        if time.time() - start > timeout_seconds:
            return json.dumps(
                {
                    "status": "TIMEOUT",
                    "stack_name": stack_name,
                    "stack_id": stack_id,
                    "final_stack_status": final_stack_status,
                    "status_reason": f"Timed out after {timeout_seconds} seconds",
                    "last_events": last_events,
                    "action": action,
                },
                ensure_ascii=False,
            )

        desc = client.describe_stacks(StackName=stack_id)
        stack = desc["Stacks"][0]
        final_stack_status = stack["StackStatus"]
        status_reason = stack.get("StackStatusReason")

        # Get a few recent events for debugging
        events_resp = client.describe_stack_events(StackName=stack_id)
        events = events_resp.get("StackEvents", [])[:10]
        last_events = [
            {
                "timestamp": e["Timestamp"].isoformat(),
                "resource_type": e["ResourceType"],
                "logical_resource_id": e["LogicalResourceId"],
                "resource_status": e["ResourceStatus"],
                "resource_status_reason": e.get("ResourceStatusReason"),
            }
            for e in events
        ]

        if final_stack_status in TERMINAL_STATUSES:
            break

        time.sleep(poll_interval_seconds)

    status = "SUCCESS" if final_stack_status.endswith("COMPLETE") else "FAILED"

    result = {
        "status": status,
        "stack_name": stack_name,
        "stack_id": stack_id,
        "final_stack_status": final_stack_status,
        "status_reason": status_reason,
        "last_events": last_events,
        "action": action,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def invoke_bedrock_flow(
    flow_id: str,
    flow_alias_id: str,
    node_name: str,
    node_output_name: str,
    document: Any,
    region: Optional[str] = None,
) -> str:
    """
    Invoke a Bedrock Flow alias and return its output and raw events.

    This tool is intentionally simple:
    - It does NOT guess the Flow input schema.
    - It does NOT guess the node name or node output name.
    - It does NOT catch errors; any exception from InvokeFlow is surfaced to the agent.

    Args:
        flow_id:
            The Flow ID (flowIdentifier).
        flow_alias_id:
            The Flow alias ID (flowAliasIdentifier).
        node_name:
            The EXACT name of the Flow input node (e.g. "InputNode").
        node_output_name:
            The EXACT output name on that input node (e.g. "document").
        document:
            The value to send as `content.document` to that node.
            This MUST match the schema the Flow input node expects
            (e.g. a string, or a JSON object with specific fields).
        region:
            AWS region for Bedrock Agent Runtime. If None, uses default Boto3 config.

    Returns:
        JSON string with a simple summary:

        {
          "flow_id": "...",
          "flow_alias_id": "...",
          "node_name": "...",
          "node_output_name": "...",
          "completion_reason": "SUCCESS" | "ERROR" | null,
          "output_document": { ... } | "..." | null,
          "raw_events": [ ... ]
        }

        Notes:
        - If InvokeFlow raises an exception, it will propagate to the agent as a tool error.
        - The agent should inspect `output_document` and `raw_events` to understand any Flow-level errors.
    """
    client = (
        boto3.client("bedrock-agent-runtime", region_name=region)
        if region is not None
        else boto3.client("bedrock-agent-runtime")
    )

    # We make NO assumptions about schema beyond using the standard "document" slot.
    request: Dict[str, Any] = {
        "flowIdentifier": flow_id,
        "flowAliasIdentifier": flow_alias_id,
        "inputs": [
            {
                "content": {
                    "document": document,
                },
                "nodeName": node_name,
                "nodeOutputName": node_output_name,
            }
        ],
    }

    response = client.invoke_flow(**request)

    raw_events = []
    aggregated: Dict[str, Any] = {}

    # We simply collect all events and also keep a flattened view
    for event in response.get("responseStream", []):
        raw_events.append(event)
        aggregated.update(event)

    # Best-effort extraction for convenience (no error handling)
    completion_reason = None
    output_document = None

    flow_completion = aggregated.get("flowCompletionEvent")
    if isinstance(flow_completion, dict):
        completion_reason = flow_completion.get("completionReason")

    flow_output = aggregated.get("flowOutputEvent")
    if isinstance(flow_output, dict):
        output_document = flow_output.get("content", {}).get("document")

    result = {
        "flow_id": flow_id,
        "flow_alias_id": flow_alias_id,
        "node_name": node_name,
        "node_output_name": node_output_name,
        "completion_reason": completion_reason,
        "output_document": output_document,
        "raw_events": raw_events,
    }

    return json.dumps(result, ensure_ascii=False)