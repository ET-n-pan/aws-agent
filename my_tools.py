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
    """
    client = (
        boto3.client("bedrock-agent-runtime", region_name=region)
        if region is not None
        else boto3.client("bedrock-agent-runtime")
    )

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

    for event in response.get("responseStream", []):
        raw_events.append(event)
        aggregated.update(event)

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


# -------------------------------------------------------------------
# S3 TEMPLATE TOOLS (new)
# -------------------------------------------------------------------

S3_TEMPLATES_BUCKET_PREFIX = "bedrock-flow-templates"
DEFAULT_TEMPLATE_KEY = "simple-bedrock-flow.yaml"


def get_s3_client(region: Optional[str] = None):
    """Get S3 client with optional region."""
    if region:
        return boto3.client("s3", region_name=region)
    return boto3.client("s3")


def get_templates_bucket_name(region: Optional[str] = None) -> str:
    """
    Deterministic bucket name:
    bedrock-flow-templates-<account-id>-<region>
    """
    session = boto3.session.Session()
    reg = region or session.region_name or "us-east-1"
    sts = boto3.client("sts", region_name=reg)
    account_id = sts.get_caller_identity()["Account"]
    return f"{S3_TEMPLATES_BUCKET_PREFIX}-{account_id}-{reg}"


def ensure_templates_bucket(region: Optional[str] = None) -> str:
    """
    Ensure the templates bucket exists. Create it if necessary.
    """
    s3 = get_s3_client(region)
    bucket_name = get_templates_bucket_name(region)

    # Try a cheap head_bucket, create if it fails
    try:
        s3.head_bucket(Bucket=bucket_name)
    except Exception:
        create_kwargs: Dict[str, Any] = {"Bucket": bucket_name}
        session = boto3.session.Session()
        reg = region or session.region_name or "us-east-1"
        if reg != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": reg
            }
        s3.create_bucket(**create_kwargs)

    return bucket_name


def _get_default_template() -> str:
    """
    Internal helper for default template body (YAML).
    """
    return """AWSTemplateFormatVersion: '2010-09-09'
Description: 'Simple Amazon Bedrock Flow with Input, Prompt, and Output nodes'

Parameters:
  FlowName:
    Type: String
    Default: 'SimpleBedrockFlow'
    Description: 'Name for the Bedrock Flow'
  
  FlowDescription:
    Type: String
    Default: 'A simple flow that processes user input through a prompt and returns the response'
    Description: 'Description for the Bedrock Flow'

Resources:
  # IAM Role for Bedrock Flow execution
  BedrockFlowExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${FlowName}-ExecutionRole'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Sid: FlowsTrustBedrock
            Effect: Allow
            Principal:
              Service: bedrock.amazonaws.com
            Action: sts:AssumeRole
            Condition:
              StringEquals:
                'aws:SourceAccount': !Ref 'AWS::AccountId'
              ArnLike:
                'AWS:SourceArn': !Sub 'arn:aws:bedrock:${AWS::Region}:${AWS::AccountId}:flow/*'
      Policies:
        - PolicyName: BedrockFlowExecutionPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Sid: InvokeModel
                Effect: Allow
                Action:
                  - bedrock:InvokeModel
                Resource: 
                  - !Sub 'arn:aws:bedrock:${AWS::Region}::foundation-model/amazon.nova-lite-v1:0'

  # Simple Bedrock Flow
  SimpleBedrockFlow:
    Type: AWS::Bedrock::Flow
    Properties:
      Name: !Ref FlowName
      Description: !Ref FlowDescription
      ExecutionRoleArn: !GetAtt BedrockFlowExecutionRole.Arn
      Definition:
        Nodes:
          # Input Node - receives user input
          - Type: Input
            Name: FlowInput
            Outputs:
              - Name: document
                Type: Object
          
          # Prompt Node - processes the input through an AI model
          - Type: Prompt
            Name: ProcessInput
            Configuration:
              Prompt:
                SourceConfiguration:
                  Inline:
                    ModelId: global.anthropic.claude-haiku-4-5-20251001-v1:0
                    TemplateType: TEXT
                    InferenceConfiguration:
                      Text:
                        Temperature: 0.7
                        MaxTokens: 2000
                    TemplateConfiguration:
                      Text:
                        Text: |
                          You are a helpful assistant. Please respond to the following input:
                          
                          {{user_input}}
                          
                          Provide a clear, helpful, and concise response.
            Inputs:
              - Name: user_input
                Type: String
                Expression: $.data.input
            Outputs:
              - Name: modelCompletion
                Type: String
          
          # Output Node - returns the final response
          - Type: Output
            Name: FlowOutput
            Inputs:
              - Name: document
                Type: String
                Expression: $.data
        
        Connections:
          # Connection from Input to Prompt Node
          - Name: FlowInput_ProcessInput_user_input
            Source: FlowInput
            Target: ProcessInput
            Type: Data
            Configuration:
              Data:
                SourceOutput: document
                TargetInput: user_input
          
          # Connection from Prompt to Output Node  
          - Name: ProcessInput_FlowOutput
            Source: ProcessInput
            Target: FlowOutput
            Type: Data
            Configuration:
              Data:
                SourceOutput: modelCompletion
                TargetInput: document

  # Flow Version
  FlowVersion:
    Type: AWS::Bedrock::FlowVersion
    Properties:
      FlowArn: !GetAtt SimpleBedrockFlow.Arn
      Description: !Sub 'Version 1 of ${FlowName}'

  # Flow Alias
  FlowAlias:
    Type: AWS::Bedrock::FlowAlias
    Properties:
      FlowArn: !GetAtt SimpleBedrockFlow.Arn
      Name: DRAFT
      Description: 'Draft alias for the simple flow'
      RoutingConfiguration:
        - FlowVersion: !GetAtt FlowVersion.Version

Outputs:
  FlowId:
    Description: 'ID of the created Bedrock Flow'
    Value: !GetAtt SimpleBedrockFlow.Id
    Export:
      Name: !Sub '${AWS::StackName}-FlowId'
  
  FlowArn:
    Description: 'ARN of the created Bedrock Flow'
    Value: !GetAtt SimpleBedrockFlow.Arn
    Export:
      Name: !Sub '${AWS::StackName}-FlowArn'
  
  FlowAliasArn:
    Description: 'ARN of the Flow Alias'
    Value: !GetAtt FlowAlias.Arn
    Export:
      Name: !Sub '${AWS::StackName}-FlowAliasArn'
  
  FlowAliasId:
    Description: 'ID of the Flow Alias'
    Value: !GetAtt FlowAlias.Id
    Export:
      Name: !Sub '${AWS::StackName}-FlowAliasId'
  
  ExecutionRoleArn:
    Description: 'ARN of the execution role'
    Value: !GetAtt BedrockFlowExecutionRole.Arn
    Export:
      Name: !Sub '${AWS::StackName}-ExecutionRoleArn'
"""


@tool
def get_default_template() -> str:
    """
    Return the default working Bedrock Flow CloudFormation template as YAML.
    """
    return _get_default_template()


def _ensure_default_template_in_bucket(region: Optional[str] = None) -> str:
    """
    Ensure the templates bucket exists and contains the default template at least once.
    Returns the bucket name.
    """
    bucket = ensure_templates_bucket(region)
    s3 = get_s3_client(region)

    # Check if bucket has any object; if empty, seed with default template
    resp = s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
    if not resp.get("Contents"):
        s3.put_object(
            Bucket=bucket,
            Key=DEFAULT_TEMPLATE_KEY,
            Body=_get_default_template().encode("utf-8"),
        )
    return bucket


@tool
def list_s3_templates(region: Optional[str] = None) -> str:
    """
    List available CFN templates in the fixed S3 bucket.
    If the bucket does not exist or is empty, create it and save the default template first.
    """
    bucket = _ensure_default_template_in_bucket(region)
    s3 = get_s3_client(region)

    keys: List[str] = []
    continuation_token: Optional[str] = None

    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "MaxKeys": 1000}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])

        if not resp.get("IsTruncated"):
            break
        continuation_token = resp.get("NextContinuationToken")

    result = {
        "bucket": bucket,
        "templates": keys,
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def save_template(
    template_name: str,
    template_body: str,
    region: Optional[str] = None,
) -> str:
    """
    Save a CFN template to the fixed S3 bucket.

    - If template_name has no extension, '.yaml' is appended.
    - Returns bucket and key as JSON.
    """
    bucket = ensure_templates_bucket(region)
    s3 = get_s3_client(region)

    key = template_name
    if "." not in key:
        key = f"{key}.yaml"

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=template_body.encode("utf-8"),
    )

    result = {
        "bucket": bucket,
        "key": key,
        "status": "SAVED",
    }
    return json.dumps(result, ensure_ascii=False)


@tool
def get_template(
    template_name: str,
    region: Optional[str] = None,
) -> str:
    """
    Retrieve a CFN template from the fixed S3 bucket.

    - If template_name has no extension, '.yaml' is appended.
    - Returns JSON: { "bucket", "key", "template_body" }.
    """
    bucket = ensure_templates_bucket(region)
    s3 = get_s3_client(region)

    key = template_name
    if "." not in key:
        key = f"{key}.yaml"

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")

    result = {
        "bucket": bucket,
        "key": key,
        "template_body": body,
    }
    return json.dumps(result, ensure_ascii=False)
