"""
check_runtime_env.py
====================
Diagnose (and optionally repair) the environment variables on the deployed
AgentCore Runtime.

Test 3.4 requires the runtime to carry AWS_REGION, PROJECT_NAME, the three
Knowledge Base IDs, AGENT_LOG_GROUP, GUARDRAIL_ID and GUARDRAIL_VERSION.
A runtime created before those values existed in .env keeps the empty
strings it was deployed with.

Usage:
    python check_runtime_env.py          # report only
    python check_runtime_env.py --fix    # push the values from .env onto the runtime
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import boto3
import config

REQUIRED = ('AWS_REGION', 'PROJECT_NAME', 'RETURNS_KB_ID', 'SHIPPING_KB_ID',
            'WARRANTY_KB_ID', 'AGENT_LOG_GROUP', 'GUARDRAIL_ID', 'GUARDRAIL_VERSION')


def main() -> int:
    fix = '--fix' in sys.argv

    runtime_arn = config.AGENTCORE_RUNTIME_ARN
    if not runtime_arn:
        print("AGENTCORE_RUNTIME_ARN is not set in .env - deploy first.")
        return 1

    client = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
    runtime_id = runtime_arn.split('/')[-1]
    current = client.get_agent_runtime(agentRuntimeId=runtime_id)
    env = dict(current.get('environmentVariables') or {})

    print(f"\nRuntime : {runtime_id}")
    print(f"Status  : {current.get('status')}")
    print(f"Network : {current.get('networkConfiguration', {}).get('networkMode')}"
          f"  |  Protocol: {current.get('protocolConfiguration', {}).get('serverProtocol')}\n")

    # What the values should be, taken from .env / CloudFormation via config.py
    wanted = {key: str(getattr(config, key, '') or '') for key in REQUIRED}

    missing_locally = [k for k, v in wanted.items() if not v]
    problems = []
    print(f"  {'VARIABLE':<20} {'ON RUNTIME':<28} {'FROM .env / CONFIG':<28} OK")
    print(f"  {'-'*20} {'-'*28} {'-'*28} --")
    for key in REQUIRED:
        on_runtime = env.get(key, '')
        expected = wanted[key]
        ok = bool(on_runtime) and (not expected or on_runtime == expected)
        if not ok:
            problems.append(key)
        print(f"  {key:<20} {(on_runtime or '(empty)'):<28} "
              f"{(expected or '(not set locally)'):<28} {'yes' if ok else 'NO'}")

    for key in ('AGENT_LOG_LEVEL', 'AGENT_LOG_TO_CLOUDWATCH',
                'AGENT_TRACING_ENABLED', 'AGENT_TRACE_SAMPLING_RATE'):
        print(f"  {key:<20} {(env.get(key) or '(empty)'):<28} "
              f"{'(set by configure_observability)':<28}")

    if missing_locally:
        print(f"\n  These are missing from your .env, so they cannot be pushed: "
              f"{', '.join(missing_locally)}")
    if not problems:
        print("\n  All required environment variables are present on the runtime.")
        return 0

    print(f"\n  Incorrect or empty on the runtime: {', '.join(problems)}")
    if not fix:
        print("  Re-run with --fix to update the runtime from your .env values,")
        print("  or run:  python src/agent_orchestrator.py deploy")
        return 1

    pushable = {k: v for k, v in wanted.items() if v}
    env.update(pushable)
    update_kwargs = {
        'agentRuntimeId':       runtime_id,
        'agentRuntimeArtifact': current['agentRuntimeArtifact'],
        'roleArn':              current['roleArn'],
        'networkConfiguration': current['networkConfiguration'],
        'environmentVariables': env,
    }
    for key in ('description', 'protocolConfiguration', 'lifecycleConfiguration',
                'authorizerConfiguration', 'requestHeaderConfiguration'):
        if current.get(key):
            update_kwargs[key] = current[key]

    print("\n  Updating runtime environment variables ...")
    client.update_agent_runtime(**update_kwargs)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from agent_observability import wait_for_runtime_ready
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(client, runtime_id)
    print(" ready.")
    print("\n  Done - now run:  python tests/test_agent.py task3")
    return 0


if __name__ == '__main__':
    sys.exit(main())
