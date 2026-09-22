"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


def _ensure_workflow_state(session_id: str, customer_id: str = 'UNKNOWN') -> dict:
    """
    Return the session's WorkflowState, creating it if it does not exist yet.
    Guards the routing tools against an orchestrator that skipped
    initialize_session - an update on a missing row would otherwise fail its
    version condition three times and raise.
    """
    state = _read_workflow_state(session_id)
    if state is not None:
        return state
    try:
        return _create_workflow_state(session_id, customer_id)
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        return _read_workflow_state(session_id)   # created concurrently


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════

def _plain(value):
    """Convert DynamoDB Decimal values into plain int/float recursively so
    tool results serialize cleanly into agent context."""
    from decimal import Decimal
    if isinstance(value, Decimal):
        return int(value) if value % 1 == 0 else float(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    # BedrockModel using the WORKER model (Claude Sonnet 4.5), low temperature
    # because this agent reports facts and must not get creative with them.
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    # System prompt for the Inventory Agent
    system_prompt = (
        "You are the InventoryAgent for NovaMart customer support.\n"
        "Your ONLY job is to gather factual order and customer data from the "
        "database using your tools, and report those facts clearly.\n\n"
        "Rules:\n"
        "- Use check_order_status to look up a specific order (it needs BOTH "
        "customer_id and order_id).\n"
        "- Use get_customer_tier to look up the customer's tier (Standard or "
        "Premium) - always include the tier when a return or refund is involved, "
        "because eligibility windows depend on it.\n"
        "- Use list_customer_orders when the order id is unknown or the customer "
        "asks about their order history.\n"
        "- Report facts only: order status, product, dates, prices, tier.\n"
        "- Do NOT make any decisions about return or refund eligibility - that is "
        "the RefundAgent's job.\n"
        "- If a record is not found, say so plainly."
    )

    # Implement check_order_status
    # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
    # order_id = sort key), so a get_item needs BOTH values. That is why this
    # tool takes customer_id as well as order_id.
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            or a not-found message
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        # Composite key: customer_id (partition) + order_id (sort) - both required.
        response = table.get_item(Key={'customer_id': customer_id,
                                       'order_id':    order_id})
        item = response.get('Item')
        if not item:
            return {'found': False,
                    'message': f"No order {order_id} found for customer {customer_id}."}
        return {'found': True, 'order': _plain(item)}

    # Implement get_customer_tier
    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)
        response = table.get_item(Key={'customer_id': customer_id})
        item = response.get('Item')
        if not item:
            return {'found': False,
                    'message': f"No customer record found for {customer_id}."}
        return {'found': True, 'customer': _plain(item)}

    # Implement list_customer_orders
    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        table = dynamodb.Table(config.ORDERS_TABLE)
        # customer_id is the partition key, so a Query returns all of the
        # customer's orders without scanning the whole table.
        response = table.query(
            KeyConditionExpression=Key('customer_id').eq(customer_id)
        )
        orders = _plain(response.get('Items', []))
        return {'customer_id': customer_id,
                'order_count': len(orders),
                'orders':      orders}

    # Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[check_order_status, get_customer_tier, list_customer_orders],
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    # BedrockModel - worker model, low temperature: eligibility decisions
    # must be consistent and rule-based.
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.1,
    )

    # System prompt for the Refund Agent
    system_prompt = (
        "You are the RefundAgent for NovaMart customer support.\n"
        "You decide return/refund eligibility using facts already gathered by "
        "the InventoryAgent.\n\n"
        "Process:\n"
        "1. ALWAYS call get_inventory_context first to read the order facts and "
        "customer tier from the shared workflow state.\n"
        "2. Apply the return window policy:\n"
        "   - Standard tier: order is returnable within 30 days of order_date.\n"
        "   - Premium tier:  order is returnable within 60 days of order_date.\n"
        f"   - Today's date is {time.strftime('%Y-%m-%d')}.\n"
        "3. The order must have been delivered (status 'delivered') to be "
        "returned; cancelled or still-processing orders cannot be returned.\n"
        "4. If and ONLY if the order is eligible, call initiate_refund to start "
        "the return, and report the return reference number.\n"
        "5. If not eligible, do NOT call initiate_refund - explain exactly why "
        "(days elapsed vs. the tier's window, or wrong order status).\n\n"
        "Always state the decision, the tier, the order date, and the window "
        "you applied. Never invent order data."
    )

    # Implement get_inventory_context
    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        state = _read_workflow_state(session_id) or {}
        inventory_context = state.get('inventory_agent', '')
        if not inventory_context:
            return {'inventory_context': {},
                    'note': 'InventoryAgent has not run yet for this session.'}
        return {'inventory_context': _plain(inventory_context)}

    # Implement initiate_refund
    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
        table = dynamodb.Table(config.ORDERS_TABLE)
        table.update_item(
            Key={'customer_id': customer_id, 'order_id': order_id},
            UpdateExpression=(
                "SET #s = :status, return_reference = :ref, "
                "return_reason = :reason, return_date = :date"
            ),
            # 'status' is a DynamoDB reserved word - alias it.
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={
                ':status': 'return_initiated',
                ':ref':    return_reference,
                ':reason': reason,
                ':date':   time.strftime('%Y-%m-%d'),
            },
            ConditionExpression='attribute_exists(order_id)',
        )
        return {
            'success':          True,
            'return_reference': return_reference,
            'order_id':         order_id,
            'instructions': (
                "Return initiated. A prepaid shipping label will be emailed "
                "within 24 hours. Pack the item in its original packaging and "
                f"quote reference {return_reference}. The refund is issued to "
                "the original payment method 5-7 business days after the item "
                "is received."
            ),
        }

    # Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_inventory_context, initiate_refund],
    )


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    # All three retriever sub-agents share the same model settings: worker
    # model, near-zero temperature - retrievers must relay passages, not write.
    def _retriever_model() -> BedrockModel:
        return BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.1,
        )

    def _retriever_prompt(domain: str) -> str:
        return (
            f"You are the {domain}PolicyRetriever, a retrieval sub-agent.\n"
            f"For every query, call your retrieval tool with the query, then "
            f"return the retrieved {domain.lower()} policy passages faithfully. "
            f"Quote the concrete numbers (days, percentages, fees) exactly as "
            f"retrieved. If nothing relevant is found, say 'No relevant "
            f"{domain.lower()} policy found.' Never invent policy terms."
        )

    # Build ReturnsPolicyRetrieverAgent
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        results = retrieve_from_knowledge_base(config.RETURNS_KB_ID, query)
        return format_kb_results(results)

    # Create the ReturnsPolicyRetrieverAgent with the tool above
    returns_retriever = Agent(
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Returns'),
        tools=[retrieve_returns_policy],
    )

    # Build ShippingPolicyRetrieverAgent
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        results = retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query)
        return format_kb_results(results)

    # Create the ShippingPolicyRetrieverAgent with the tool above
    shipping_retriever = Agent(
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Shipping'),
        tools=[retrieve_shipping_policy],
    )

    # Build WarrantyPolicyRetrieverAgent
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        results = retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query)
        return format_kb_results(results)

    # Create the WarrantyPolicyRetrieverAgent with the tool above
    warranty_retriever = Agent(
        model=_retriever_model(),
        system_prompt=_retriever_prompt('Warranty'),
        tools=[retrieve_warranty_policy],
    )

    # search_all_policies - parallel RAG retrieval tool
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        # Map each domain to its retriever sub-agent
        retrievers = {
            'Returns':  returns_retriever,
            'Shipping': shipping_retriever,
            'Warranty': warranty_retriever,
        }

        # ── Trace: show parallel KB dispatch to learners ──────────────────
        trace.kb_start({
            'Returns':  config.RETURNS_KB_ID,
            'Shipping': config.SHIPPING_KB_ID,
            'Warranty': config.WARRANTY_KB_ID,
        })

        # Define a helper to run one retriever sub-agent
        def _run_retriever(domain: str, agent, query: str) -> tuple:
            """
            Run one retriever sub-agent and return (domain, result_text).

            stdout is suppressed globally for all threads by the
            _TraceWriter._suppress_parallel flag set in kb_start().
            This covers both the direct worker thread and any internal
            streaming child threads that Strands SDK spawns internally -
            which do NOT inherit thread-local variables and therefore cannot
            be suppressed with a thread-local capture approach.
            Results are returned as values and printed cleanly and
            sequentially by trace.kb_result() after all futures join.
            """
            try:
                result = agent(
                    f"Retrieve the relevant {domain.lower()} policy passages "
                    f"for this customer question: {query}"
                )
                return domain, str(result).strip()
            except Exception as exc:
                return domain, f"[{domain} retrieval failed: {exc}]"

        # Run all three retrievers in PARALLEL with ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }
            for future in as_completed(futures):
                domain, text = future.result()
                results[domain] = text

        # ── Trace: all KBs responded - print each result sequentially ─────
        trace.kb_done(len(retrievers))
        for domain in ['Returns', 'Shipping', 'Warranty']:
            trace.kb_result(domain, results.get(domain, '[No results]'))

        # Combine results from all three domains and return
        combined = []
        for domain in ['Returns', 'Shipping', 'Warranty']:
            combined.append(
                f"=== {domain.upper()} POLICY ===\n"
                f"{results.get(domain, 'No results returned.')}"
            )
        return "\n\n".join(combined)

    # BedrockModel for the PolicyAgent coordinator - slightly higher
    # temperature (0.2): it synthesizes across three sources.
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2,
    )

    # System prompt for the PolicyAgent coordinator
    system_prompt = (
        "You are the PolicyAgent for NovaMart customer support - the "
        "coordinator of a multi-agent RAG system.\n\n"
        "For EVERY policy question, call search_all_policies exactly once with "
        "the customer's question. It queries the Returns, Shipping and "
        "Warranty knowledge bases in parallel and returns their combined "
        "passages.\n\n"
        "Then synthesize a complete, grounded answer:\n"
        "- Use ONLY the retrieved passages - never invent policy terms.\n"
        "- Quote concrete numbers (day windows, fees, percentages) exactly.\n"
        "- If different domains are relevant (e.g. returns AND warranty), "
        "cover each briefly.\n"
        "- If the passages do not answer the question, say the policy "
        "documents do not cover it.\n"
        "You only know policy TEXT. You have no access to customer accounts "
        "or orders - never speculate about a specific customer's data."
    )

    # Instantiate and return the PolicyAgent coordinator (single tool!)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[search_all_policies],
    )


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    # BedrockModel - worker model, temperature 0.3: the final customer
    # message benefits from a slightly warmer, more natural voice.
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.3,
    )

    # System prompt for the Communication Agent
    system_prompt = (
        "You are the CommunicationAgent for NovaMart customer support. You "
        "write the single, final customer-facing reply.\n\n"
        "Process:\n"
        "1. ALWAYS call get_full_workflow_context first to read everything the "
        "other agents found (order facts, policy passages, refund decision, "
        "and any orchestrator_notes such as a worked price calculation).\n"
        "2. Compose ONE coherent, empathetic reply to the customer:\n"
        "   - Address the customer politely and acknowledge their request.\n"
        "   - Weave in the concrete facts and decisions from the workflow "
        "state (order details, eligibility decision, return reference number, "
        "policy terms) - do not repeat raw JSON.\n"
        "   - If a return was approved, state the reference number and the "
        "next steps. If denied, explain why kindly and clearly.\n"
        "   - Keep it concise: a short paragraph or two, plus next steps.\n"
        "3. Never invent facts that are not in the workflow state, and never "
        "mention internal agents, sessions, or systems."
    )

    # Implement get_full_workflow_context
    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent,
            orchestrator_notes)
        """
        state = _read_workflow_state(session_id)
        if not state:
            return {'error': f"No workflow state found for session {session_id}."}
        return _plain(state)

    # Instantiate and return the Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[get_full_workflow_context],
    )


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    # BedrockModel using the ORCHESTRATOR model - Claude Haiku 4.5 at
    # temperature 0.0: routing must be fast, cheap and deterministic.
    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.0,
    )

    # System prompt for the Orchestrator - encodes the routing rules exactly.
    system_prompt = (
        "You are the OrchestratorAgent for NovaMart customer support. You "
        "NEVER answer customers yourself (with one narrow exception below) - "
        "you route requests to specialist agents via your tools and manage "
        "the shared workflow state.\n\n"
        "Every incoming message starts with '[Session ID: <id>] "
        "[Customer ID: <id>]'. Extract both and pass them to your tools.\n\n"
        "ROUTING RULES - follow them exactly:\n"
        "1. ALWAYS call initialize_session FIRST, for every request.\n"
        "2. Order status / return / refund requests: call "
        "route_to_inventory_agent to gather order facts, THEN "
        "route_to_refund_agent to decide eligibility.\n"
        "3. Policy meaning questions (return windows, shipping rates, "
        "warranty terms - 'what is the policy?'): call route_to_policy_agent.\n"
        "4. Account questions ('what is my tier?', 'am I premium?'): call "
        "route_to_inventory_agent. NEVER route_to_policy_agent for these - "
        "the PolicyAgent only knows policy text, not customer data.\n"
        "5. Pure math / calculation questions: work out the answer yourself - "
        "no specialist routing is needed - but still follow rules 1 and 6, "
        "and pass the full worked calculation and result to "
        "route_to_communication_agent in its context_notes argument.\n"
        "6. ALWAYS call route_to_communication_agent as your VERY LAST tool "
        "call, for every request. It composes the final customer-facing "
        "reply.\n\n"
        "CRITICAL: You are never permitted to write the final customer-facing "
        "response yourself. Even when you believe you already have the "
        "complete answer, your last action MUST be "
        "route_to_communication_agent - no exceptions. After it returns, "
        "output its response as-is."
    )

    # Each routing tool follows the same pattern:
    #   1. read the current WorkflowState  (_read_workflow_state)
    #   2. invoke the worker agent
    #   3. write its result back with optimistic locking
    #      (_update_workflow_state(session_id, {'<column>': text}, expected_version))
    # The terminal trace UI can show each step: call trace.step_start('inventory_agent')
    # before the worker runs and trace.step_done('inventory_agent', old_version) after.

    # Implement route_to_inventory_agent
    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        state = _ensure_workflow_state(session_id, customer_id)
        old_version = int(state['version'])

        trace.step_start('inventory_agent')
        result = inventory_agent(
            f"Customer {customer_id} asks: {request}\n"
            f"Gather the relevant order and customer facts (include the "
            f"customer's tier if a return or refund might be involved)."
        )
        text = str(result).strip()

        _update_workflow_state(session_id, {'inventory_agent': text}, old_version)
        trace.step_done('inventory_agent', old_version)
        return text

    # Implement route_to_policy_agent
    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        state = _ensure_workflow_state(session_id)
        old_version = int(state['version'])

        trace.step_start('policy_agent')
        result = policy_agent(
            f"Answer this policy question using the knowledge bases: {request}"
        )
        text = str(result).strip()

        _update_workflow_state(session_id, {'policy_agent': text}, old_version)
        trace.step_done('policy_agent', old_version)
        return text

    # Implement route_to_refund_agent
    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        state = _ensure_workflow_state(session_id, customer_id)
        old_version = int(state['version'])

        trace.step_start('refund_agent')
        result = refund_agent(
            f"[Session ID: {session_id}] Customer {customer_id} requests: "
            f"{request}\n"
            f"Read the inventory context for session '{session_id}' with "
            f"get_inventory_context, then decide return/refund eligibility "
            f"and act accordingly."
        )
        text = str(result).strip()

        _update_workflow_state(session_id, {'refund_agent': text}, old_version)
        trace.step_done('refund_agent', old_version)
        return text

    # Implement route_to_communication_agent
    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str,
                                     context_notes: str = "") -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message
            context_notes:    Optional facts you worked out yourself without a
                              specialist (e.g. the full result of a math /
                              price calculation). Leave empty otherwise.

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        state = _ensure_workflow_state(session_id, customer_id)
        old_version = int(state['version'])
        if context_notes.strip():
            # Persist the orchestrator's own findings so the CommunicationAgent
            # reads them from WorkflowState like every other agent's output.
            state = _update_workflow_state(
                session_id, {'orchestrator_notes': context_notes.strip()},
                old_version)
            old_version = int(state['version'])

        trace.step_start('communication_agent')
        result = communication_agent(
            f"[Session ID: {session_id}] Compose the final reply for customer "
            f"{customer_id}.\n"
            f"Their original request was: {original_request}\n"
            f"Read the full workflow state for session '{session_id}' with "
            f"get_full_workflow_context first, then write the reply."
        )
        text = str(result).strip()

        _update_workflow_state(session_id, {'communication_agent': text},
                               old_version)
        trace.step_done('communication_agent', old_version)
        return text

    # Implement initialize_session
    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            _create_workflow_state(session_id, customer_id)
            return (f"Session {session_id} initialized for customer "
                    f"{customer_id}. WorkflowState created (version 0).")
        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            # Same session already initialized (e.g. multi-turn chat) - fine.
            return (f"Session {session_id} already initialized for customer "
                    f"{customer_id}. Continuing with existing WorkflowState.")

    # Instantiate and return the OrchestratorAgent with its 5 routing tools
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT PACKAGING
#
#  AgentCore Runtime "direct code deployment" runs a zip that contains the
#  code AND every dependency, compiled for linux/arm64 and the Python version
#  selected in codeConfiguration.runtime - the runtime installs nothing.
#  build_deployment_package() downloads matching wheels with pip
#  (--platform/--python-version/--only-binary) and zips them together with
#  this file, config.py and the other src/ modules. Inside the runtime this
#  same file is the entry point: with no command-line argument it starts the
#  HTTP server (see run_serve) instead of printing usage.
# ═══════════════════════════════════════════════════════

RUNTIME_ENTRYPOINT   = 'agent_orchestrator.py'      # codeConfiguration.entryPoint
RUNTIME_PYTHON       = 'PYTHON_3_12'                # codeConfiguration.runtime
_RUNTIME_PY_VERSION  = '3.12'                       # must match RUNTIME_PYTHON
_RUNTIME_PLATFORM    = 'manylinux2014_aarch64'      # AgentCore runs on arm64
_RUNTIME_MARKER      = '.agentcore-runtime'         # tells __main__ to serve
_RUNTIME_REQUIREMENTS = ['strands-agents>=1.0', 'bedrock-agentcore>=0.1',
                         'boto3>=1.42', 'python-dotenv>=1.0']

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
_RUNTIME_PROJECT_FILES = [
    os.path.join(_SRC_DIR, 'agent_orchestrator.py'),
    os.path.join(_SRC_DIR, 'agent_utils.py'),
    os.path.join(_SRC_DIR, 'agent_observability.py'),
    os.path.join(_SRC_DIR, 'bedrock_kb_retrieval.py'),
    os.path.join(_ROOT_DIR, 'config.py'),
]


def _zip_write(zf, full: str, arcname: str, data: bytes = None) -> None:
    """Add one file with the 644/755 permissions AgentCore requires."""
    import zipfile
    info = zipfile.ZipInfo.from_file(full, arcname) if data is None else zipfile.ZipInfo(arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    executable = data is None and os.access(full, os.X_OK) and not full.endswith('.py')
    info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
    if data is None:
        with open(full, 'rb') as fh:
            data = fh.read()
    zf.writestr(info, data)


def build_deployment_package(output_path: str) -> str:
    """Build the AgentCore deployment zip at output_path and return the path."""
    import shutil, subprocess, tempfile, zipfile

    missing = [p for p in _RUNTIME_PROJECT_FILES if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Cannot package runtime, missing: {missing}")

    with tempfile.TemporaryDirectory(prefix='agentcore-pkg-') as tmp:
        deps_dir = os.path.join(tmp, 'deps')
        os.makedirs(deps_dir)
        print(f"  Downloading arm64 dependencies (python {_RUNTIME_PY_VERSION}) ...", flush=True)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
            '--target', deps_dir, '--platform', _RUNTIME_PLATFORM,
            '--python-version', _RUNTIME_PY_VERSION, '--implementation', 'cp',
            '--only-binary=:all:', '--upgrade', *_RUNTIME_REQUIREMENTS,
        ], check=True)
        for junk in ('bin', 'tests'):
            shutil.rmtree(os.path.join(deps_dir, junk), ignore_errors=True)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for dirpath, dirnames, filenames in os.walk(deps_dir):
                dirnames[:] = [d for d in dirnames if d != '__pycache__']
                for name in filenames:
                    if not name.endswith(('.pyc', '.pyo')):
                        full = os.path.join(dirpath, name)
                        _zip_write(zf, full, os.path.relpath(full, deps_dir))
            for path in _RUNTIME_PROJECT_FILES:
                _zip_write(zf, path, os.path.basename(path))
            _zip_write(zf, '', _RUNTIME_MARKER, data=b'agentcore runtime package\n')

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Package built: {output_path} ({size_mb:.1f} MB, entry point {RUNTIME_ENTRYPOINT})")
    if size_mb > 250:
        raise RuntimeError("Deployment package exceeds the 250 MB AgentCore limit")
    return output_path


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)

    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            numbered = [v['version'] for v in versions.get('guardrails', [])
                        if str(v.get('version', 'DRAFT')).isdigit()]
            if numbered:
                guardrail_version = max(numbered, key=int)
            else:
                # A previous run created the guardrail but never published a
                # version - publish one now so we never hand back DRAFT.
                guardrail_version = bedrock_client.create_guardrail_version(
                    guardrailIdentifier=guardrail_id,
                    description='Initial published version',
                )['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version

    # Create the guardrail
    # Use bedrock_client.create_guardrail() with:
    #   - name (config.GUARDRAIL_NAME) and description
    #   - contentPolicyConfig - filtersConfig for SEXUAL, VIOLENCE, HATE at HIGH
    #     strength and INSULTS, MISCONDUCT at MEDIUM strength (input + output)
    #   - sensitiveInformationPolicyConfig - piiEntitiesConfig:
    #       CREDIT_DEBIT_CARD_NUMBER and US_SOCIAL_SECURITY_NUMBER -> BLOCK
    #       EMAIL and PHONE -> ANONYMIZE
    #   - topicPolicyConfig - one DENY topic per entry in config.GUARDRAIL_BLOCKED_TOPICS
    #     (competitor products, pricing negotiations, legal threats)
    #   - wordPolicyConfig - managedWordListsConfig with type PROFANITY
    #   - blockedInputMessaging and blockedOutputsMessaging
    #
    # Then promote it from DRAFT to a numbered version with
    # bedrock_client.create_guardrail_version(guardrailIdentifier=...)
    # and return (guardrail_id, guardrail_version).

    # Content filters: harmful content HIGH, insults/misconduct MEDIUM
    content_filters = [
        {'type': 'SEXUAL',     'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
        {'type': 'VIOLENCE',   'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
        {'type': 'HATE',       'inputStrength': 'HIGH',   'outputStrength': 'HIGH'},
        {'type': 'INSULTS',    'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
        {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
    ]

    # PII: block card numbers and SSNs outright; anonymize emails and phones
    pii_entities = [
        {'type': 'CREDIT_DEBIT_CARD_NUMBER',  'action': 'BLOCK'},
        {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
        {'type': 'EMAIL',                     'action': 'ANONYMIZE'},
        {'type': 'PHONE',                     'action': 'ANONYMIZE'},
    ]

    # One DENY topic per blocked subject from config
    # Definitions are kept narrow (and backed by examples) so that ordinary
    # price arithmetic - e.g. "5 items at $29.99 with 10% off" in Task 7 -
    # is NOT mistaken for a pricing negotiation.
    topic_definitions = {
        'competitor products':  ('Discussion, comparison, or recommendation of '
                                 'competitor products, stores, or brands.'),
        'pricing negotiations': ('The customer tries to bargain or haggle with '
                                 'NovaMart for a lower price, a special deal, '
                                 'or a price match.'),
        'legal threats':        ('Legal threats, lawsuit discussions, or '
                                 'requests for legal advice or liability admissions.'),
    }
    topic_examples = {
        'competitor products':  ['Is Best Buy cheaper than you for this laptop?',
                                 'Should I buy the Sony headphones from Amazon instead?'],
        'pricing negotiations': ['Can you knock 20% off if I buy today?',
                                 'Will you match the price I found on another site?',
                                 "I'll only order if you give me a better deal."],
        'legal threats':        ["I'm going to sue NovaMart over this.",
                                 'My lawyer will be contacting you.'],
    }
    topics = [
        {
            'name':       topic.replace(' ', '-'),
            'definition': topic_definitions.get(
                topic, f"Discussion of {topic} with customer support."),
            'examples':   topic_examples.get(topic, []),
            'type':       'DENY',
        }
        for topic in config.GUARDRAIL_BLOCKED_TOPICS
    ]

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description=('NovaMart customer support guardrail: blocks harmful '
                     'content, protects PII, keeps conversations on-topic.'),
        contentPolicyConfig={'filtersConfig': content_filters},
        sensitiveInformationPolicyConfig={'piiEntitiesConfig': pii_entities},
        topicPolicyConfig={'topicsConfig': topics},
        wordPolicyConfig={'managedWordListsConfig': [{'type': 'PROFANITY'}]},
        blockedInputMessaging=(
            "I'm sorry, but I can't help with that request. I'm here to help "
            "with NovaMart orders, returns, shipping, and warranty questions."),
        blockedOutputsMessaging=(
            "I'm sorry, but I can't provide that response. Is there anything "
            "else I can help you with regarding your NovaMart order?"),
    )
    guardrail_id = response['guardrailId']
    print(f"  Guardrail created: {guardrail_id}")

    # Promote DRAFT to a numbered version so the tests (and production
    # config) reference an immutable snapshot rather than a moving DRAFT.
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description='Initial published version',
    )
    guardrail_version = version_response['version']
    print(f"  Guardrail version published: {guardrail_version}")

    return guardrail_id, guardrail_version


def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime.

    AgentCore does not serialize Python objects, so `orchestrator_agent` is
    not uploaded directly. Instead the packaging step below zips this file,
    which doubles as the HTTP entry point (see run_serve), together with its
    helper modules and all dependencies
    compiled for arm64. The runtime is then created from that zip ("direct
    code deployment").

    The guardrail is attached by environment variables: inside the runtime
    build_agent_graph() reads GUARDRAIL_ID / GUARDRAIL_VERSION and applies
    them to every agent's model (see _apply_guardrail), exactly as `test`
    and `chat` do locally.

    Returns:
        The AgentCore Runtime ARN
    """
    runtime_name = config.AGENTCORE_RUNTIME_NAME
    s3_client    = boto3.client('s3', region_name=config.AWS_REGION)

    # Check if runtime already exists
    try:
        existing = agentcore_control.list_agent_runtimes()
        for r in existing.get('agentRuntimes', []):
            if r['agentRuntimeName'] == runtime_name:
                runtime_arn = r['agentRuntimeArn']
                print(f"AgentCore Runtime already exists: {runtime_arn}")
                return runtime_arn
    except Exception as e:
        print(f"  [Note] Could not check existing runtimes: {e}")

    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")

    # Build the deployment package and upload it to S3.
    package_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'build', 'deployment_package.zip')
    build_deployment_package(package_path)

    artifact_key = f"agentcore-artifacts/{runtime_name}/deployment_package.zip"
    s3_client.upload_file(package_path, config.POLICY_BUCKET, artifact_key)
    print(f"  Artifact uploaded: s3://{config.POLICY_BUCKET}/{artifact_key}")

    # Deploy to AgentCore Runtime
    # Use agentcore_control.create_agent_runtime() with:
    #   - agentRuntimeName (runtime_name), description
    #   - roleArn (config.AGENTCORE_ROLE_ARN)
    #   - agentRuntimeArtifact = {'codeConfiguration': {
    #         'code': {'s3': {'bucket': config.POLICY_BUCKET, 'prefix': artifact_key}},
    #         'runtime': RUNTIME_PYTHON,
    #         'entryPoint': [RUNTIME_ENTRYPOINT]}}
    #   - networkConfiguration  {'networkMode': 'PUBLIC'}
    #   - protocolConfiguration {'serverProtocol': 'HTTP'}
    #   - environmentVariables: AWS_REGION, PROJECT_NAME, RETURNS_KB_ID,
    #     SHIPPING_KB_ID, WARRANTY_KB_ID, AGENT_LOG_GROUP, and the guardrail
    #     (GUARDRAIL_ID = guardrail_id, GUARDRAIL_VERSION = guardrail_version)
    # Store the API response in `response`.
    response = agentcore_control.create_agent_runtime(
        agentRuntimeName=runtime_name,
        description='NovaMart multi-agent customer support system',
        roleArn=config.AGENTCORE_ROLE_ARN,
        agentRuntimeArtifact={
            'codeConfiguration': {
                'code': {
                    's3': {
                        'bucket': config.POLICY_BUCKET,
                        'prefix': artifact_key,
                    }
                },
                'runtime':    RUNTIME_PYTHON,
                'entryPoint': [RUNTIME_ENTRYPOINT],
            }
        },
        networkConfiguration={'networkMode': 'PUBLIC'},
        protocolConfiguration={'serverProtocol': 'HTTP'},
        # The guardrail travels as environment variables: inside the runtime
        # build_agent_graph() -> _apply_guardrail() reads GUARDRAIL_ID /
        # GUARDRAIL_VERSION and sets them on every agent's BedrockModel.
        environmentVariables={
            'AWS_REGION':        config.AWS_REGION,
            'PROJECT_NAME':      config.PROJECT_NAME,
            'RETURNS_KB_ID':     config.RETURNS_KB_ID,
            'SHIPPING_KB_ID':    config.SHIPPING_KB_ID,
            'WARRANTY_KB_ID':    config.WARRANTY_KB_ID,
            'AGENT_LOG_GROUP':   config.AGENT_LOG_GROUP,
            'GUARDRAIL_ID':      guardrail_id,
            'GUARDRAIL_VERSION': guardrail_version,
        },
    )

    if response is None:
        raise NotImplementedError("deploy_to_agentcore_runtime: create_agent_runtime() not implemented")

    # Wait for the runtime to become READY and return its ARN.
    runtime_arn = response['agentRuntimeArn']
    print(f"  Runtime created: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, response['agentRuntimeId'])
    print(' ready.')
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # Create AgentCore Memory
    # Use agentcore_control.create_memory() with:
    #   - name (memory_name) and a description
    #   - eventExpiryDuration = 7   (days)
    #   - memoryStrategies = [{'summaryMemoryStrategy': {
    #         'name': 'SessionSummary',
    #         'namespaces': ['/summaries/{actorId}/{sessionId}']}}]
    #   - clientToken (e.g. str(uuid.uuid4())) for idempotency
    # Store the API response in `response`.
    response = agentcore_control.create_memory(
        name=memory_name,
        description=('Session-summary memory for NovaMart customer support - '
                     'keeps conversation context across turns for 7 days.'),
        eventExpiryDuration=7,
        memoryStrategies=[{
            'summaryMemoryStrategy': {
                'name':       'SessionSummary',
                'namespaces': ['/summaries/{actorId}/{sessionId}'],
            }
        }],
        clientToken=str(uuid.uuid4()),
    )

    if response is None:
        raise NotImplementedError("configure_memory: create_memory() not implemented")

    # Wait until the memory resource is ACTIVE and return its ARN.
    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    # Build the logging configuration
    # logging_configuration = {
    #     'cloudWatchConfig': {'logGroupName': config.AGENT_LOG_GROUP,
    #                          'logLevel': 'INFO', 'enabled': True},
    #     'xRayConfig':       {'enabled': True, 'samplingRate': 1.0},
    # }
    # Then apply it:  summary = apply_observability_config(runtime_arn, logging_configuration)
    # Wrap the call in try/except - on success print the CloudWatch log group
    # and the X-Ray sampling rate; on exception print
    #   "[Note] Observability configuration failed: <e>"

    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel':     'INFO',
            'enabled':      True,
        },
        'xRayConfig': {
            'enabled':      True,
            'samplingRate': 1.0,   # 100% in dev; reduce to ~0.05 in prod
        },
    }

    try:
        summary = apply_observability_config(runtime_arn, logging_configuration)
        print(f"  CloudWatch Logs : {config.AGENT_LOG_GROUP} (level INFO)")
        print(f"  X-Ray tracing   : enabled, sampling rate "
              f"{logging_configuration['xRayConfig']['samplingRate']:.0%}")
        if summary:
            print(f"  {summary}")
    except Exception as e:
        print(f"  [Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses a three-step deployment pattern:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def _write_env_values(values: dict) -> Optional[str]:
    """
    Set KEY=value lines in the project .env (next to config.py), replacing
    existing keys in place and appending missing ones. Returns the path, or
    None if the file could not be written.
    """
    env_path = os.path.join(_ROOT_DIR, '.env')
    try:
        lines = []
        if os.path.exists(env_path):
            with open(env_path, encoding='utf-8') as fh:
                lines = fh.read().splitlines()
        pending = dict(values)
        for i, line in enumerate(lines):
            key = line.split('=', 1)[0].strip()
            if not line.lstrip().startswith('#') and key in pending:
                lines[i] = f"{key}={pending.pop(key)}"
        lines += [f"{k}={v}" for k, v in pending.items()]
        with open(env_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        for k, v in values.items():
            os.environ[k] = str(v)
        return env_path
    except OSError as exc:
        print(f"  [Note] Could not update .env automatically: {exc}")
        return None


def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    missing_kbs = [k for k in ('RETURNS_KB_ID', 'SHIPPING_KB_ID', 'WARRANTY_KB_ID')
                   if not getattr(config, k, '')]
    if missing_kbs:
        print(f"  [Warning] {', '.join(missing_kbs)} not set in .env (Task 5).")
        print("  The runtime will deploy, but policy questions will fail until you")
        print("  create the Knowledge Bases, add their IDs to .env and re-run deploy")
        print("  (re-running refreshes the KB IDs on the existing runtime).\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    deployed = {
        'AGENTCORE_RUNTIME_ARN': runtime_arn,
        'GUARDRAIL_ID':          guardrail_id,
        'GUARDRAIL_VERSION':     guardrail_version,
    }
    env_path = _write_env_values(deployed)
    print(f"\n  {'Saved to ' + env_path if env_path else 'Add these to your .env file'}:")
    for key, value in deployed.items():
        print(f"  {key}={value}")
    print()
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
