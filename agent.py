"""This is a part in which an AI agent is connected"""

import json
import os
import re
from typing import Any

from tools.tools import get_order_details, get_customer_profile
from tools.rag import retrieve_policy
from llm_providers import get_llm

TOOL_REGISTRY = {
    "get_order_details": get_order_details,
    "get_customer_profile": get_customer_profile
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": (
                "Look up a specific order in the CRM by order ID."
                "Use this when the user references an order number (e.g. 'Order #8892' or '#8892' or '8892' or 'order status of #8892' or 'order status of 8892' )."
                "Returns: customer type (VIP/Standard), delivery status, days since delivery, "
                "item name, and order amount. "
                "Do not use this for policy questions - use retrieve_policy instead."    
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up (e.g. '8892' or '#8892')",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_profile",
            "description": (
                "Retrieve the customer profile for a specific order (VIP or Standard)."
                "Use this when you need detailed perks, support level, or return policy information."
                "for a customer tier — beyond just what the policy says. "
                "Input: 'VIP' or 'Standard'."    
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_type": {
                        "type": "string",
                        "description": "The customer tier: 'VIP' or 'Standard'",
                    }
                },
                "required": ["customer_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_policy",
            "description": (
                "Search the customer service knowledge base for relevant policies. "
                "Use this for questions about: return windows, refund rules, "
                "delay compensation, or any 'what is the policy on X' question. "
                "Do not use for order lookup - use get_order_details for that. "
                "Input: a short description of what policy you are looking for."    
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy topic to search for (e.g. 'VIP return window')",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

#Master prompt for the agent
PROMPT = """
You are a concise customer service AI agent.

Rules:
- Answer ONLY with the facts from the tool results provided.
- Do NOT add explanations, caveats, disclaimers, or extra commentary.
- Keep your answer to 2-4 short sentences maximum.
- Quote exact numbers (days, amounts) from the data.
- If an order is not found, say so in one sentence.

Format your answer as plain facts, not a conversation.
"""

POLICY_KEYWORDS = [
    "policy", "return", "refund", "eligible", "eligibility",
    "can i return", "can the customer return", "can they return",
    "compensation", "return window", "based on policy",
]


def _needs_policy(query: str) -> bool:
    """Return True only if the user is asking about policy / returns / refunds."""
    q = query.lower()
    return any(kw in q for kw in POLICY_KEYWORDS)


def _run_tool(tool_name: str, tool_input: dict) -> Any:
    """Run a tool call to the correct function and return result as JSON result."""
    if tool_name == "retrieve_policy":
        result = retrieve_policy(**tool_input)
    elif tool_name in TOOL_REGISTRY:
        result = TOOL_REGISTRY[tool_name](**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result, indent=2)


def run_agent(
        query: str,
        max_itr: int = 10,
        chat_history: list[dict] | None = None,
) -> dict:
    """Run the agentic loop for a user query."""

    llm = get_llm()

    steps: list[dict] = []
    order_id_match = re.search(r"#\s*(\d+)", query) or re.search(r"order\s+.*?(\d{4,})", query, re.IGNORECASE)

    messages = [{"role": "system", "content": PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    if order_id_match:
        order_id = order_id_match.group(1)

        # Step 1 — always fetch order details when an order ID is present
        order_data = get_order_details(order_id)
        steps.append({
            "step": 1,
            "tool": "get_order_details",
            "input": {"order_id": order_id},
            "result": order_data,
        })

        if "error" not in order_data and _needs_policy(query):
            # Step 2 — fetch policy ONLY when user asks about policy/return/refund
            customer_type = order_data.get("customer_type", "Standard")
            policy_data = retrieve_policy(f"{customer_type} return window")
            steps.append({
                "step": 2,
                "tool": "retrieve_policy",
                "input": {"query": f"{customer_type} return window"},
                "result": policy_data,
            })

            messages.append({
                "role": "user",
                "content": (
                    f"{query}\n\n"
                    f"Tool Results:\n"
                    f"1. Order Data: {json.dumps(order_data, indent=2)}\n"
                    f"2. Policy: {policy_data.get('context', 'No policy found')}\n\n"
                    f"Answer concisely using ONLY these tool results."
                ),
            })
        elif "error" not in order_data:
            # Order found, no policy needed — answer from order data only
            messages.append({
                "role": "user",
                "content": (
                    f"{query}\n\n"
                    f"Tool Result:\n"
                    f"Order Data: {json.dumps(order_data, indent=2)}\n\n"
                    f"Answer concisely using ONLY this tool result."
                ),
            })
        else:
            # Order not found
            messages.append({
                "role": "user",
                "content": (
                    f"{query}\n\n"
                    f"Tool Result:\n"
                    f"{json.dumps(order_data)}\n\n"
                    f"Report that the order was not found."
                ),
            })
    elif _needs_policy(query):
        # No order ID but it's a policy question — answer with policy rag only
        policy_data = retrieve_policy(query)
        steps.append({
            "step": 1,
            "tool": "retrieve_policy",
            "input": {"query": query},
            "result": policy_data,
        })
        
        messages.append({
            "role": "user",
            "content": (
                f"{query}\n\n"
                f"Tool Result:\n"
                f"Policy: {policy_data.get('context', 'No policy found')}\n\n"
                f"Answer concisely using ONLY this tool result."
            ),
        })
    else:
        # No order ID and not a policy question — allow general conversation
        messages.append({
            "role": "user",
            "content": query,
        })

    # Get response from the LLM
    response = llm.invoke(messages)
    answer = response.content

    return {
        "answer": answer,
        "steps": steps,
        "iterations": len(steps) if steps else 1,
    }
