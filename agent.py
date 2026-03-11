"""This is a part in which an AI agent is connected"""

import json
import os
import re
from typing import Any
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

from tools.tools import get_order_details, get_customer_profile
from tools.rag import retrieve_policy


DEFAULT_MODEL = os.environ.get("MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

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
                "Use this when the user references an order number (e.g. 'Order #8892)."
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
    You are a helpful customer service AI agent with access to:
    1. CRM tools to look up specific order and customer data
    2. A policy knowledge base for rules about returns and refunds

    Your Behaviour:
    - Always look up relevant order data FIRST when an order number is mentioned
    - After getting order details, check the policy knowledge base to answer policy questions
    - Be specific: quote exact numbers (days, dollar amounts) from both the CRM and the policy
    - If an order doesn't exist, report clearly and concisely that you could not find the order
    - Keep answers concise and factual without unnecessary elaboration

    Decision Rule:
    - Use get_order_details() for -> specific order information (status, customer type, delivery date or processing date)
    - If you find the answer to the user's question in the order details, use that and stop rightaway. If not, then:
    - Use retrieve_policy() only if there is a question about policy for -> rules, return window, refund eligibility questions
    - Use get_customer_profile() for -> detailed customer tier information (perks, support level, return policy differences)

    Always follow the decision rule above to decide which tool to use.
    If you don't have enough information to answer the question, use the tools to gather more information before answering.
    Always complete your reasoning Before giving a final answer.
"""

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
        model: str | None = None, 
) -> dict:
    """Run the agentic loop for a user query."""

    # local Ollama 
    llm = ChatOllama(
        model=model or DEFAULT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        num_predict=1024,
    )

    # Gemini Google API
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-1.5-flash",
    #     temperature=0.3,
    # )
    
    # Simple orchestration: Get order, then get policy
    steps = []
    order_id_match = re.search(r"order\s*#?\s*(\d+)", query, re.IGNORECASE)
    
    messages = [{"role": "system", "content": PROMPT}]
    
    if chat_history:
        messages.extend(chat_history)

    if order_id_match:
        order_id = order_id_match.group(1)
        
        order_data = get_order_details(order_id)
        steps.append({
            "step": 1,
            "tool": "get_order_details",
            "input": {"order_id": order_id},
            "result": order_data,
        })
        
        if "error" not in order_data:
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
                    f"Tool Results for Reference:\n"
                    f"1. Order Data: {json.dumps(order_data, indent=2)}\n"
                    f"2. Policy Context: {policy_data.get('context', 'No policy found')}\n\n"
                    f"Please answer the original query using these tool results."
                ),
            })
        else:
            messages.append({"role": "user", "content": query})
    else:
        messages.append({"role": "user", "content": query})
    
    # Get response from the selected LLM and return structured result
    response = llm.invoke(messages)
    answer = response.content

    return {
        "answer": answer,
        "steps": steps,
        "iterations": len(steps) if steps else 1,
    }

class Agent:
    def __init__(self, request: str):
        self.request = request

    def extract_order_id(self):
        """Extract order ID from the request using regex"""
        pattern = r"order\s*#?\s*(\d+)"
        match = re.search(pattern, self.request, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
    
    def agent_logic(self):
        order_id = self.extract_order_id()

        if not order_id:
            return "Sorry, I couldn't find an order ID in your request. Please provide a valid order ID."
        
        order = get_order_details(order_id)

        if not order or "error" in order:
            return f"Sorry, I couldn't find any details for order ID {order_id}."
        
        policy = retrieve_policy(order.get("customer_type", "Standard"))

        customer_type = order["customer_type"]
        status = order["status"]
        days = order.get("days_since_delivery", 0) or 0

        if customer_type == "VIP":
            return_window = 60
        else:
            return_window = 30

        if days <= return_window:
            return_eligible = "Yes"
        else:
            return_eligible = "No"
        
        return f"""
                Order {order_id} status: {status}.
                Customer type: {customer_type}.
                Delivered {days} days ago.

                Policy: {json.dumps(policy)}

                Return eligible: {return_eligible}.
        """
