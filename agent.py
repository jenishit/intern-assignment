"""This is a part in which an AI agent is connected"""

import json
import os
import re
from typing import Any
from tools/tools import get_order_details
from tools/rag import retrieve_policy

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
            "paramters": {
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
            "paramters": {
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
            "paramters": {
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
    - Use retrieve_policy() for -> rules, return window, refund eligibility questions
    - Use get_customer_profile() for -> detailed customer tier information (perks, support level, return policy differences)

    Always follow the decision rule above to decide which tool to use.
    If you don't have enough information to answer the question, use the tools to gather more information before answering.
    Always complete your reasoning Before giving a final answer.
"""

class Agent:
    def __init__(self, request: str):
        self.request = request

    def extract_order_id(text):
        """Extract order ID from the request using regex"""
        pattern = r"order\s*#?\s*(\d+)"
        match = re.search(pattern, text.request, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
    
    def agent_logic(self):
        order_id = self.extract_order_id(self.request)

        if not order_id:
            return "Sorry, I couldn't find an order ID in your request. Please provide a valid order ID."
        
        order = get_order_details(order_id)

        if not order:
            return f"Sorry, I couldn't find any details for order ID {order_id}."
        
        policy = retrieve_policy(order)

        customer_type = order["customer_type"]
        status = order["status"]
        days = order["days_since_delivery"]

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

                Policy: {policy}

                Return eligible: {return_eligible}.
        """