from typing import Any

ORDERS_DB = {
    "8892": {
        "order_id": "8892",
        "customer_type": "VIP",
        "status": "Delivered",
        "days_since_delivery": 45,
        "item": "Random Item1",
        "amount": 200.0,
    },
    "9910": {
        "order_id": "9910",
        "customer_type": "Standard",
        "status": "Processing",
        "days_since_delivery": None,
        "item": "Random Item2",
        "amount": 100.0,
        "days_since_order": 2,
    },
}

CUSTOMER_PROFILES_DB = {
    "VIP": {
        "tier": "VIP",
        "return_window_days": 60,
        "delay_compensation": "Full refund",
        "support_level": "Priority 24/7",
        "shipping": "Free express",
        "perks": ["Dedicated account manager", "Early access to sales", "Free returns"],
    },
    "Standard": {
        "tier": "Standard",
        "return_window_days": 30,
        "delay_compensation": "$20 credit",
        "support_level": "Business hours",
        "shipping": "Standard 5-7 days",
        "perks": ["Loyalty points on purchases"],
    },
}

def get_order_details(order_id: str) -> dict[str, Any]:
    """Get the details of an order by its order ID"""
    clean_id = order_id.strip().lstrip("#")
    order = ORDERS_DB.get(clean_id)
    if not order:
        return {"error": f"Order #{clean_id} not found in the system"}
    return order

def get_customer_profile(customer_type: str) -> dict[str, Any]:
    """Get the customer profile for a given order by its customer type (VIP or Standard)"""
    tier = customer_type.strip().upper()

    if tier == "VIP":
        profile = CUSTOMER_PROFILES_DB.get("VIP")
    else:
        profile = CUSTOMER_PROFILES_DB.get("Standard")

    if not profile:
        return {"error": f"Customer type '{customer_type}' not found in the system"}
    return profile
