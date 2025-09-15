from __future__ import annotations
from typing import Dict, Any

def policy_recommendation(payload: Dict[str, Any]) -> str:
    if payload.get("risk_label") != "High":
        return "Notify & monitor"

    pdl = payload.get("promised_days_left", 999)
    stock = payload.get("dc_stock", 0)
    priority = str(payload.get("item_priority","")).lower()
    tier = str(payload.get("customer_tier","")).upper()

    if pdl <= 1:
        return "Expedite to 2-day"
    if stock and stock > 0:
        return "Split-ship from alternate DC"
    if priority in {"low","normal"}:
        return "Notify customer & propose partial defer"
    if tier in {"GOLD","PLATINUM"}:
        return "Expedite to meet SLA"
    return "Escalate to human planner"
