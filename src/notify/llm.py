from __future__ import annotations
from typing import Dict, Any, List, Optional
import os

def _format_top_features(feats: List[Dict[str, Any]]) -> str:
    lines = []
    for f in feats[:4]:
        lines.append(f"- {f.get('name')}={f.get('value')} ({f.get('direction','')})")
    return "\n".join(lines) if lines else "- n/a"

def _fallback_manager(ctx: Dict[str, Any]) -> str:
    p = ctx.get("p_high", 0.0)
    action = ctx.get("policy_recommendation","Notify & monitor")
    promised = ctx.get("promised_by","n/a")
    feat_names = ", ".join([f['name'] for f in ctx.get('top_features', [])[:3]]) or "supply/route conditions"
    return (
        f"- Risk: **High** (p={p:.2f}). Promised-by: **{promised}** at risk.\n"
        f"- Drivers: {feat_names}.\n"
        f"- Recommend: {action}. Tradeoff: minor cost increase vs. SLA protection.\n"
        f"- Next: confirm or propose alternative."
    )

def _fallback_customer(ctx: Dict[str, Any]) -> str:
    promised = ctx.get("promised_by","your promised date")
    return (
        f"Quick update: your order is seeing minor route delays. "
        f"We’re taking steps to keep your delivery on time for {promised}. "
        f"If timing is critical, reply 'ALT' and we’ll share options."
    )

def llm_manager_summary(context: Dict[str, Any]) -> str:
    prompt_user = f"""
RISK:
label={context.get('risk_label')}  p_high={context.get('p_high'):.2f}

TOP_FEATURES (name=value,direction):
{_format_top_features(context.get('top_features', []))}

CONTEXT:
order_id={context.get('order_id')}
customer_tier={context.get('customer_tier')}
promised_by={context.get('promised_by')}
item_priority={context.get('item_priority')}
policy_recommendation={context.get('policy_recommendation')}

STYLE:
- 3–5 bullet points
- lead with risk & SLA impact
- one-line tradeoff if recommending action
- <= 120 words
""".strip()

    sys = "You are an operations planner. Be crisp, data-driven, and actionable."
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_manager(context)

    try:
        from openai import OpenAI  # defer import so repo installs even without
        client = OpenAI(api_key=api_key)  # type: ignore
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt_user}],
            temperature=0.2, max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _fallback_manager(context)

def llm_customer_message(context: Dict[str, Any]) -> str:
    prompt_user = f"""
ORDER: {context.get('order_id')}
PROMISED_BY: {context.get('promised_by')}
STATUS: Potential delay risk due to route conditions.
ACTION: {context.get('policy_recommendation','We’re taking steps')} to keep delivery on time.
CTA: If timing is critical, reply 'ALT' to discuss options.

Write 2–3 short sentences. No internal jargon. Friendly, reassuring tone.
""".strip()
    sys = "You are a helpful customer support assistant. Keep it concise and empathetic."
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_customer(context)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)  # type: ignore
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt_user}],
            temperature=0.3, max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return _fallback_customer(context)
