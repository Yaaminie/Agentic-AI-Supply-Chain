from __future__ import annotations
import os, json, requests, logging

log = logging.getLogger(__name__)

def send_slack(text: str, webhook_url: str | None = None) -> None:
    webhook = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        log.warning("[SLACK] Missing SLACK_WEBHOOK_URL. Message:\n%s", text)
        return
    payload = {"text": text if len(text) < 2900 else (text[:2800] + "\n…(truncated)")}
    try:
        resp = requests.post(webhook, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=10)
        if resp.status_code >= 300:
            log.error("[SLACK] Error %s: %s", resp.status_code, resp.text)
    except Exception as e:
        log.exception("[SLACK] Post failed: %s", e)
