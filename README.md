# Agentic Supply Chain Risk (Dash + XGBoost + SHAP + Slack + LLM)

A compact, demo-ready platform for **logistics risk detection** with:
- **EDA & Modeling UI** (Dash) — explore data, train **XGBoost**, inspect **SHAP** explanations
- **Agentic notifications** — auto-decide an action, compose **manager** and **customer** updates via LLM (with fallback), and push to **Slack**
- **Tiny REST endpoint** — `/agent/summarize` for external systems (e.g., MuleSoft) to obtain action + messages on demand

## Why this project?
- Showcases an **end-to-end** stack (data → model → explainability → action → comms)
- Built for interviews: clean code, typed config, logging, tests, and integration points
- Minimal infra assumptions; runs locally with Python

---

## Features

- **EDA**: missingness, correlations, histograms, optional lat/lon mapping  
- **Modeling**: class weights, median imputation, train/valid curves, metrics (Accuracy, **Macro-F1**, AUC)  
- **Explainability**: global SHAP, dependence plots (multi-class friendly)  
- **Agentic Step**: policy rules map context → recommended action  
- **Messaging**: LLM-crafted Slack updates (with safe fallback copy if no API key)  
- **API**: `POST /agent/summarize` returns `{recommended_action, manager_message, customer_message}`

---

## Requirements

- Python **3.10+**
- (Optional) Slack Incoming Webhook
- (Optional) OpenAI API key (for LLM summaries)

See `requirements.txt` for Python deps.

---

## Quickstart

```bash
git clone <your-fork-url>
cd agentic-supply-risk

# 1) Python env + deps
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 2) Configure environment
cp .env.sample .env
# edit .env to set CSV_PATH, SLACK_WEBHOOK_URL, OPENAI_API_KEY if available

# 3) Run the app
python -m supplychain_risk.web.app
# open http://localhost:8050


# how to create a slack channel 

step 1 - create workspace, then add channel to that, then copy the incoming webhook 