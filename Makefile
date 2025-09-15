.PHONY: setup run lint test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && python -m agentic_supply_chain.app

test:
	. .venv/bin/activate && pytest -q
