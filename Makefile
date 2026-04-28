.PHONY: install scrape serve run test lint typecheck

install:
	python3 -m pip install -r requirements.txt
	python3 -m playwright install chromium

scrape:
	python3 -m scraper.scraper

serve:
	python3 -m mcp_server.server

run:
	python3 -m client.client

test:
	python3 -m pytest tests/ -v --asyncio-mode=auto

lint:
	python3 -m ruff check .

typecheck:
	python3 -m mypy . --strict --ignore-missing-imports