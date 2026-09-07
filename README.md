# Market Signal Engine

A Python-based system that fetches daily news for a watchlist of stocks, runs them through Claude to generate event-driven investment signals (BUY/HOLD/SELL), and logs the results for manual evaluation.

This is not a trading bot. It does not execute trades.

## How it works

The system fetches the last 24 hours of news and current price for each ticker via Alpha Vantage. Claude analyzes the news and determines whether a concrete, identifiable event justifies a signal. The result is appended as a structured JSONL record to `output/signals.jsonl`.

A signal is only valid if it is driven by a specific event, not general sentiment. On quiet days the system returns HOLD with LOW confidence rather than inventing a catalyst.

## Tech stack

| Component   | Technology                   |
| ----------- | ---------------------------- |
| LLM         | Claude Haiku (Anthropic API) |
| Market data | Alpha Vantage                |
| Language    | Python 3.12                  |

## Getting started

```bash
git clone https://github.com/vita-tak/market-signal-engine.git
cd market-signal-engine

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY and ALPHAVANTAGE_API_KEY to .env
```

## Usage

```bash
# Offline run (fixtures + stub LLM, default)
python main.py

# Real analysis, zero Alpha Vantage quota
USE_FIXTURES=true USE_LLM_STUB=false python main.py

# Full live run
USE_FIXTURES=false USE_LLM_STUB=false python main.py
```

## Running tests

```bash
pytest -q
mypy --strict config.py main.py src/
```
