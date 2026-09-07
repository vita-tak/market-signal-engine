import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID")

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

WATCHLIST = [
    "AROC",
    "CRWD",
    "CRSP",
    "IRDM",
]

NEWS_LIMIT = 10
NEWS_DAYS_BACK = 1