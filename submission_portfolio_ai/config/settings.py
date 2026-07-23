import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the project root (portfolio_ai/.env)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# Constants
DB_PATH = "portfolio_ai.db"
RISK_FREE_RATE = 0.04
BENCHMARK_TICKER = "SPY"
MONTE_CARLO_SIMULATIONS = 1000
TRADING_DAYS = 252

# Robinhood credentials (provided via .env)
ROBINHOOD_USERNAME = os.getenv("ROBINHOOD_USERNAME", "")
ROBINHOOD_PASSWORD = os.getenv("ROBINHOOD_PASSWORD", "")

# Financial Modeling Prep (FMP)
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

# LLM providers
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_LLM = os.getenv("DEFAULT_LLM", "anthropic")
