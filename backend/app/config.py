import os

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Single source of truth for the model. Never hardcode a model id elsewhere.
MODEL = "claude-sonnet-5"

# Per-source content cap (characters). Keeps the synthesis prompt bounded even
# when a scraped page is enormous.
MAX_DOC_CHARS = 12_000

# How many docs the retriever returns for a single Q&A turn.
RETRIEVAL_TOP_K = 4

DB_PATH = os.environ.get("PREP_DB_PATH", "sessions.db")
