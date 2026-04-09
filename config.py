from dotenv import load_dotenv
import os

load_dotenv()

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_MODEL    = "llama-3.3-70b-versatile"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

# ── Embeddings & Vector Store ─────────────────────────────────────────────────
EMBED_MODEL       = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH  = "./faiss_index"
KEY_FILE          = "./.esg_secret.key"

# ── Roles ─────────────────────────────────────────────────────────────────────
ROLES = {
    "guest":   {"can_query": True, "see_financial": False},
    "analyst": {"can_query": True, "see_financial": False},
    "admin":   {"can_query": True, "see_financial": True},
}

# ── Sensitive metadata keys (Fernet-encrypted before FAISS indexing) ──────────
SENSITIVE_META_KEYS = {
    "investment_target",
    "budget_allocation",
    "internal_cost",
    "financial_projection",
}

# ── Presidio confidence threshold ─────────────────────────────────────────────
PRESIDIO_SCORE_THRESHOLD = 0.75