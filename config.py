from dotenv import load_dotenv
load_dotenv()           # now .env is honoured automatically
from __future__ import annotations
import os
from pathlib import Path

# MCP / Geocoder
MCP_URL         = os.getenv("MCP_URL", "http://localhost:8000/mcp/")
MCP_LABEL       = os.getenv("MCP_LABEL", "geonext")

# LLM provider pick
DEFAULT_PROVIDER = os.getenv("GEONEXT_PROVIDER", "openai").lower()

# Output flushing
FLUSH_EVERY     = int(os.getenv("GEONEXT_FLUSH", "1"))   # flush after N items

# Logging
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE        = Path(os.getenv("LOG_FILE", "geonext.log"))
