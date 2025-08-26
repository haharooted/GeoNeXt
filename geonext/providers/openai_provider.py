# providers/openai_provider.py
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, List

import openai
from geonext.prompts import SYSTEM_PROMPT
from geonext.config import MCP_URL, MCP_LABEL
from geonext.utils import extract_json
from geonext.prompts import LOCATION_SCHEMA

log = logging.getLogger("geonext.openai")


class OpenAIProvider:
    """Wraps calls to the OpenAI Responses API and stores each raw HTTP
    response in its own JSON file inside a dedicated *responses/* folder so
    developers can inspect them later.
    """

    MODEL = "o4-mini-2025-04-16"  # any responses‑capable model

    def __init__(self) -> None:
        self.client = openai.OpenAI()
        # Remote‑MCP tool block
        self.mcp_tool: Dict[str, Any] = {
            "type": "mcp",
            "server_url": MCP_URL,
            "server_label": MCP_LABEL,
            "allowed_tools": ["geocode_location"],
            "require_approval": "never",
        }

        # Ensure a folder exists to store the raw JSON responses
        self.responses_dir = os.path.join(
            os.path.dirname(__file__), "..", "responses"
        )
        os.makedirs(self.responses_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _response_filename(self) -> str:
        """Generate a unique filename like *response_20250803_142501_123456.json*."""
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(self.responses_dir, f"response_{ts}.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, text: str) -> List[Dict[str, Any]]:
        """Return GeoNeXt JSON for one document and archive the raw response."""
        try:
            resp = self.client.responses.create(
                model=self.MODEL,
                instructions=SYSTEM_PROMPT,
                input=text,
                tools=[self.mcp_tool],
                tool_choice="auto",
                text=LOCATION_SCHEMA,
                parallel_tool_calls=False,
            )

            # Persist full raw response for debugging
            try:
                with open(self._response_filename(), "w", encoding="utf-8") as fh:
                    json.dump(resp.model_dump(), fh, indent=2, ensure_ascii=False)
            except Exception:
                log.exception("Failed to write response to disk.")

            # Extract the assistant‑formatted JSON payload
            json_str = resp.output[-1].content[0].text
            return json.loads(json_str)

        except Exception:
            # Let the caller decide what to do with unexpected errors
            raise
