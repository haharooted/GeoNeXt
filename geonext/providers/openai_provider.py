from __future__ import annotations
import openai, json, logging
from geonext.prompts import SYSTEM_PROMPT, LOCATION_SCHEMA, OPENAI_TOOL
from geonext.config  import MCP_URL, MCP_LABEL

log = logging.getLogger("geonext.openai")

class OpenAIProvider:
    MODEL = "gpt-4o-mini"          # choose any tools-capable model

    def __init__(self):
        self.client = openai.OpenAI()
        # patch the MCP tool block with runtime values
        self.tool_block = {**OPENAI_TOOL,
                           "server_url": MCP_URL,
                           "server_label": MCP_LABEL}

    def run(self, *, text: str) -> list[dict]:
        """Ask GPT-4o, let it invoke MCP tools, return JSON list."""
        resp = self.client.responses.create(
            model=self.MODEL,
            tools=[self.tool_block],
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0,
        )
        msg = resp.choices[0].message
        try:
            return json.loads(msg.content)
        except Exception as exc:
            log.error("Provider returned non-JSON: %s", exc)
            raise
