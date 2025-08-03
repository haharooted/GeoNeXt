from __future__ import annotations
import json, logging, requests, os
from mistralai.client import MistralClient
from geonext.prompts  import FALLBACK_PROMPT
from geonext.config   import MCP_URL

log = logging.getLogger("geonext.mistral")

def _call_mcp(location: str) -> dict|None:
    """Minimal HTTP client for /call/geocode_location."""
    payload = {
        "tool_name": "geocode_location",
        "args":      {"location": location, "max_results": 1}
    }
    try:
        r = requests.post(f"{MCP_URL.rstrip('/')}/call",
                          json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["result"][0] if data["result"] else None
    except Exception as exc:
        log.warning("MCP call failed: %s", exc)
        return None

class MistralProvider:
    MODEL = "mistral-large-latest"

    def __init__(self):
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

    def run(self, *, text: str) -> list[dict]:
        # 1) ask Mistral to *only* list query strings
        resp = self.client.chat(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": FALLBACK_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0
        )
        try:
            queries = json.loads(resp.choices[0].message.content)
        except Exception as exc:
            log.error("Bad JSON from Mistral: %s", exc)
            raise

        # 2) de-dupe & geocode via MCP ourselves
        seen, results = set(), []
        for q in sorted(set(queries)):
            if (geo := _call_mcp(q)):
                key = (round(float(geo["latitude"]), 4),
                       round(float(geo["longitude"]), 4))
                if key in seen:
                    continue
                seen.add(key)
                geo.update({
                    "name": q,
                    "confidence": 7,        # heuristics – we picked top-1
                    "precision": 5,         # or inspect address details…
                })
                results.append(geo)
        return results
