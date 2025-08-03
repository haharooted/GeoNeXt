from __future__ import annotations
import json
from typing import List

LOCATION_SCHEMA={
    "format": {
        "type": "json_schema",
        "name": "geolocation_response",
        "schema": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "description": "One element per UNIQUE real-world location mentioned in the input text.  Empty if none found.",
                    "items": {
                        "type": "object",
                        "properties": {
                            # ── Required core fields ────────────────────────────────────
                            "name":       {
                                "type": "string",
                                "description": "The location mention exactly as it appeared in the input text (case preserved)."
                            },
                            "latitude":   {
                                "type": "number",
                            },
                            "longitude":  {
                                "type": "number",
                            },
                            "confidence": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "Set *confidence* to 10 only when the text explicitly and unambiguously names the exact street, city, and country, and the geocoder result matches this with zero ambiguity. Use 7-9 when fairly sure, 4-6 when uncertain, and 1-3 as a last-resort guess.\n"
                            },
                            "precision":  {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "How specific was the *mention* itself (not what the geocoder returns). 10 = street number present (e.g. “1600 Amphitheatre Pkwy”). 7-9 = street name but no number (e.g. “Amphitheatre Pkwy”). 5-6 = named neighbourhood or POI (e.g. “Silicon Valley”, “Eiffel Tower”). 3-4 = city or town only (e.g. “Toronto”). 1-2 = region/state/country only (e.g. “Ontario”, “Canada”). - 'Toronto' is precision 4 - 'Amphitheatre Pkwy, Mountain View' is precision 8 - '1600 Amphitheatre Pkwy, Mountain View, CA, USA' is precision 10"
                            },

                            # ── Optional but highly useful extras ──────────────────────
                            "address": {
                                "type": "string",
                                "description": "Formatted address returned by the geocoder."
                            },
                            "status": {
                                "type": "string",
                                "enum": ["matched", "guessed"],
                                "description": "matched = high-fidelity match; guessed = best-effort when some ambiguity remained."
                            },
                            "original_snippet": {
                                "type": "string",
                                "description": "Short excerpt (≤80 chars) of input text surrounding the mention (for traceability)."
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "One sentence explanation for the location."
                            }
                        },
                        "required": ["name", "address", "latitude", "longitude",
                                    "confidence", "precision", "original_snippet", "status", "reasoning"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["locations"],
            "additionalProperties": False
        },
        "strict": True
    }
}

SYSTEM_PROMPT = (
    "You are GeoNeXt, an expert location-resolution agent.\n"
    "Goal: read the user text ➔ output ONLY a JSON object that strictly "
    "conforms to the *geolocation_response* schema provided above—no markdown, "
    "no commentary.\n\n"

    "1️⃣  Extraction\n"
    "• Identify every UNIQUE rea-‑world location mention (city, town, POI, street address, landmark, region, country).\n"
    "• Ignore non-geographic entities and fictional places.\n"
    "• Deduplicate (only keep unique locations, but locations can be nearby, just not the exact same location twice).\n\n"

    "2️⃣  Geocoding (one location at a time)\n"
    "• Use the MCP tools provided.\n"
    "• Start with the raw mention.  If context clearly points to a region or country, append it "
    "on the first attempt (e.g. 'Odense' → 'Odense, Denmark').\n"
    "• If the first call is ambiguous or mismatched, refine the query (add region/state/country, "
    "use alternate spellings) and retry—**max 2 extra attempts** per location.\n"
    "• Reason over response metadata and choose one (or none) that best fits the surrounding text.\n\n"

    "4️⃣  Output rules\n"
    "• Include only locations successfully geocoded (skip any failures).\n"
    "• Return the JSON structure with no extra keys and no explanatory prose.\n\n"

    "Think internally; never reveal chain-of-thought.  Do not wrap the JSON in triple back-ticks."
)


# SYSTEM_PROMPT = (
#     "You are *GeoNeXt*, an expert geolocation agent.\n"
#     "Task: read the user text, find every UNIQUE real-world location mentioned."
#     "(city, town, address, landmark, region, country). "
#     "For each found unique location you must:\n"
#     "  • Call the provided geocoding tools (via MCP) **one location at a time**;\n"
#     "  • Append the country/state to the query when obvious (e.g. 'Odense' → "
#     "'Odense, Denmark');\n"
#     "  • If multiple geocoder candidates are returned, reason over the context and response, and pick the best one.\n"
#         "    and pick the best, otherwise refine the query (max 2 extra tries) and query MCP tool again.\n"
#     "  • So to be clear: extract unique locations from the text, geocode each one and return one geolocation (name, address, latitude, longitude, confidence, precision) for each location.\n"
#     "  • If the location cannot be geocoded, then don't include it.\n"
# )

# For providers that *cannot* do tool calling (e.g. today’s Mistral v0.3),
# we instruct them to *only* list distinct 'query' strings in sorted order:
FALLBACK_PROMPT = (
    "You are GeoNeXt-Lite. Extract all distinct locations from the text. "
    "Return a JSON array of strings **only**. "
    "Each string must already include the country/state if obvious. "
    "Do NOT geocode; do NOT add anything else."
)