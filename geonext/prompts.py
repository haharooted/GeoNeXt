from __future__ import annotations
import json

LOCATION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name":       {"type": "string"},
            "latitude":   {"type": "number"},
            "longitude":  {"type": "number"},
            "address":    {"type": "string"},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
            "precision":  {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["name", "latitude", "longitude", "address",
                     "confidence", "precision"],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = (
    "You are *GeoNeXt*, an expert geopolitical agent.\n"
    "Task: read the user document, extract every UNIQUE, real-world location "
    "(city, town, address, landmark, region, country). "
    "For each you must:\n"
    "  • Call the provided geocoding tools (via MCP) **one location at a time**;\n"
    "  • Append the country/state to the query when obvious (e.g. 'Odense' → "
    "'Odense, Denmark');\n"
    "  • If multiple geocoder candidates are returned, reason over the context "
    "    and pick the best, otherwise refine the query (max 2 extra tries).\n"
    "  • Drop exact duplicates (same lat/lon ±1 m).\n"
    "  • Produce ONLY JSON that validates against the schema the tool call "
    "    specifies. If no places found, output [] and nothing else."
)

# For providers that *cannot* do tool calling (e.g. today’s Mistral v0.3),
# we instruct them to *only* list distinct 'query' strings in sorted order:
FALLBACK_PROMPT = (
    "You are GeoNeXt-Lite. Extract all distinct locations from the text. "
    "Return a JSON array of strings **only**. "
    "Each string must already include the country/state if obvious. "
    "Do NOT geocode; do NOT add anything else."
)

# serialised schema for OpenAI function-calling
OPENAI_TOOL = {
    "type": "mcp",
    "server_url": "<DYNAMICALLY_FILLED>",
    "server_label": "<DYNAMICALLY_FILLED>",
    "require_approval": "never"
}
