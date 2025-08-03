from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from rich.logging import RichHandler

from geonext import __version__
from geonext.config import LOG_LEVEL, LOG_FILE, DEFAULT_PROVIDER
from geonext.pipeline import run_pipeline
from geonext.providers.openai_provider   import OpenAIProvider
from geonext.providers.mistral_provider  import MistralProvider

# ---------------------------------------------------------------------------
def _setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        handlers=[RichHandler(rich_tracebacks=True),
                  logging.FileHandler(LOG_FILE, encoding="utf-8")],
        datefmt="[%X]"
    )

def _get_provider(name: str):
    if name == "openai":
        return OpenAIProvider()
    if name == "mistral":
        return MistralProvider()
    raise SystemExit(f"Unsupported provider: {name}")

def main(argv=None):
    parser = argparse.ArgumentParser(prog="geonext",
        description="LLM-powered geo-coder for unstructured text")
    parser.add_argument("--input",  required=True, help="JSON file - array of objects")
    parser.add_argument("--output", required=True, help="Where to write results")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER,
                        choices=["openai", "mistral"])
    args = parser.parse_args(argv)

    _setup_logging()
    log = logging.getLogger("geonext.cli")
    log.info("GeoNeXt %s – provider=%s", __version__, args.provider)

    items = json.loads(Path(args.input).read_text())
    provider = _get_provider(args.provider)
    run_pipeline(items=items,
                 provider=provider,
                 out_path=Path(args.output))

if __name__ == "__main__":
    main()
