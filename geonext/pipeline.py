from __future__ import annotations
import json, logging, itertools
from pathlib import Path
from geonext.utils import deep_to_str
from geonext.config import FLUSH_EVERY, STOP_ON_ERROR
from tqdm import tqdm

log = logging.getLogger("geonext.pipeline")

def run_pipeline(*,
                 items: list[dict],
                 provider,
                 out_path: Path):

    results = []
    if out_path.exists():                         # resume
        results = json.loads(out_path.read_text())

    start_idx = len(results)
    log.info("Resuming at index %s / %s", start_idx, len(items))

    bar = tqdm(total=len(items), initial=start_idx, unit="doc")

    for idx in range(start_idx, len(items)):
        item = items[idx]
        text = deep_to_str(item)                  # flatten nested JSON->str
        try:
            log.info("Processing item %s:\n%s", idx, text)
            locs = provider.run(text=text)
        except Exception as exc:
            log.exception("Provider failed on index %s: %s", idx, exc)
            if STOP_ON_ERROR==1:
                raise
            locs = {"error": str(exc)}

        results.append(locs)

        if (idx + 1) % FLUSH_EVERY == 0:
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        bar.update(1)

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    bar.close()
