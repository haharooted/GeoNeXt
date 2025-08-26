#!/usr/bin/env python3
"""
sample_polygon_addresses.py
---------------------------
Uniformly sample Danish city addresses that fall **inside a user-defined polygon**
and save them as GeoJSON in WGS-84 / EPSG 4326.

• Requires Python ≥3.8 and the packages: requests, geojson
    $ pip install requests geojson
"""

import argparse
import gzip
import json
import os
import random
import sys
import urllib.parse

import requests
import geojson


# ----------------------------------------------------------------------
# 1.  Paste your polygon here  (WGS-84 lon/lat)
POLYGON = [[
    [11.2304526349009,55.227471258492415],
    [10.866726573413871,54.84603985400824],
    [10.983115039303897,54.627644081854754],
    [11.705084419717991,54.58304511325272],
    [12.02881515589985,54.57672071685647],
    [12.49804285216348,54.80797399845173],
    [12.87633494837641,54.937736608166745],
    [12.647177236246677,55.18978143829122],
    [12.54532934644888,55.33693746802456],
    [12.48349312038252,55.48246891236897],
    [12.752662484341243,55.519552628937504],
    [12.749025060504664,55.6408601181823],
    [12.647173673356662,55.802701097479286],
    [12.632614345861754,56.03099245250834],
    [12.578052508000724,56.10205784399085],
    [12.323442107053154,56.150719517136594],
    [11.898062431407453,56.177695607997464],
    [11.40337280042209,56.11487984174482],
    [11.199677070016492,56.03773974212399],
    [10.999618763367408,55.89523498830053],
    [10.843209538930353,55.8196988901754],
    [10.737724277360599,55.70919327517663],
    [10.863552237284296,55.48649014678924],
    [11.2304526349009,55.227471258492415]      # closed ring
]]

# ----------------------------------------------------------------------
# 2.  Build the DAWA URL
API_BASE = (
    "https://api.dataforsyningen.dk/adgangsadresser"
    "?bebyggelsestype=by"      # only built-up areas
    "&status=1"                # current, valid addresses
    "&struktur=mini"           # lean JSON
    "&ndjson"           # newline-delimited JSON stream
    "&srid=4326"               # WGS-84 coords
)

polygon_param = urllib.parse.quote(
    json.dumps(POLYGON, separators=(",", ":"))
)
API_URL = f"{API_BASE}&polygon={polygon_param}"

HEADERS = {
    "Accept-Encoding": "gzip",
    "User-Agent": "dk-addr-sampler/1.1 (polygon)"
}


# ----------------------------------------------------------------------
# 3.  Reservoir sampling from the polygon-filtered stream
def reservoir_sample(url: str, k: int, seed: int | None = None) -> list[dict]:
    if seed is not None:
        random.seed(seed)

    sample: list[dict] = []

    with requests.get(url, stream=True, headers=HEADERS, timeout=120) as resp:
        resp.raise_for_status()
        stream = gzip.GzipFile(fileobj=resp.raw)      # DAWA gzips NDJSON

        for i, line in enumerate(stream):
            rec = json.loads(line)
            if i < k:
                sample.append(rec)
            else:
                j = random.randint(0, i)
                if j < k:
                    sample[j] = rec

    # If the polygon contains fewer than k addresses, just return what we got
    return sample


# ----------------------------------------------------------------------
# 4.  Convert to GeoJSON
def to_feature(addr: dict) -> geojson.Feature:
    lon, lat = addr["x"], addr["y"]
    print(lon, lat)
    props = {
        "id": addr["id"],
        "vejnavn": addr.get("vejnavn"),
        "husnr": addr.get("husnr"),
        "postnr": addr.get("postnr"),
        "postnrnavn": addr.get("postnrnavn"),
        "betegnelse": addr.get("betegnelse"),
        "lat": lat,
        "lon": lon
    }
    return geojson.Feature(geometry=geojson.Point((lon, lat)), properties=props)

# ----------------------------------------------------------------------
# 5.  CLI wrapper
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample Danish city addresses INSIDE a polygon and write GeoJSON."
    )
    parser.add_argument(
        "-n", "--number", type=int, default=50_000,
        help="Number of random addresses to keep (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output", default="polygon_50k_addresses.geojson",
        help="Output GeoJSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sampling (default: system-random)",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------- download
    print("→ Downloading and sampling inside polygon …", file=sys.stderr)
    sample = reservoir_sample(API_URL, args.number, args.seed)
    print(f"   … kept {len(sample):,} addresses.", file=sys.stderr)

    # ----------------------------------------------------------- to GeoJSON
    features = [to_feature(a) for a in sample]
    fc = geojson.FeatureCollection(features)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)

    print(
        f"✓ Wrote {args.output}  ({os.path.getsize(args.output):,} bytes)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
