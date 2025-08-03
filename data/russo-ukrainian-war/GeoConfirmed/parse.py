#!/usr/bin/env python3
"""
Stream-convert a (possibly huge) KML file to GeoJSON,
keeping only <Placemark> elements that contain a <Point>.

Usage:
    python kml_points_to_geojson_stream.py  input.kml  output.geojson
"""
from __future__ import annotations
import json, sys, os, codecs, xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

NS = {"kml": "http://www.opengis.net/kml/2.2"}

def sniff_encoding(fp: Path) -> str:
    """
    Look at the first four bytes and guess UTF-8-SIG, UTF-16-LE/BE or fallback to UTF-8.
    """
    with fp.open("rb") as f:
        bom = f.read(4)
    if bom.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if bom.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    if bom.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    return "utf-8"

def convert(kml_path: str, geojson_path: str) -> None:
    enc = sniff_encoding(Path(kml_path))
    features: List[Dict] = []
    opened = open(kml_path, "r", encoding=enc, errors="replace")

    try:
        # We only care about the *end* event on Placemark,
        # which keeps memory consumption tiny.
        for event, elem in ET.iterparse(opened, events=("end",)):
            if not elem.tag.endswith("Placemark"):
                continue
            if elem.find(".//kml:Point", NS) is None:
                elem.clear(); continue

            coord_el = elem.find(".//kml:Point/kml:coordinates", NS)
            if coord_el is None or not coord_el.text:
                elem.clear(); continue

            lon, lat, *_ = map(float, coord_el.text.strip().split(","))
            desc_el = elem.find("kml:description", NS)
            when_el = elem.find(".//kml:TimeStamp/kml:when", NS)
            doodoo=(desc_el.text or "").strip() if desc_el is not None else None
            if doodoo is None:
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "description": (desc_el.text or "").strip() if desc_el is not None else None,
                    "timestamp":   when_el.text if when_el is not None else None,
                    "lat": lat,
                    "lon": lon,
                },
            })

            elem.clear()      # free memory
    except ET.ParseError as e:
        # Keep whatever we managed to parse; warn the user.
        print(f"⚠️  XML stopped being well-formed at {e}. "
              f"Keeping {len(features)} features collected so far.", file=sys.stderr)
    finally:
        opened.close()

    # ---------- write GeoJSON -------------
    out = {"type": "FeatureCollection", "features": features}
    Path(geojson_path).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(f"✅  Wrote {len(features):,} point features →  {geojson_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python kml_points_to_geojson_stream.py <input.kml> <output.geojson>")
    convert(sys.argv[1], sys.argv[2])
