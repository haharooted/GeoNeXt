#!/usr/bin/env python3
"""convert_json_to_geojson.py

Convert a JSON file containing an array of items, each with a geolocation.locations
array of latitude/longitude dictionaries, into a GeoJSON FeatureCollection.

Each location becomes an individual Feature whose geometry is a Point and whose
properties contain **all** fields from the parent item *plus* all fields from
that location object.  Nested structures are flattened into dot‑separated keys.
The output file name is derived from the input file by replacing the .json
extension with .geojson unless an explicit output file is supplied.

Usage
-----
$ python convert_json_to_geojson.py incidents.json            # writes incidents.geojson
$ python convert_json_to_geojson.py incidents.json out.geojson
"""

import json
import os
import sys
from typing import Any, Dict, List

SEP = "."


def flatten(obj: Any, parent_key: str = "", sep: str = SEP) -> Dict[str, Any]:
    """Recursively flattens nested dicts/lists into a single‑level dict.

    Lists are enumerated with numeric indices so that each element becomes a
    separate key, e.g. ``tags.0``, ``tags.1``.  The default separator is ``.``.
    """
    items: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            items.update(flatten(v, new_key, sep=sep))
    else:
        # Primitive value
        items[parent_key] = obj
    return items


def build_features(data: List[dict]) -> List[dict]:
    """Return a list of GeoJSON Feature dicts from the input data list."""
    features: List[dict] = []
    for item in data:
        # Copy so we can pop without mutating original
        parent = dict(item)
        geo = parent.pop("geolocation", {})
        base_props = flatten(parent)

        for loc in geo.get("locations", []):
            # Build merged properties
            loc_props = flatten(loc)
            properties = {**base_props, **loc_props}

            # Geometry: lon, lat order per GeoJSON spec
            try:
                lon = float(loc["longitude"])
                lat = float(loc["latitude"])
            except (KeyError, ValueError, TypeError):
                # Skip malformed coordinates
                continue

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": properties,
            }
            features.append(feature)
    return features


def main(argv: List[str]) -> None:
    if len(argv) < 2 or len(argv) > 3:
        print("Usage: python convert_json_to_geojson.py <input.json> [output.geojson]", file=sys.stderr)
        sys.exit(1)

    in_path = argv[1]
    if not os.path.isfile(in_path):
        print(f"Error: '{in_path}' does not exist or is not a file", file=sys.stderr)
        sys.exit(1)

    if len(argv) == 3:
        out_path = argv[2]
    else:
        base, _ = os.path.splitext(in_path)
        out_path = f"{base}.geojson"

    # Load input
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: root JSON element must be a list of objects", file=sys.stderr)
        sys.exit(1)

    features = build_features(data)
    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    # Write output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feature_collection, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(features)} features to '{out_path}'")


if __name__ == "__main__":
    main(sys.argv)
