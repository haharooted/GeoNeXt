import json
import re
import sys
from pathlib import Path

def extract_links(text):
    # Simple regex for most standard URLs
    url_pattern = r'https?://[^\s)]+'
    return re.findall(url_pattern, text)

def find_links_in_geojson(geojson_data):
    features = geojson_data.get("features", [])
    all_links = []

    for feature in features:
        description = feature.get("properties", {}).get("description", "")
        links = extract_links(description)
        unique_links = sorted(set(links))
        all_links.append(unique_links)

    return all_links

def main():
    if len(sys.argv) != 3:
        print("Usage: python find_links.py input.geojson output.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with input_path.open("r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    links_data = find_links_in_geojson(geojson_data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(links_data, f, indent=2)

    print(f"Extracted links saved to {output_path}")

if __name__ == "__main__":
    main()
