import json
import sys
from pathlib import Path

def count_unique_links(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        all_links = json.load(f)

    # Flatten and deduplicate
    flat_links = {link for group in all_links for link in group}
    return len(flat_links), flat_links

def main():
    if len(sys.argv) != 2:
        print("Usage: python counter.py links.json")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    count, unique_links = count_unique_links(json_path)

    print(f"Total unique links: {count}")
    # Uncomment the following line if you want to see all the unique links:
    # print("\n".join(sorted(unique_links)))

if __name__ == "__main__":
    main()
