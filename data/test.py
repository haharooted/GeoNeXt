import requests

url = (
    "https://api.dataforsyningen.dk/adgangsadresser"
    "?bebyggelsestype=by"
    "&status=1"
    "&struktur=mini"
    "&ndjson"
    "&srid=4326"
    "&polygon=%7B%22type%22%3A%22Polygon%22%2C%22coordinates%22%3A%5B%5B%5B12.56553%2C55.68356%5D%2C%5B12.56611%2C55.68356%5D%2C%5B12.56611%2C55.68322%5D%2C%5B12.56553%2C55.68322%5D%2C%5B12.56553%2C55.68356%5D%5D%5D%7D"
)

headers = {
    "Accept-Encoding": "gzip",
    "User-Agent": "dk-addr-sampler/1.1 (polygon)"
}

# Make request
response = requests.get(url, headers=headers, stream=True)

# Check response
if response.status_code == 200:
    for line in response.iter_lines(decode_unicode=True):
        if line:
            print(line)
else:
    print(f"Request failed: {response.status_code} - {response.text}")
