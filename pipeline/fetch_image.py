import requests
import os


def fetch_static_map(lat, lon, out_path, zoom=20, size="640x640"):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if api_key is None:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY environment variable not set"
        )
    
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}"
        f"&maptype=satellite&key={api_key}"
    )

    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    print("Error:", r.text)
    return False
