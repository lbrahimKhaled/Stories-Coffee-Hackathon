import time
import pandas as pd
import requests

PROFILE_CSV = "datasets/branch_profile_group_for_nmf.csv"
OUTPUT_CSV = "datasets/branch_locations.csv"

USER_AGENT = "stories-location-script"  # required by Nominatim


def geocode_branch(branch_name):
    """
    Uses OpenStreetMap Nominatim to geocode branch name.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": branch_name,
        "format": "json",
        "limit": 1
    }

    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        return None, None

    data = response.json()
    if len(data) == 0:
        return None, None

    lat = data[0]["lat"]
    lon = data[0]["lon"]
    return lat, lon


def main():
    df = pd.read_csv(PROFILE_CSV)
    branches = df["Branch"].unique()

    results = []

    for branch in branches:
        print("Geocoding:", branch)
        lat, lon = geocode_branch(branch)

        results.append({
            "Branch": branch,
            "lat": lat,
            "lon": lon
        })

        time.sleep(1)  # IMPORTANT: Nominatim rate limit

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)

    print("Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()