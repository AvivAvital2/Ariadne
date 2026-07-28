"""Station-status fetcher — stdlib HTTP only (the adopter archetype:
this codebase has ZERO environment calls; battery questions ask about
migrating it)."""
import json
import urllib.request


def fetch_station_status(station_id: str, timeout: float = 10.0) -> dict:
    url = f'https://relay.example/stations/{station_id}/status'
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def fetch_all(station_ids: list[str]) -> dict[str, dict]:
    """Sequential fetch — one blocking request per station."""
    return {sid: fetch_station_status(sid) for sid in station_ids}
