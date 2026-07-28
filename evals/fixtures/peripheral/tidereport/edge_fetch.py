"""The single httpx edge: fetch raw tide readings over HTTP.

Everything else in tidereport is HTTP-free; httpx is used here the way a
peripheral consumer uses an environment — one wrapper at the boundary.
"""
import httpx


def fetch_readings(station: str, day: str) -> list[tuple[str, float]]:
    with httpx.Client(base_url='https://tides.example', timeout=10.0) as client:
        response = client.get(f'/readings/{station}', params={'day': day})
        response.raise_for_status()
        payload = response.json()
    return [(row['hour'], row['level']) for row in payload['readings']]
