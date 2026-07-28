"""Mesh node client — httpx woven through the product (the integrated
archetype: the environment IS part of the artifact)."""
import httpx


class NodeAuth(httpx.Auth):
    """Per-request token auth implemented on httpx's Auth surface."""

    def __init__(self, token: str):
        self._token = token

    def auth_flow(self, request):
        request.headers['X-Node-Token'] = self._token
        yield request


def make_client(base_url: str, token: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        auth=NodeAuth(token),
        timeout=httpx.Timeout(5.0, connect=2.0),
        transport=httpx.HTTPTransport(retries=2),
    )


def pull_snapshot(client: httpx.Client, node_id: str, dest_path: str) -> int:
    """Stream a node snapshot to disk without loading it into memory."""
    written = 0
    with client.stream('GET', f'/nodes/{node_id}/snapshot') as response:
        response.raise_for_status()
        with open(dest_path, 'wb') as out:
            for chunk in response.iter_bytes():
                out.write(chunk)
                written += len(chunk)
    return written
