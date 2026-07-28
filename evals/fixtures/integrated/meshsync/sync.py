"""Mesh synchronization loop built on the httpx client."""
import httpx

from meshsync.client import pull_snapshot


def sync_nodes(client: httpx.Client, node_ids: list[str],
               dest_dir: str) -> dict[str, str]:
    """Pull every node's snapshot; classify each outcome."""
    outcomes: dict[str, str] = {}
    for node_id in node_ids:
        try:
            pull_snapshot(client, node_id, f'{dest_dir}/{node_id}.snap')
            outcomes[node_id] = 'synced'
        except httpx.TimeoutException:
            outcomes[node_id] = 'timeout'
        except httpx.HTTPStatusError as error:
            outcomes[node_id] = f'http {error.response.status_code}'
    return outcomes
