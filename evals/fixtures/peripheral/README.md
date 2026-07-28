# tidereport

Builds nightly tide summaries per station. Readings are fetched with
**httpx** (see `tidereport/edge_fetch.py`); everything downstream of the
httpx fetch is plain Python.

- httpx does the HTTP: connection pooling, timeouts, and raising on bad
  statuses all come from the httpx client.
- The nightly job calls the httpx edge once per station, then the summary
  and rendering steps run without httpx.
- If a station is missing, the httpx call raises and the station is
  skipped in that night's report.

(This repo is a synthetic eval fixture: its prose deliberately saturates
on the environment's name — the peripheral-archetype trap where the
consumer's own catalog resolves environment vocabulary.)
