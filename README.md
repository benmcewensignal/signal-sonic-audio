# signal-sonic-audio

The audio pass for Sonic, on its own runner: heavier per-record work that would
otherwise block the main pipeline's chain.

- `audio/olaf.py`  — adapter for Olaf, a fingerprinter built for DJ-set conditions
- `audio/bench.py` — Olaf against our own matcher, both scored on NTS tracklist truth
- next: source separation (Demucs), beat tracking (madmom), stem-level features

Reads the main pipeline's database from `signal-sonic`; writes results to `out/`.
Needs repo secrets `BEATPORT_CLIENT_ID` and `BEATPORT_CLIENT_SECRET` (same values as the main repo).
