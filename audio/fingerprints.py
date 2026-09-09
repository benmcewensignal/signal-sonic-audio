"""Constellation fingerprinting: find our known tracks inside mix audio.

Shazam-family approach, self-contained (numpy + librosa only):
  1. spectrogram -> local peaks ("constellation")
  2. anchor->target peak pairs -> 32-bit hashes (f1, f2, dt quantised)
  3. reference previews indexed into SQLite (fingerprints table)
  4. a mix is hashed the same way; matches vote on (track, time-offset);
     a tight offset histogram peak = the track is playing there

Honest limits, stated up front:
  - Robust to noise, EQ, compression, and crowd bleed.
  - Degrades under tempo shift: at precision-safe thresholds the rate
    sweep reliably recovers ~±2%; beyond that misses. Recall is a floor,
    not a census — fine for trend weighting, wrong for royalties.
  - Matches only what we have indexed: charting/release previews. The
    unmatched remainder of a mix is itself signal (unreleased density).
"""
from __future__ import annotations
import hashlib
import numpy as np

# spectrogram / peak parameters
SR = 22050
N_FFT = 2048
HOP = 512                      # ~23ms per frame
PEAK_NEIGH_T = 12              # frames (~0.28s) local-max window
PEAK_NEIGH_F = 20              # bins
MIN_PEAK_DB = -38.0            # relative to file max (stricter = leaner index)
# pairing parameters
FANOUT = 6                     # targets per anchor
DT_MIN, DT_MAX = 2, 80         # frames (~0.05s .. ~1.9s)
DT_QUANT = 2                   # frames per dt bucket: tempo tolerance
FREQ_QUANT = 2                 # bins per freq bucket
# matching parameters
MATCHER_V = 2                  # bump when the matcher changes: mixes with a lower matcher_v are re-scored
MIN_VOTES = 60                 # aligned hash votes: real-audio calibrated.
MATCHER_ID = "w1"              # bump when the matcher changes; rescan re-scores only mixes not on this id
MIN_WVOTES = 12.0              # rarity-weighted votes at the peak offset (weight = 1/log2(1+df)); calibrated against the chronological null, see mix_plays.wvotes
DF_MAX = 2000                  # drop only truly ubiquitous hashes (grammar); the rest vote weighted by rarity: grammar hashes (kick/hat/tempo regularities shared by thousands of records) vote for everyone and calibrate to nothing. Commons analysis: 93% of hashes recur in >=10 records; discriminating power lives in the rare tail.
                               # Live control group (Vietnamese mixes, ~zero
                               # true overlap with our corpus) matched ~1.2
                               # tracks/min at a floor of 15: real collisions
                               # cluster just above it; true plays land in
                               # the hundreds.
VERIFY_TOL_S = 1.2             # alignment residual tolerance
VERIFY_MIN_INLIERS = 60        # aligned pairs on the line required
VERIFY_MIN_SPAN_S = 15.0       # contiguous reference coverage required
NMS_MIN_GAP_S = 35.0           # two claims within this gap compete: the
                               # stronger owns the position (clone control)
MAX_DF_MIN = 12                # a hash in more tracks than max(this,
MAX_DF_FRAC = 0.02             # frac*n_tracks) is furniture, not identity
DOMINANCE = 2.5                # peak offset bin must beat runner-up by this
OFFSET_BIN_S = 1.0             # offset histogram resolution
RATE_SWEEP = (0.96, 0.98, 1.0, 1.02, 1.04)  # query-side tempo tolerance


def _local_max_filter(A: np.ndarray, wt: int, wf: int) -> np.ndarray:
    """Neighbourhood max via separable shift-max passes (vectorised)."""
    M = A
    for axis, w in ((0, wt), (1, wf)):
        out = M.copy()
        for k in range(1, w + 1):
            sl_f = [slice(None)] * 2
            sl_b = [slice(None)] * 2
            sl_f[axis] = slice(k, None)
            sl_b[axis] = slice(None, -k)
            np.maximum(out[tuple(sl_f)], M[tuple(sl_b)], out=out[tuple(sl_f)])
            np.maximum(out[tuple(sl_b)], M[tuple(sl_f)], out=out[tuple(sl_b)])
        M = out
    return M


def _peaks(y: np.ndarray) -> list[tuple[int, int]]:
    S = np.abs(np.fft.rfft(_frames(y), axis=1))
    Sdb = 20 * np.log10(S + 1e-9)
    Sdb -= Sdb.max()
    neigh = _local_max_filter(Sdb, PEAK_NEIGH_T, PEAK_NEIGH_F)
    mask = (Sdb >= neigh - 1e-9) & (Sdb >= MIN_PEAK_DB)
    mask[:, :2] = False
    mask[:, -2:] = False
    ts, fs = np.nonzero(mask)
    order = np.argsort(ts, kind="stable")
    return list(zip(ts[order].tolist(), fs[order].tolist()))


def _frames(y: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(y) - N_FFT)) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    w = np.hanning(N_FFT)
    return y[idx] * w


CHUNK_S = 300          # hash long audio in 5-min blocks
CHUNK_OVERLAP_S = 4    # overlap so pairs spanning a boundary survive


def hashes_long(y: np.ndarray) -> list[tuple[int, int]]:
    """Chunked hashing for mix-length audio: constant memory, global frames."""
    if len(y) <= (CHUNK_S + CHUNK_OVERLAP_S) * SR:
        return hashes(y)
    out = []
    step = CHUNK_S * SR
    win = (CHUNK_S + CHUNK_OVERLAP_S) * SR
    for start in range(0, len(y), step):
        seg = y[start:start + win]
        if len(seg) < N_FFT * 2:
            break
        base = start // HOP
        for h, t in hashes(seg):
            out.append((h, t + base))
    return out


def hashes(y: np.ndarray) -> list[tuple[int, int]]:
    """Return [(hash32, anchor_frame)] for one audio buffer (mono, SR)."""
    pk = _peaks(y)
    out = []
    for i, (t1, f1) in enumerate(pk):
        made = 0
        for t2, f2 in pk[i + 1:]:
            dt = t2 - t1
            if dt < DT_MIN:
                continue
            if dt > DT_MAX:
                break
            h = (int(f1 // FREQ_QUANT) & 0x3FF) << 22 \
                | (int(f2 // FREQ_QUANT) & 0x3FF) << 12 \
                | (int(dt // DT_QUANT) & 0xFFF)
            out.append((h, t1))
            made += 1
            if made >= FANOUT:
                break
    return out


def load_audio(path: str, max_seconds: float | None = None) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True, duration=max_seconds)
    m = np.max(np.abs(y)) or 1.0
    return (y / m).astype(np.float32)


# ---------------------------------------------------------------------------
# index (SQLite, same db as everything else)
# ---------------------------------------------------------------------------

def open_store(path: str = "fingerprints.db"):
    """The fingerprint index lives in its OWN file, never in the
    git-committed sonic.db: at full corpus it is hundreds of MB and
    GitHub rejects files over 100 MB. It persists as a release asset."""
    import sqlite3
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema(conn)
    return conn


FP_SCHEMA = """
CREATE TABLE IF NOT EXISTS fp_tracks (
    track_id  TEXT PRIMARY KEY,
    n_hashes  INTEGER NOT NULL,
    hashes    BLOB NOT NULL,             -- uint32[n] packed
    frames    BLOB NOT NULL,             -- uint16[n] packed
    indexed_at REAL NOT NULL
);
"""


def ensure_schema(conn):
    conn.executescript(FP_SCHEMA)


def index_track(conn, track_id: str, y: np.ndarray) -> int:
    """Hash one reference clip into the store as packed blobs (~6 bytes per
    hash vs ~35+ as rows: the difference between a corpus index that fits a
    release asset and one that does not)."""
    ensure_schema(conn)
    if conn.execute("SELECT 1 FROM fp_tracks WHERE track_id=?",
                    (track_id,)).fetchone():
        return 0
    hs = hashes(y)
    if not hs:
        return 0
    H = np.array([h for h, _ in hs], dtype=np.uint32)
    T = np.array([min(t, 65535) for _, t in hs], dtype=np.uint16)
    import time
    conn.execute("INSERT INTO fp_tracks VALUES (?,?,?,?,?)",
                 (track_id, len(hs), H.tobytes(), T.tobytes(), time.time()))
    conn.commit()
    _CACHE.pop(id(conn), None)   # index changed; rebuild on next match
    return len(hs)


_CACHE: dict[int, tuple] = {}


def _load_index(conn):
    """Build the in-memory search index once per connection: all hashes
    concatenated and sorted, with parallel track-idx and frame arrays."""
    key = id(conn)
    if key in _CACHE:
        return _CACHE[key]
    tids, Hs, Ts, Tidx = [], [], [], []
    for i, row in enumerate(conn.execute(
            "SELECT track_id, n_hashes, hashes, frames FROM fp_tracks")):
        tids.append(row[0])
        h = np.frombuffer(row[2], dtype=np.uint32)
        t = np.frombuffer(row[3], dtype=np.uint16)
        Hs.append(h)
        Ts.append(t)
        Tidx.append(np.full(len(h), i, dtype=np.int32))
    if not tids:
        _CACHE[key] = (tids, None, None, None)
        return _CACHE[key]
    H = np.concatenate(Hs)
    T = np.concatenate(Ts).astype(np.int32)
    X = np.concatenate(Tidx)
    order = np.argsort(H, kind="stable")
    H, T, X = H[order], T[order], X[order]
    # IDF pruning: a hash occurring across many DIFFERENT tracks is genre
    # furniture (four-on-floor, stock basses), not evidence of identity.
    # Without it a 1h mix against a real corpus matched 1,771 tracks.
    pair = (H.astype(np.int64) << 20) ^ X.astype(np.int64)
    df_h = (np.unique(pair) >> 20).astype(np.uint32)
    dfs_h, dfs_c = np.unique(df_h, return_counts=True)
    max_df = max(MAX_DF_MIN, int(MAX_DF_FRAC * len(tids)))
    banned = dfs_h[dfs_c > max_df]
    if len(banned):
        keep = ~np.isin(H, banned)
        H, T, X = H[keep], T[keep], X[keep]
    _CACHE[key] = (tids, H, T, X)
    return _CACHE[key]


def _resample(y: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return y
    idx = np.arange(0, len(y) - 1, factor)
    lo = idx.astype(int)
    frac = (idx - lo).astype(np.float32)
    return (y[lo] * (1 - frac) + y[lo + 1] * frac).astype(np.float32)


def match_mix(conn, y: np.ndarray, rates=RATE_SWEEP) -> list[dict]:
    """Find indexed tracks inside a (long) mix buffer.

    Voting is vectorised against the packed index: query hashes are located
    with searchsorted; every (track, offset-bin) pair accumulates votes.
    Defences as before: rate sweep for tempo adjustment, and dominance of
    the merged peak bin over the comb background (periodic music echoes its
    true offset at beat-period aliases; spurious matches spread flat).
    """
    tids, H, T, Tidx = _load_index(conn)
    if H is None:
        return []
    frame_s = HOP / SR
    best: dict[str, dict] = {}
    for rate in rates:
        yr = _resample(y, rate)
        qh = hashes_long(yr)
        del yr
        if not qh:
            continue
        qH = np.array([h for h, _ in qh], dtype=np.uint32)
        qT = np.array([t for _, t in qh], dtype=np.int64)
        lo = np.searchsorted(H, qH, side="left")
        hi = np.searchsorted(H, qH, side="right")
        n = hi - lo
        has = (n > 0) & (n <= DF_MAX)     # rare hashes only: see DF_MAX
        if not has.any():
            continue
        # expand each query hash to all its reference occurrences
        reps = n[has]
        q_frames = np.repeat(qT[has], reps)
        q_w = np.repeat(1.0 / np.log2(1.0 + reps.astype(np.float64)), reps)   # rarity weight per posting
        starts = lo[has]
        flat = np.concatenate([np.arange(a, b) for a, b in
                               zip(starts, starts + reps)]) if len(reps) else np.array([], int)
        r_tidx = Tidx[flat]
        r_frames = T[flat]
        off = ((q_frames - r_frames) * frame_s / OFFSET_BIN_S).astype(np.int64)
        shift = off.min() - 1
        off -= shift
        key = r_tidx.astype(np.int64) * (off.max() + 2) + off
        uniq, inv, votes = np.unique(key, return_inverse=True, return_counts=True)
        wvotes = np.bincount(inv, weights=q_w, minlength=len(uniq))
        u_t = (uniq // (off.max() + 2)).astype(int)
        u_o = (uniq % (off.max() + 2)).astype(int)
        for ti in np.unique(u_t):
            m = u_t == ti
            offs = u_o[m]
            vs = votes[m]; ws = wvotes[m]
            by_off = dict(zip(offs.tolist(), vs.tolist()))
            by_w = dict(zip(offs.tolist(), ws.tolist()))
            merged = {o: by_off.get(o - 1, 0) + v + by_off.get(o + 1, 0)
                      for o, v in by_off.items()}
            peak_off = max(merged, key=merged.get)
            peak_v = int(merged[peak_off])
            peak_w = float(by_w.get(peak_off - 1, 0) + by_w.get(peak_off, 0) + by_w.get(peak_off + 1, 0))
            if peak_w < MIN_WVOTES:
                continue
            others = [v for o, v in by_off.items() if abs(o - peak_off) > 1]
            background3 = 3 * (sorted(others)[len(others) // 2] if others else 0)
            if peak_v < MIN_VOTES or peak_v < DOMINANCE * max(background3, 1):
                continue
            # alignment verification: genuine presence puts matched pairs
            # on a line q = ref + offset over a CONTIGUOUS span of the
            # reference; chance collisions on dense real-music spectra
            # scatter. Vote floors alone hallucinated wholesale live.
            pm = r_tidx == ti
            qf = q_frames[pm].astype(np.float64) * frame_s
            rf = r_frames[pm].astype(np.float64) * frame_s
            resid = qf - (rf + (peak_off + shift) * OFFSET_BIN_S)
            inl = np.abs(resid) < VERIFY_TOL_S
            if int(inl.sum()) < VERIFY_MIN_INLIERS:
                continue
            r_in = np.sort(rf[inl])
            best_span, start = 0.0, r_in[0]
            for j in range(1, len(r_in)):
                if r_in[j] - r_in[j - 1] > 8.0:
                    best_span = max(best_span, r_in[j - 1] - start)
                    start = r_in[j]
            best_span = max(best_span, r_in[-1] - start)
            if best_span < VERIFY_MIN_SPAN_S:
                continue
            tid = tids[ti]
            if tid not in best or peak_v > best[tid]["votes"]:
                best[tid] = {"track_id": tid, "votes": peak_v, "wvotes": round(peak_w, 2),
                             "mix_offset_s": max(0.0, (peak_off + shift) * OFFSET_BIN_S * rate),
                             "rate": rate}
    # non-maximum suppression over mix time: audio can only be one track
    # at once, so overlapping weaker claims (near-clones, shared samples,
    # param-neighbours) yield to the strongest owner of each window.
    kept = []
    for h in sorted(best.values(), key=lambda x: -x["votes"]):
        if all(abs(h["mix_offset_s"] - k["mix_offset_s"]) >= NMS_MIN_GAP_S
               for k in kept):
            kept.append(h)
    return sorted(kept, key=lambda h: h["mix_offset_s"])


def hash_id(y: np.ndarray) -> str:
    """Stable content id for dedupe of mix audio."""
    return hashlib.sha1(y[::100].tobytes()).hexdigest()[:12]
