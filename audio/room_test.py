"""Which part of a room defeats recognition? The two room models in use disagree by a factor of five (audio.rehearse's
room, and the calibration test's in signal-sonic-features/worker/recognise_test_all.py: 69 and 13 recognised in 100 at
8,000 fingerprints a record). This takes records from the index, fingerprints their whole previews at 8,000 spread
across each (as the catalogue-wide index on Modal does), cuts 25-second clips from random points, and degrades each clip
one way at a time: band-limited, reverberant, noisy at two levels, dulled, and the two rooms whole. For each it reports
how many clips are recognised and what share of the clip's clean fingerprints survive (same hash, same place within a
frame), which is the number the fingerprinter would have to defend.

    python -m audio.room_test --n 24 --out out/room-test.json
"""
import argparse, json, os, random, sys, time
import numpy as np

from . import fingerprints as FP
from .beatport import get_token, _get, download_preview
from .rehearse import degrade

MIN_VOTES, NEXT_RECORD, PER = 10, 1.4, 8000


def cal_room(y, rng):
    """The calibration test's room: a 0.6 s exponential reverb, a 200 to 6,000 Hz band, noise 15 dB under the signal."""
    from scipy.signal import butter, sosfilt, fftconvolve
    sr = FP.SR; n = int(0.6 * sr); ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.12 * sr)); ir[0] = 1.0
    w = fftconvolve(y, ir)[: len(y)]; w = sosfilt(butter(4, [200, 6000], btype="band", fs=sr, output="sos"), w)
    w = w / (np.max(np.abs(w)) + 1e-9); return (w + rng.normal(0, 10 ** (-15 / 20) * np.sqrt(np.mean(w ** 2)), len(w))).astype(np.float32)


def stage(y, kind, rng):
    from scipy.signal import butter, sosfilt, fftconvolve
    sr = FP.SR
    if kind == "clean": return y
    if kind == "band 200-6000": w = sosfilt(butter(4, [200, 6000], btype="band", fs=sr, output="sos"), y); return (w / (np.max(np.abs(w)) + 1e-9)).astype(np.float32)
    if kind == "reverb 0.6s":
        n = int(0.6 * sr); ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.12 * sr)); ir[0] = 1.0
        w = fftconvolve(y, ir)[: len(y)]; return (w / (np.max(np.abs(w)) + 1e-9)).astype(np.float32)
    if kind.startswith("noise "):
        snr = float(kind.split()[1].rstrip("dB")); return (y + rng.normal(0, 10 ** (-snr / 20) * np.sqrt(np.mean(y ** 2)), len(y))).astype(np.float32)
    if kind == "calibration room": return cal_room(y, rng)
    if kind == "rehearsal dull": return degrade(y, "dull", rng)
    if kind == "rehearsal room": return degrade(y, "room", rng)
    raise ValueError(kind)


KINDS = ["clean", "band 200-6000", "reverb 0.6s", "noise 25dB", "noise 15dB", "rehearsal dull", "rehearsal room", "calibration room"]


def match(H, T, F, query):
    if not query: return None
    qh = np.array([h for h, _ in query], dtype=np.uint32); qf = np.array([f for _, f in query], dtype=np.int64)
    lo = np.searchsorted(H, qh, "left"); hi = np.searchsorted(H, qh, "right"); hit = hi > lo
    if not hit.any(): return {"ok": False, "votes": 0}
    reps = (hi - lo)[hit]; pos = np.repeat(lo[hit], reps) + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))
    rec = T[pos].astype(np.int64); off = F[pos].astype(np.int64) - np.repeat(qf[hit], reps)
    key = rec * 20000 + (off + 10000); u, c = np.unique(key, return_counts=True)
    # a bin and its neighbours, as the Modal matcher counts them
    cm = c.copy(); cm[1:] += np.where(u[1:] - u[:-1] == 1, c[:-1], 0); cm[:-1] += np.where(u[1:] - u[:-1] == 1, c[1:], 0)
    b = int(np.argmax(cm)); top_rec, top = int(u[b] // 20000), int(cm[b])
    others = cm[(u // 20000) != top_rec]; nxt = int(others.max()) if len(others) else 0
    return {"ok": bool(top >= MIN_VOTES and top >= NEXT_RECORD * max(nxt, 1)), "rec": top_rec, "votes": top, "next": nxt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="out/index.json"); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=25); ap.add_argument("--out", default="out/room-test.json")
    a = ap.parse_args()
    T_ = json.load(open(a.index))["tracks"]; pick = random.Random(5); rng = np.random.default_rng(5)
    order = list(range(len(T_))); pick.shuffle(order)
    token = get_token(); audio, Hs, Ts, Fs = [], [], [], []
    for ti in order:
        if len(audio) >= a.n: break
        t = T_[ti]
        if not str(t.get("track_id", "")).startswith("bp:"): continue
        try:
            d = _get(f"/catalog/tracks/{t['track_id'][3:]}/", token)
            url = d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url")
            if not url: continue
            p = download_preview(url)
            try: y = FP.load_audio(p, max_seconds=120)
            finally:
                try: os.unlink(p)
                except OSError: pass
            if len(y) < FP.SR * 60: continue
            hs = FP.hashes(y)
            if len(hs) < 500: continue
            h = np.array([x for x, _ in hs], dtype=np.uint32); fr = np.array([f for _, f in hs], dtype=np.int64)
            if len(h) > PER:
                o = np.argsort(fr, kind="stable"); sel = o[np.unique(np.linspace(0, len(h) - 1, PER).astype(np.int64))]; h, fr = h[sel], fr[sel]
            Hs.append(h); Fs.append(fr); Ts.append(np.full(len(h), len(audio), dtype=np.int64)); audio.append((t.get("name"), y))
            print(f"  {len(audio)}/{a.n} {str(t.get('name'))[:40]} ({len(hs):,} fingerprints, {len(h):,} kept)", flush=True)
        except Exception as e:
            print(f"  skip {t.get('track_id')}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    H = np.concatenate(Hs); T = np.concatenate(Ts); F = np.concatenate(Fs); o = np.argsort(H, kind="stable"); H, T, F = H[o], T[o], F[o]
    print(f"reference: {len(audio)} records, {len(H):,} fingerprints", flush=True)
    score = {k: [0, 0] for k in KINDS}; surv = {k: [] for k in KINDS}; votes = {k: [] for k in KINDS}
    L = int(FP.SR * a.seconds)
    for ti, (nm, y) in enumerate(audio):
        for _ in range(2):
            st = pick.randint(0, len(y) - L - 1); clip = y[st:st + L]
            clean = {(h, f) for h, f in FP.hashes(clip)}
            for k in KINDS:
                q = FP.hashes(stage(clip, k, rng).astype(np.float32))
                m = match(H, T, F, q); ok = bool(m and m["ok"] and m["rec"] == ti)
                score[k][0] += ok; score[k][1] += 1; votes[k].append(m["votes"] if m else 0)
                qs = set(q); surv[k].append(sum(1 for h, f in clean if (h, f) in qs or (h, f - 1) in qs or (h, f + 1) in qs) / max(len(clean), 1))
    out = {"records": len(audio), "clip_seconds": a.seconds, "per_record": PER,
           "by_degradation": {k: {"recognised": s[0], "of": s[1], "rate": round(s[0] / max(s[1], 1), 3),
                                  "fingerprints_surviving": round(float(np.median(surv[k])), 3), "median_votes": int(np.median(votes[k]))} for k, s in score.items()}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print("::notice title=room test::" + "; ".join(f"{k}: {v['recognised']}/{v['of']} recognised, {100*v['fingerprints_surviving']:.0f}% of fingerprints survive, median {v['median_votes']} votes"
                                                  for k, v in out["by_degradation"].items()), flush=True)


if __name__ == "__main__":
    sys.exit(main())
