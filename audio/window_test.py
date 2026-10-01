"""Which part of a record can Sonic recognise? The index keeps the first 1,800 fingerprints of each record ("about the
first minute" of its two-minute preview), so a clip from anywhere else in the preview has nothing to match. Here three
versions of an index are built from the same classics: the first 1,800 (as now), 1,800 spread evenly across the whole
preview (same size), and every fingerprint. Then 25-second clips from random points anywhere in the preview, played
clean, through a room and through noise, are matched with the rehearsal's rule. Writes data/window-test.json.
  python -m audio.window_test --n 120"""
import argparse, glob, json, os, random, tempfile, urllib.request
import numpy as np
from . import fingerprints as FP
from .rehearse import degrade

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=120); ap.add_argument("--clips", type=int, default=3); ap.add_argument("--seconds", type=float, default=25.0)
    a = ap.parse_args(); rng = np.random.default_rng(7); pick = random.Random(7)
    recs, seen = [], set()
    for f in sorted(glob.glob("out/fp-canon-2*.jsonl")):
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("found") and d.get("preview") and d.get("track_id") not in seen: seen.add(d["track_id"]); recs.append(d)
    pick.shuffle(recs); recs = recs[: a.n]
    Y, full = [], []
    for d in recs:
        fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
        try:
            urllib.request.urlretrieve(d["preview"], p); y = FP.load_audio(p, max_seconds=None)
            hs = sorted(FP.hashes(y), key=lambda x: x[1]); Y.append(y); full.append(hs)
        except Exception: Y.append(None); full.append([])
        finally: os.remove(p)
    keep = [i for i, hs in enumerate(full) if len(hs) >= 200]
    def variant(kind):
        ref = {}
        for i in keep:
            hs = full[i]
            if kind == "first": sub = hs[:1800]
            elif kind == "spread": sub = [hs[j] for j in np.unique(np.linspace(0, len(hs) - 1, min(1800, len(hs))).astype(int))]
            else: sub = hs
            for h, fr in sub: ref.setdefault(h, []).append((i, fr))
        return ref
    REF = {k: variant(k) for k in ("first", "spread", "all")}
    KINDS = ("clean", "room", "noise"); score = {v: {k: {"early": [0, 0], "late": [0, 0]} for k in KINDS} for v in REF}
    for i in keep:
        y = Y[i]; L = int(FP.SR * a.seconds)
        for _ in range(a.clips):
            st = pick.randint(0, max(0, len(y) - L - 1)); clip = y[st:st + L]; where = "early" if st / FP.SR < 60 else "late"
            for kind in KINDS:
                q = FP.hashes(degrade(clip, kind, rng).astype(np.float32))
                for v, ref in REF.items():
                    votes = {}
                    for h, qf in q:
                        for (rt, rf) in ref.get(h, []): votes[(rt, rf - qf)] = votes.get((rt, rf - qf), 0) + 1
                    ok = False
                    if votes:
                        (bt, _), vv = max(votes.items(), key=lambda kv: kv[1]); rest = sorted((x for k2, x in votes.items() if k2[0] != bt), reverse=True); nx = rest[0] if rest else 0
                        ok = bt == i and vv >= 8 and vv >= 1.8 * max(nx, 1)
                    s = score[v][kind][where]; s[0] += ok; s[1] += 1
    out = {"note": __doc__.split("  python")[0].strip(), "records": len(keep), "preview_seconds_median": round(float(np.median([len(Y[i]) / FP.SR for i in keep])), 1),
           "fingerprints_per_record_median": int(np.median([len(full[i]) for i in keep])), "score": score}
    json.dump(out, open("data/window-test.json", "w"), indent=1)
    summ = {v: {k: f"early {score[v][k]['early'][0]}/{score[v][k]['early'][1]}, late {score[v][k]['late'][0]}/{score[v][k]['late'][1]}" for k in KINDS} for v in REF}
    print("::notice title=window test::" + json.dumps({"records": len(keep), "per_record": out["fingerprints_per_record_median"], "score": summ}))

if __name__ == "__main__": main()
