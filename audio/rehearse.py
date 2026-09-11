"""Does recognition survive a room? Measured on real records, not synthetic ones.

Every synthetic test I wrote for this produced a number that looked like an algorithm
failure and turned out to be an artefact of the signal: a loop where every beat-offset
ties, a drone that dominates every band, a clip renormalised so the log's epsilon moved.
Real records have none of those pathologies, and we have eight thousand of them.

    python -m audio.rehearse --n 20 --out out/rehearse.json

Takes n records, fingerprints each in full, cuts fifteen-second clips from random points,
degrades them the way a phone in a room degrades audio, and matches each clip back against
the set. Reports the share recognised per condition.
"""
import argparse, json, os, random, sys, tempfile, time
import numpy as np

from . import fingerprints as FP
from .beatport import get_token, fetch_genre_top100, download_preview


def degrade(y, kind, rng):
    if kind == "clean":
        return y
    if kind == "quiet":
        return y * 0.15
    if kind == "very quiet":
        return y * 0.03
    if kind == "noise":
        return y + 0.02 * rng.standard_normal(len(y)).astype(np.float32)
    if kind == "loud noise":
        return y + 0.06 * rng.standard_normal(len(y)).astype(np.float32)
    if kind == "dull":                      # a small speaker, and a phone microphone
        k = np.hanning(11); k /= k.sum()
        return np.convolve(y, k, mode="same").astype(np.float32)
    if kind == "room":                      # dull, quiet, noisy and slightly reverberant
        k = np.hanning(11); k /= k.sum()
        z = np.convolve(y, k, mode="same")
        ir = np.zeros(1200, dtype=np.float32); ir[0] = 1.0
        for d in (330, 570, 910):
            ir[d] = 0.3 * rng.random()
        z = np.convolve(z, ir, mode="same")
        return (z * 0.2 + 0.02 * rng.standard_normal(len(z))).astype(np.float32)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--clips", type=int, default=3, help="clips cut from each record")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--out", default="out/rehearse.json")
    a = ap.parse_args()
    rng = np.random.default_rng(7); pick = random.Random(7)

    token = get_token()
    tracks = []
    for gid in (5, 6, 11, 12, 15):                     # a spread of scenes
        try:
            tracks += fetch_genre_top100(token, gid)[:20]
        except Exception as e:
            print(f"  genre {gid}: {type(e).__name__}", flush=True)
    pick.shuffle(tracks)

    ref, names, kept = {}, {}, 0
    for t in tracks:
        if kept >= a.n: break
        url = t.get("sample_url") or (t.get("preview") or {}).get("mp3", {}).get("url")
        if not url: continue
        try:
            p = download_preview(url)
            y = FP.load_audio(p, max_seconds=90)
            if len(y) < FP.SR * 30:
                os.unlink(p); continue
            for h, fr in FP.hashes(y):
                ref.setdefault(h, []).append((kept, fr))
            names[kept] = (t.get("name"), p, len(y))
            kept += 1
            print(f"  {kept}/{a.n} {str(t.get('name'))[:40]}", flush=True)
        except Exception as e:
            print(f"  skip: {type(e).__name__}", flush=True)
    print(f"\nreference: {kept} records, {len(ref):,} distinct hashes", flush=True)

    KINDS = ["clean", "quiet", "very quiet", "dull", "noise", "loud noise", "room"]
    score = {k: [0, 0] for k in KINDS}
    detail = []
    for ti, (nm, path, n) in names.items():
        y = FP.load_audio(path, max_seconds=90)
        for _ in range(a.clips):
            start = pick.randint(0, max(0, len(y) - int(FP.SR * a.seconds) - 1))
            clip = y[start:start + int(FP.SR * a.seconds)]
            for kind in KINDS:
                q = FP.hashes(degrade(clip, kind, rng).astype(np.float32))
                votes = {}
                for h, qf in q:
                    for (rt, rf) in ref.get(h, []):
                        votes[(rt, rf - qf)] = votes.get((rt, rf - qf), 0) + 1
                if votes:
                    (bt, _), v = max(votes.items(), key=lambda kv: kv[1])
                    rest = sorted((x for k2, x in votes.items() if k2[0] != bt), reverse=True)
                    nxt = rest[0] if rest else 0
                    ok = bt == ti and v >= 8 and v >= 1.8 * max(nxt, 1)
                else:
                    ok, v, nxt = False, 0, 0
                score[kind][0] += ok; score[kind][1] += 1
                detail.append({"track": nm, "kind": kind, "ok": bool(ok), "votes": v, "next": nxt})
        try: os.unlink(path)
        except OSError: pass

    print(f"\n{'condition':14}{'recognised':>12}")
    out = {}
    for k in KINDS:
        ok, tot = score[k]
        out[k] = {"ok": ok, "of": tot, "rate": round(ok / max(tot, 1), 3)}
        print(f"  {k:12}{ok:>5} of {tot:<5} {ok/max(tot,1)*100:5.0f}%")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"records": kept, "clip_seconds": a.seconds, "by_condition": out,
               "median_votes": {k: int(np.median([d["votes"] for d in detail if d["kind"] == k] or [0])) for k in KINDS}},
              open(a.out, "w"), indent=1)
    print(f"\nwritten to {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
