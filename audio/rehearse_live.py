"""Does the published index recognise its own records? Measured on the index the site serves, not on the algorithm.

audio.rehearse matches twenty records against each other with their full fingerprints, which tests the matching and
nothing else: it reported 100% clean recognition while the site's index could not recognise half its records at all
(a fixed pruning rule, changed 17 Oct) and covered only the first seven seconds of each (still true: the rest is the
catalogue-wide index on Modal). This takes records from index.json, fetches their previews, cuts clips the way a phone
would, and matches them against index.bin with the site's own rules (api/listen.js in signalgood: at least 8 votes at
one offset, 1.8 times the record's next offset, 1.4 times the next record). Two clip positions: from the start, which
the index covers, and from the middle, which it does not and the Modal index must. Two conditions: clean, and the room
from audio.rehearse.

    python -m audio.rehearse_live --index out/index --n 40 --out out/rehearse-live.json

With --site, a share of the clips also go through the live site's /api/listen as a phone's would, which answers from this
index and, when it is not confident, from the catalogue-wide index on Modal; the first call wakes Modal, and the time it
takes is the time a first listen after idle takes. That measures the whole path rather than its pieces.
"""
import argparse, json, os, random, struct, sys, time
import numpy as np

from . import fingerprints as FP
from .beatport import get_token, _get, download_preview
from .rehearse import degrade

MIN_VOTES, DOMINANCE, NEXT_RECORD = 8, 1.8, 1.4


def load_index(path):
    b = open(path + ".bin", "rb").read()
    magic, n, nt = struct.unpack("<4sII", b[:12])
    if magic != b"SFP2": raise SystemExit(f"index format {magic!r}: this reads SFP2")
    H = np.frombuffer(b, dtype="<u4", count=n, offset=12)
    P = np.frombuffer(b, dtype="<u4", count=n, offset=12 + 4 * n)
    T = json.load(open(path + ".json"))["tracks"]
    return H, P, T


def match(H, P, query):
    """api/listen.js's match, in numpy: votes per (record, offset) over every posting of each query hash."""
    if not query: return None
    qh = np.array([h for h, _ in query], dtype=np.uint32); qf = np.array([f for _, f in query], dtype=np.int64)
    lo = np.searchsorted(H, qh, side="left"); hi = np.searchsorted(H, qh, side="right")
    hit = hi > lo
    if not hit.any(): return {"ok": False, "votes": 0, "next": 0, "hit": 0}
    reps = (hi - lo)[hit]
    pos = np.repeat(lo[hit], reps) + (np.arange(reps.sum()) - np.repeat(np.cumsum(reps) - reps, reps))
    rec = (P[pos] >> 13).astype(np.int64); off = (P[pos] & 0x1FFF).astype(np.int64) - np.repeat(qf[hit], reps)
    key = rec * 20000 + (off + 10000)
    u, c = np.unique(key, return_counts=True)
    best = np.argmax(c); top_rec, top = int(u[best] // 20000), int(c[best])
    same = (u // 20000) == top_rec
    runner = int(np.sort(c[same])[-2]) if same.sum() > 1 else 0
    others = c[~same]; nxt = int(others.max()) if len(others) else 0
    ok = top >= MIN_VOTES and top >= DOMINANCE * max(runner, 1) and top >= NEXT_RECORD * max(nxt, 1)
    return {"ok": bool(ok), "rec": top_rec, "votes": top, "runner": runner, "next": nxt, "hit": int(hit.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="out/index"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=15); ap.add_argument("--out", default="out/rehearse-live.json")
    ap.add_argument("--site", default="", help="also send clips through this site's /api/listen (e.g. https://earlysignal.live)")
    ap.add_argument("--site-n", type=int, default=12, help="records whose clips go through the site")
    a = ap.parse_args()
    H, P, T = load_index(a.index)
    per = np.bincount(P >> 13, minlength=len(T))
    print(f"index: {len(T):,} records, {len(H):,} postings; records with fewer than {MIN_VOTES} fingerprints "
          f"{int((per < MIN_VOTES).sum()):,}, with 50 or more {int((per >= 50).sum()):,}", flush=True)
    rng, pick = np.random.default_rng(11), random.Random(int(time.time()) // 86400)   # a different sample each day
    order = list(range(len(T))); pick.shuffle(order)
    token = get_token()
    POS, KINDS = ("start", "middle"), ("clean", "room")
    score = {(p, k): [0, 0] for p in POS for k in KINDS}
    votes = {(p, k): [] for p in POS for k in KINDS}
    tried = kept = 0
    site = {(p, k): [] for p in POS for k in KINDS}; site_first = None
    def through_site(tid, q):
        """One clip through the live site: whether it named the record, by which index, and how long it took."""
        import urllib.request, urllib.error
        body = json.dumps({"hashes": [[int(h), int(f)] for h, f in q][:20000]}).encode()
        t0_ = time.time()
        try:
            r = urllib.request.urlopen(urllib.request.Request(a.site.rstrip("/") + "/api/listen", data=body,
                                       headers={"Content-Type": "application/json", "User-Agent": "sonic-rehearse"}), timeout=40)
            d = json.loads(r.read()); took = round(time.time() - t0_, 1)
            got = (d.get("track") or {}).get("track_id") if d.get("found") else None
            dz = d.get("dense") if isinstance(d.get("dense"), dict) else {}
            return {"right": got == tid, "found": bool(d.get("found")), "via": d.get("via") or ("site" if d.get("found") else None),
                    "dense": dz.get("found"), "dense_error": dz.get("error") or dz.get("why"), "seconds": took}
        except Exception as e:
            return {"right": False, "found": False, "via": None, "dense": None, "dense_error": f"{type(e).__name__}: {str(e)[:60]}", "seconds": round(time.time() - t0_, 1)}
    for ti in order:
        if kept >= a.n or tried >= 3 * a.n: break
        t = T[ti]; tried += 1
        if not str(t.get("track_id", "")).startswith("bp:"): continue
        try:
            d = _get(f"/catalog/tracks/{t['track_id'][3:]}/", token)
            url = d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url")
            if not url: continue
            path = download_preview(url)
            try:
                y = FP.load_audio(path, max_seconds=120)
            finally:
                try: os.unlink(path)
                except OSError: pass
            L = int(FP.SR * a.seconds)
            if len(y) < 3 * L: continue
            clips = {"start": y[:L], "middle": y[(s := pick.randint(2 * L, len(y) - L)):s + L]}
            for p_, clip in clips.items():
                for k in KINDS:
                    q = FP.hashes(degrade(clip, k, rng).astype(np.float32))
                    m = match(H, P, q)
                    ok = bool(m and m["ok"] and m["rec"] == ti)
                    score[(p_, k)][0] += ok; score[(p_, k)][1] += 1; votes[(p_, k)].append(m["votes"] if m else 0)
                    if a.site and kept < a.site_n:
                        r_ = through_site(t["track_id"], q); site[(p_, k)].append(r_)
                        if site_first is None: site_first = r_
            kept += 1
            print(f"  {kept}/{a.n} {str(t.get('name'))[:36]:36s} start clean {score[('start','clean')][0]}/{score[('start','clean')][1]}", flush=True)
        except Exception as e:
            print(f"  skip {t.get('track_id')}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    out = {"records": kept, "clip_seconds": a.seconds, "index_records": len(T), "postings": int(len(H)),
           "under_min_votes": int((per < MIN_VOTES).sum()), "with_50_or_more": int((per >= 50).sum()),
           "by_position_and_condition": {f"{p} {k}": {"ok": s[0], "of": s[1], "rate": round(s[0] / max(s[1], 1), 3),
                                                      "median_votes": int(np.median(votes[(p, k)] or [0]))}
                                         for (p, k), s in score.items()}}
    if a.site:
        out["through_site"] = {f"{p} {k}": {"right": sum(1 for r_ in rs if r_["right"]), "of": len(rs),
                                            "by site index": sum(1 for r_ in rs if r_["right"] and r_["via"] == "site"),
                                            "by modal": sum(1 for r_ in rs if r_["right"] and r_["via"] == "classics"),
                                            "modal no answer": sum(1 for r_ in rs if r_["dense"] is None and not r_["found"]),
                                            "median_seconds": round(float(np.median([r_["seconds"] for r_ in rs])), 1) if rs else None}
                               for (p, k), rs in site.items()}
        out["site_first_call"] = site_first
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    line = ", ".join(f"{p} {k} {s[0]}/{s[1]}" for (p, k), s in score.items())
    print(f"::notice title=live rehearsal::{kept} records from the published index, {a.seconds:.0f}s clips: {line}; "
          f"{out['under_min_votes']:,} of {len(T):,} records under {MIN_VOTES} fingerprints", flush=True)
    if a.site:
        ts = out["through_site"]
        print("::notice title=through the site::" + "; ".join(f"{k}: {v['right']}/{v['of']} named ({v['by site index']} by the site's index, {v['by modal']} by Modal"
              f"{', ' + str(v['modal no answer']) + ' with no answer from Modal' if v['modal no answer'] else ''}), median {v['median_seconds']}s" for k, v in ts.items())
              + (f"; first call {site_first['seconds']}s" + (f", Modal: {site_first['dense_error']}" if site_first.get('dense_error') else "") if site_first else ""), flush=True)


if __name__ == "__main__":
    sys.exit(main())
