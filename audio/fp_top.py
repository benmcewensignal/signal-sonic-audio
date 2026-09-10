"""Fingerprint the records people would actually play.

The store holds 8,241 records, but they are whatever the set-matching pass happened to
index. Of the eighty-three highest-charting records in the corpus, fourteen are in it.
A recognition demo that misses a number one is not a demo.

This picks the records most likely to be played at a phone — best chart position first,
then records a DJ has played in a scanned set, then anything by an artist with bookings —
and fingerprints whatever the store is missing. It publishes only the additions, as
base64 blobs in jsonl, so nothing has to move the gigabyte store around.

  python -m audio.fp_top --meta sonic.db --out out/fp-extra.jsonl --limit 400
"""
import argparse, base64, collections, json, os, sqlite3, tempfile, time, urllib.request
import numpy as np
from . import fingerprints as FP
from .beatport import get_token, _get

STORE_INDEX = "https://raw.githubusercontent.com/benmcewensignal/signal-sonic-audio/main/out/index.json"


def indexed_already(out_dir):
    have = set()
    try:
        req = urllib.request.Request(STORE_INDEX, headers={"User-Agent": "signal-sonic-audio/fp-top"})
        with urllib.request.urlopen(req, timeout=60) as r:
            have |= {t["track_id"] for t in json.loads(r.read())["tracks"]}
    except Exception as e:
        print(f"  could not read the current index ({type(e).__name__}); treating everything as missing", flush=True)
    for f in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if f.startswith("fp-extra") and f.endswith(".jsonl"):
            for line in open(os.path.join(out_dir, f)):
                try: have.add(json.loads(line)["track_id"])
                except Exception: pass
    return have


BEATPORT_GENRES = {"deep-house": 12, "tech-house": 11, "house": 5, "techno-peak-time": 6,
 "techno-raw-deep-hypnotic": 92, "melodic-house-techno": 90, "afro-house": 89, "amapiano": 98,
 "drum-and-bass": 1, "uk-garage-speed-garage": 86, "140-deep-dubstep-grime": 95,
 "breaks-breakbeat-uk-bass": 9, "bass-house": 91, "trance-main-floor": 7, "psy-trance": 13,
 "progressive-house": 15, "indie-dance": 37, "organic-house": 93, "hard-techno": 2, "uk-funky-gqom": 85}


def beatport_top(token, limit_per_genre=100):
    """Beatport's current Top 100 per genre: the records most likely to be played at a phone
    today. Our own chart slices are only two weeks deep, so this is a far better base."""
    out = []
    for scene, gid in BEATPORT_GENRES.items():
        try:
            d = _get(f"/catalog/genres/{gid}/top/{limit_per_genre}/", token)
            rows = d if isinstance(d, list) else d.get("results", [])
            for i, t in enumerate(rows):
                tid = "bp:" + str(t.get("id"))
                out.append((tid, scene, i + 1))
        except Exception as e:
            print(f"  top 100 for {scene}: {type(e).__name__}", flush=True)
    return out


def wanted(meta_db, have, limit):
    c = sqlite3.connect(meta_db); c.row_factory = sqlite3.Row
    best = {t: rk for t, rk in c.execute(
        "select track_id, min(chart_rank) from track_scenes where chart_rank is not null group by track_id")}
    played = collections.Counter()
    for (t,) in c.execute("select track_id from mix_plays"): played[t] += 1
    named = {r["track_id"]: (r["name"], json.loads(r["artists"]) if r["artists"] else [])
             for r in c.execute("select track_id, artists, name from track_meta where name is not null")}
    scored = []
    for t in named:
        if t in have or not t.startswith("bp:"): continue
        if t in best: scored.append((0, best[t], t))            # charted, best position first
        elif played[t]: scored.append((1, -played[t], t))       # then played in a set, most-played first
    scored.sort()
    return [t for _, _, t in scored[:limit]], len(scored)


def wanted_with_top(meta_db, have, limit, token):
    """Beatport's live Top 100 first, then our own charted and set-played records."""
    top = beatport_top(token)
    picks, seen = [], set(have)
    for tid, scene, rank in sorted(top, key=lambda x: x[2]):
        if tid in seen: continue
        seen.add(tid); picks.append(tid)
    print(f"Beatport top 100 across {len(BEATPORT_GENRES)} genres: {len(top)} records, {len(picks)} not yet indexed", flush=True)
    rest, total = wanted(meta_db, seen, max(0, limit - len(picks)))
    return (picks + rest)[:limit], len(picks) + total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="sonic.db"); ap.add_argument("--out", default="out/fp-extra.jsonl")
    ap.add_argument("--limit", type=int, default=400); ap.add_argument("--budget-minutes", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    have = indexed_already(os.path.dirname(a.out))
    token0 = get_token()
    todo, total = wanted_with_top(a.meta, have, a.limit, token0)
    print(f"{len(have)} already indexed | {total} worth adding | {len(todo)} this run", flush=True)
    if not todo: return
    token = token0
    t0 = time.time(); done = err = 0
    mode = "a" if os.path.exists(a.out) else "w"
    with open(a.out, mode) as out:
        for tid in todo:
            if (time.time() - t0) / 60 > a.budget_minutes:
                print("budget reached", flush=True); break
            p = None
            try:
                d = _get(f"/catalog/tracks/{tid.split(':')[-1]}/", token)
                url = d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url")
                if not url: raise ValueError("no preview url")
                fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
                req = urllib.request.Request(url, headers={"User-Agent": "signal-sonic-audio/fp-top"})
                with urllib.request.urlopen(req, timeout=30) as r, open(p, "wb") as f: f.write(r.read())
                y = FP.load_audio(p, max_seconds=90)
                hs = FP.hashes(y)
                if len(hs) < 50: raise ValueError("too few fingerprints")
                H = np.array([h for h, _ in hs], dtype="<u4")
                Fr = np.minimum(np.array([f for _, f in hs]), 0xFFFF).astype("<u2")
                out.write(json.dumps({"track_id": tid, "n": int(len(H)),
                                      "hashes": base64.b64encode(H.tobytes()).decode(),
                                      "frames": base64.b64encode(Fr.tobytes()).decode()}) + "\n")
                done += 1
                if done % 25 == 0: out.flush(); print(f"  {done}/{len(todo)}, {err} failed", flush=True)
            except Exception as e:
                err += 1
                if err <= 3: print(f"  {tid}: {type(e).__name__}: {str(e)[:70]}", flush=True)
            finally:
                if p:
                    try: os.unlink(p)
                    except OSError: pass
    print(json.dumps({"added": done, "failed": err, "still_wanted": max(0, total - len(have) - done)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
