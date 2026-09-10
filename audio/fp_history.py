"""Build the recognition corpus backwards through Beatport's own history.

The canon list is 158 records chosen by hand. Beatport holds fifteen years of releases
and can be asked for them a month at a time, so the corpus of "the records that mattered"
can be built rather than curated — as far back as the API will go, for every genre we
measure.

Two things this is not. It is not a measurement corpus: these records never enter the
sound layer, because a series built from different sampling would not be comparable.
And it is not "the greatest" — it is what Beatport ranked highest in each month, which
is a store's sales, not a canon. The job reports which ordering the API actually honoured
so we know which of those two we got.

  python -m audio.fp_history --meta sonic.db --out out/fp-history.jsonl --from 2016-01 --per-month 8
"""
import argparse, base64, json, os, tempfile, time, urllib.request
import numpy as np
from . import fingerprints as FP
from .beatport import get_token, _get
from .fp_top import BEATPORT_GENRES, indexed_already

# tried in order; the first that returns results is used and reported
ORDER_VARIANTS = ["-sales", "-popularity", "-hype", "-release_date", "-publish_date"]


def month_range(month):
    y, m = int(month[:4]), int(month[5:7])
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}-01", f"{ny:04d}-{nm:02d}-01"


def months(start, end):
    y, m = int(start[:4]), int(start[5:7])
    out = []
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12: m, y = 1, y + 1
    return out


def top_for(token, gid, month, n, order_state):
    s, e = month_range(month)
    tried = order_state.get("order") and [order_state["order"]] or ORDER_VARIANTS
    for order in tried:
        try:
            d = _get("/catalog/tracks/", token, {"genre_id": gid, "per_page": n,
                                                 "publish_date_start": s, "publish_date_end": e,
                                                 "order_by": order})
            rows = d.get("results", [])
            if rows:
                order_state["order"] = order
                return rows
        except Exception:
            continue
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="sonic.db"); ap.add_argument("--out", default="out/fp-history.jsonl")
    ap.add_argument("--from", dest="start", default="2016-01")
    ap.add_argument("--to", dest="end", default=time.strftime("%Y-%m"))
    ap.add_argument("--per-month", type=int, default=8)
    ap.add_argument("--budget-minutes", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    have = indexed_already(os.path.dirname(a.out))
    if os.path.exists(a.out):
        for line in open(a.out):
            try: have.add(json.loads(line)["track_id"])
            except Exception: pass
    token = get_token()
    ms = months(a.start, a.end)
    print(f"{len(BEATPORT_GENRES)} genres x {len(ms)} months, {a.per_month} a month | {len(have)} already held", flush=True)
    order_state = {}
    t0 = time.time(); done = err = skipped = 0
    mode = "a" if os.path.exists(a.out) else "w"
    with open(a.out, mode) as out:
        # newest months first: the most recognisable records are the recent ones
        for month in reversed(ms):
            if (time.time() - t0) / 60 > a.budget_minutes: break
            for scene, gid in BEATPORT_GENRES.items():
                if (time.time() - t0) / 60 > a.budget_minutes: break
                try: rows = top_for(token, gid, month, a.per_month, order_state)
                except Exception as e:
                    err += 1; continue
                for t in rows:
                    tid = "bp:" + str(t.get("id"))
                    if tid in have: skipped += 1; continue
                    url = t.get("sample_url") or ((t.get("preview") or {}).get("mp3") or {}).get("url")
                    name = t.get("name") or ""
                    arts = [x.get("name") for x in (t.get("artists") or []) if x.get("name")]
                    if not url or not name: continue
                    p = None
                    try:
                        fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
                        req = urllib.request.Request(url, headers={"User-Agent": "signal-sonic-audio/history"})
                        with urllib.request.urlopen(req, timeout=25) as r, open(p, "wb") as f: f.write(r.read())
                        y = FP.load_audio(p, max_seconds=90)
                        hs = FP.hashes(y)
                        if len(hs) < 50: raise ValueError("too few fingerprints")
                        H = np.array([h for h, _ in hs], dtype="<u4")
                        Fr = np.minimum(np.array([f for _, f in hs]), 0xFFFF).astype("<u2")
                        out.write(json.dumps({"track_id": tid, "name": name, "artists": arts,
                                              "scene": scene, "month": month, "canon": True,
                                              "n": int(len(H)),
                                              "hashes": base64.b64encode(H.tobytes()).decode(),
                                              "frames": base64.b64encode(Fr.tobytes()).decode()}) + "\n")
                        have.add(tid); done += 1
                        if done % 50 == 0:
                            out.flush(); print(f"  {done} added ({month}), {err} errors, {skipped} already held", flush=True)
                    except Exception as e:
                        err += 1
                        if err <= 3: print(f"  {tid}: {type(e).__name__}: {str(e)[:60]}", flush=True)
                    finally:
                        if p:
                            try: os.unlink(p)
                            except OSError: pass
    print(json.dumps({"added": done, "errors": err, "already_held": skipped,
                      "ordering_used": order_state.get("order"),
                      "note": "ordering tells you whether this is Beatport's ranking or merely its release order"},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
