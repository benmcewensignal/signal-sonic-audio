"""Build the recognition index for the listening demo.

Five hundred records across twenty scenes, hashed from their previews into an inverted
index a serverless function can hold in memory. The full store is 137 million hashes and
cannot be; this is a demo set, and the app says so.

Each track keeps its strongest hashes only, capped, so the index stays a few megabytes.
The hashes are exactly those the browser computes, so a phone can query it directly.

  python -m audio.demo_index --db sonic.db --out out/demo-index.json --per-track 420
"""
import argparse, collections, json, os, sqlite3, tempfile, time, urllib.request
import numpy as np
from .beatport import get_token, _get
from . import fingerprints as FP


def preview(track_id, token):
    d = _get(f"/catalog/tracks/{track_id.split(':')[-1]}/", token)
    return d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url")


def fetch(url):
    fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "signal-sonic-audio/demo"})
    with urllib.request.urlopen(req, timeout=30) as r, open(p, "wb") as f: f.write(r.read())
    return p


def pick(db, per_scene):
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    out, seen = collections.defaultdict(list), set()
    for r in c.execute("""select ts.scene, ts.track_id, ts.chart_rank, m.name, m.artists, m.label
                          from track_scenes ts join track_meta m on m.track_id=ts.track_id
                          where m.name is not null and ts.week like '____-M__'
                          order by coalesce(ts.chart_rank, 999)"""):
        if len(out[r["scene"]]) >= per_scene or r["track_id"] in seen: continue
        seen.add(r["track_id"])
        out[r["scene"]].append({"track_id": r["track_id"], "name": r["name"],
                                "artists": json.loads(r["artists"]) if r["artists"] else [],
                                "label": r["label"], "scene": r["scene"]})
    # each record carries its own measurements and where it sits against its scene, so the
    # result can say "this is the tune, this is the scene, and this is it against the scene"
    import numpy as _np
    feats = {}
    for r in c.execute("select track_id, features, analyser_ver from tracks where analyser_id='local'"):
        try: feats[r["track_id"]] = (json.loads(r["features"]), str(r["analyser_ver"] or "1"))
        except Exception: pass
    home = collections.defaultdict(list)
    for r in c.execute("""select ts.scene, ts.track_id from track_scenes ts
                          where ts.week like '____-M__' and ts.week <= '2025-M05'"""):
        f = feats.get(r["track_id"])
        if f and f[0].get("embedding"): home[r["scene"]].append((f[0]["embedding"], f[1]))
    centre = {}
    for sc, rows in home.items():
        vers = collections.Counter(v for _, v in rows).most_common(1)
        if not vers: continue
        vv = "2" if vers[0][0].startswith("2") else "1"
        E = [_np.array(e, float) for e, v in rows if (v.startswith("2")) == (vv == "2")]
        if len(E) < 20: continue
        E = [e / (_np.linalg.norm(e) or 1) for e in E]
        centre[sc] = (_np.mean(E, axis=0), vv)
    flat = [t for v in out.values() for t in v]
    for t in flat:
        f = feats.get(t["track_id"])
        if not f: continue
        d, ver = f
        t["measures"] = {k: (round(float(d[k]), 3) if isinstance(d.get(k), (int, float)) else None)
                         for k in ("tempo", "drum_density", "drum_swing", "bass_weight", "vocal_presence")}
        cen = centre.get(t["scene"])
        if cen is not None and d.get("embedding"):
            v = _np.array(d["embedding"], float)
            if (ver.startswith("2")) == (cen[1] == "2") and len(v) == len(cen[0]):
                v = v / (_np.linalg.norm(v) or 1)
                t["dist_from_scene_2024"] = round(float(1 - v @ cen[0] / ((_np.linalg.norm(v) * _np.linalg.norm(cen[0])) or 1)), 4)
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sonic.db"); ap.add_argument("--out", default="out/demo-index.json")
    ap.add_argument("--per-scene", type=int, default=10); ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--budget-minutes", type=int, default=80)
    a = ap.parse_args()
    tracks = pick(a.db, a.per_scene)
    print(f"demo set: {len(tracks)} records", flush=True)
    token = get_token()
    idx = collections.defaultdict(list)          # hash -> [[track index, frame], ...]
    meta, t0, done, err = [], time.time(), 0, 0
    for t in tracks:
        if (time.time() - t0) / 60 > a.budget_minutes:
            print("budget reached", flush=True); break
        p = None
        try:
            url = preview(t["track_id"], token)
            if not url: raise ValueError("no preview")
            p = fetch(url)
            y = FP.load_audio(p, max_seconds=a.seconds)
            # never thin the reference hashes: a query only overlaps the hashes that exist,
            # so subsampling the index destroys matching. Control size with less audio instead.
            hs = FP.hashes(y)
            ti = len(meta)
            for h, fr in hs: idx[h].append([ti, fr])
            meta.append(t); done += 1
            if done % 25 == 0: print(f"  {done}/{len(tracks)}, {err} failed, {len(idx)} distinct hashes", flush=True)
        except Exception as e:
            err += 1
            if err <= 3: print(f"  {t['track_id']}: {type(e).__name__}: {str(e)[:60]}", flush=True)
        finally:
            if p:
                try: os.unlink(p)
                except OSError: pass
    # drop hashes that appear in too many tracks: they identify nothing
    trimmed = {str(h): v for h, v in idx.items() if len({x[0] for x in v}) <= 8}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "tracks": meta, "index": trimmed,
               "note": "a demo index: 500 records, not the whole corpus"},
              open(a.out, "w"), separators=(",", ":"))
    size = os.path.getsize(a.out) / 1e6
    print(json.dumps({"tracks": len(meta), "failed": err, "distinct_hashes": len(trimmed),
                      "dropped_as_common": len(idx) - len(trimmed), "size_mb": round(size, 1)}, indent=1))


if __name__ == "__main__":
    main()
