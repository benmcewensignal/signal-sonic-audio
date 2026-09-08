"""Benchmark Olaf against the NTS tracklist truth, on real audio.

Reads the main pipeline's database (fetched from the signal-sonic repo), indexes the
previews of every record that appears in an NTS tracklist plus a random control set,
downloads each NTS mix, queries it with Olaf, and scores precision and recall against
the published tracklist. The same truth set scores our own matcher, so the two are
directly comparable.

  python -m audio.bench --db sonic.db --out out/olaf-bench.json
"""
import argparse, json, os, random, sqlite3, subprocess, tempfile, time, urllib.request
from . import olaf
from .beatport import _get

from .beatport import get_token as bp_token, _get


def preview(track_id, token):
    d = _get(f"/catalog/tracks/{track_id.split(':')[-1]}/", token)
    return d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url")


def fetch(url, suffix=".mp3"):
    fd, p = tempfile.mkstemp(suffix=suffix); os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "signal-sonic-audio/bench"})
    with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f: f.write(r.read())
    return p


def mix_audio(mix_url):
    d = tempfile.mkdtemp()
    subprocess.run(["yt-dlp", "-q", "-f", "bestaudio/best", "-P", d, "-o", "mix.%(ext)s", mix_url], check=True, timeout=600)
    files = [os.path.join(d, f) for f in os.listdir(d) if f.startswith("mix.")]
    return files[0] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sonic.db"); ap.add_argument("--out", default="out/olaf-bench.json")
    ap.add_argument("--controls", type=int, default=300)
    a = ap.parse_args()
    c = sqlite3.connect(a.db); c.row_factory = sqlite3.Row
    truth = {}
    for r in c.execute("select mix_url, track_id from nts_tracklist where track_id is not null"):
        truth.setdefault(r["mix_url"], set()).add(r["track_id"])
    if not truth: raise SystemExit("no NTS tracklist truth in the database yet")
    ours = {}
    for r in c.execute("select mix_url, track_id from mix_plays"): ours.setdefault(r["mix_url"], set()).add(r["track_id"])
    truth_ids = set().union(*truth.values())
    pool = [r[0] for r in c.execute("select track_id from tracks where analyser_id='local' and track_id like 'bp:%'")]
    controls = set(random.Random(1).sample([t for t in pool if t not in truth_ids], min(a.controls, len(pool))))
    token = bp_token(); olaf.build()
    indexed = 0
    for tid in sorted(truth_ids | controls):
        try:
            url = preview(tid, token)
            if not url: continue
            p = fetch(url); olaf.store(tid, p); os.unlink(p); indexed += 1
        except Exception as e:
            print(f"  index {tid}: {type(e).__name__}", flush=True)
    print(f"indexed {indexed} records ({len(truth_ids)} truth + controls)", flush=True)
    tp = fp = fn = 0; per = []; o_tp = o_fp = o_fn = 0
    for mix_url, T in truth.items():
        try:
            path = mix_audio(mix_url)
            if not path: continue
            hits, dur = olaf.query(path)
            H = {h["track_id"] for h in hits if h["count"] >= 8}
            tp += len(T & H); fp += len(H - T); fn += len(T - H)
            O = ours.get(mix_url, set()); o_tp += len(T & O); o_fp += len(O - T); o_fn += len(T - O)
            per.append({"mix": mix_url[-50:], "truth": len(T), "olaf_hits": len(H), "olaf_agree": len(T & H), "ours_hits": len(O), "ours_agree": len(T & O)})
            print(f"  {mix_url[-40:]}: truth {len(T)}, olaf {len(H)} ({len(T & H)} agree), ours {len(O)} ({len(T & O)} agree)", flush=True)
        except Exception as e:
            print(f"  mix {mix_url[-40:]}: {type(e).__name__}: {str(e)[:60]}", flush=True)
    def pr(tp, fp, fn): return {"precision": round(tp / (tp + fp), 2) if tp + fp else None, "recall": round(tp / (tp + fn), 2) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "indexed": indexed, "mixes": len(per),
           "olaf": pr(tp, fp, fn), "ours": pr(o_tp, o_fp, o_fn), "per_mix": per,
           "note": "precision: of the records a matcher heard, how many the published tracklist confirms; recall: of the tracklist records we hold, how many it found"}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("indexed", "mixes", "olaf", "ours")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
