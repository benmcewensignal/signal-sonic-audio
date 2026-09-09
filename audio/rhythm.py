"""Rhythm and loudness per record, on the second runner.

The main analyser measures timbre well and rhythm barely: drum density and swing are
proxies derived from onset strength, not from a beat grid. That matters because groove
is where a house record and a tech house record differ to a human ear, and it is one of
the things the hand-built statistics average away.

Essentia's RhythmExtractor2013 finds the actual beats. From them and the loudness
standard used in broadcast, this measures per record:

  bpm, beat confidence, beat regularity   how strict the grid is
  onset rate                              how much is happening
  danceability                            Essentia's own measure
  integrated loudness, loudness range     the flatness we can currently only infer
  dynamic complexity                      how much the level actually moves

The last three matter most: the field-wide finding is that records got less dynamic, and
we measured that from an eight-point energy curve. This measures it properly.

Results go to out/rhythm-<n>.jsonl for the main pipeline to import, so the two runners
never write to the same database.

  python -m audio.rhythm --db sonic.db --limit 400 --budget-minutes 50
"""
import argparse, collections, json, os, random, sqlite3, tempfile, time, urllib.request
from .beatport import get_token, _get

SR = 44100


def preview(track_id, token):
    d = _get(f"/catalog/tracks/{track_id.split(':')[-1]}/", token)
    return d.get("sample_url") or (d.get("preview") or {}).get("mp3", {}).get("url")


def fetch(url):
    fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "signal-sonic-audio/rhythm"})
    with urllib.request.urlopen(req, timeout=30) as r, open(p, "wb") as f: f.write(r.read())
    return p


def make_measurer():
    import essentia.standard as es, numpy as np
    rhythm = es.RhythmExtractor2013(method="multifeature")
    dance = es.Danceability()
    onset = es.OnsetRate()
    dyn = es.DynamicComplexity()
    def measure(path):
        audio = es.MonoLoader(filename=path, sampleRate=SR)()
        if len(audio) < SR * 5: raise ValueError("under five seconds")
        bpm, beats, conf, _, intervals = rhythm(audio)
        reg = float(np.std(np.diff(beats))) if len(beats) > 3 else None
        d, _ = dance(audio)
        _, orate = onset(audio)
        dc, loud_mean = dyn(audio)
        st = es.LoudnessEBUR128()(np.column_stack([audio, audio]).astype(np.float32))
        return {"bpm": round(float(bpm), 2), "beat_confidence": round(float(conf), 2),
                "beat_irregularity": round(reg, 4) if reg is not None else None,
                "onset_rate": round(float(orate), 3), "danceability": round(float(d), 3),
                "dynamic_complexity": round(float(dc), 3), "loudness_mean": round(float(loud_mean), 2),
                "loudness_integrated": round(float(st[2]), 2), "loudness_range": round(float(st[3]), 2),
                "n_beats": int(len(beats))}
    return measure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sonic.db"); ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--budget-minutes", type=int, default=50); ap.add_argument("--out-dir", default="out")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    have = set()
    for f in sorted(os.listdir(a.out_dir)):
        if f.startswith("rhythm-") and f.endswith(".jsonl"):
            for line in open(os.path.join(a.out_dir, f)):
                try: have.add(json.loads(line)["track_id"])
                except Exception: pass
    c = sqlite3.connect(a.db); c.row_factory = sqlite3.Row
    # spread the sample across scenes and months: sweeping newest-first fills up with whichever
    # scene was backfilled last, and the whole point is to compare 2024 against now
    pool = collections.defaultdict(list)
    for r in c.execute("""select ts.scene, ts.week, ts.track_id from track_scenes ts
                          join tracks t on t.track_id=ts.track_id and t.analyser_id='local'
                          where ts.week like '____-M__' and ts.track_id like 'bp:%'"""):
        if r["track_id"] not in have: pool[(r["scene"], r["week"])].append(r["track_id"])
    todo, rr = [], random.Random(4)
    keys = sorted(pool)
    for k in keys: rr.shuffle(pool[k])
    while len(todo) < a.limit and any(pool[k] for k in keys):
        for k in keys:                                  # one per scene-month, round robin
            if pool[k] and len(todo) < a.limit: todo.append(pool[k].pop())
    print(f"rhythm: {len(have)} already measured, {len(todo)} this run", flush=True)
    if not todo: return
    token = get_token(); measure = make_measurer()
    n = len([f for f in os.listdir(a.out_dir) if f.startswith("rhythm-")])
    path = os.path.join(a.out_dir, f"rhythm-{n:03d}.jsonl")
    t0 = time.time(); done = err = 0
    with open(path, "w") as out:
        for tid in todo:
            if (time.time() - t0) / 60 > a.budget_minutes:
                print("budget reached", flush=True); break
            p = None
            try:
                url = preview(tid, token)
                if not url: raise ValueError("no preview url")
                p = fetch(url)
                rec = measure(p); rec["track_id"] = tid
                out.write(json.dumps(rec) + "\n"); done += 1
                if done % 50 == 0: out.flush(); print(f"  {done}/{len(todo)}, {err} failed", flush=True)
            except Exception as e:
                err += 1
                if err <= 3: print(f"  {tid}: {type(e).__name__}: {str(e)[:70]}", flush=True)
            finally:
                if p:
                    try: os.unlink(p)
                    except OSError: pass
    print(f"rhythm: {done} measured, {err} failed, written to {path}", flush=True)


if __name__ == "__main__":
    main()
