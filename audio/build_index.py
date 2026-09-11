"""Turn the fingerprint store into a recognition index for the whole corpus.

The store holds 137 million hashes across 8,241 records, packed by track. Recognition
needs the inverse: hash to (track, frame). Done naively that is 1.1 GB of postings and
28 MB of JSON per two hundred records, which is why the demo index stopped at two hundred.

Two changes make the whole corpus fit:

  binary, not JSON      a posting is 8 bytes: a 32-bit hash and a 32-bit packed
                        (track index, frame). JSON spent 68 bytes on the same thing.
  prune what identifies nothing
                        a hash appearing in many records cannot name one, and a record
                        needs only its first minute to be recognisable. Both cuts are
                        lossless for the question being asked.

The result is one sorted array a serverless function can hold in memory and binary-search,
published as a release asset because it is too large for a repository file.

  python -m audio.build_index --db fingerprints.db --meta sonic.db --out out/index
"""
import argparse, collections, json, os, sqlite3, struct, time
import numpy as np

MAX_TRACKS_PER_HASH = 6      # above this a hash names nothing
MAX_HASHES_PER_TRACK = 1800  # about the first minute of a record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="fingerprints.db"); ap.add_argument("--meta", default="sonic.db")
    ap.add_argument("--out", default="out/index")
    a = ap.parse_args()
    fp = sqlite3.connect(a.db)
    n_tracks = fp.execute("select count(*) from fp_tracks").fetchone()[0]
    print(f"store holds {n_tracks} fingerprinted records", flush=True)

    meta = sqlite3.connect(a.meta); meta.row_factory = sqlite3.Row
    info = {r["track_id"]: {"name": r["name"], "artists": json.loads(r["artists"]) if r["artists"] else [],
                            "label": r["label"]}
            for r in meta.execute("select track_id, name, artists, label from track_meta where name is not null")}
    scene = {}
    for r in meta.execute("select track_id, scene, min(chart_rank) rk from track_scenes group by track_id"):
        scene.setdefault(r["track_id"], {"scene": r["scene"], "chart_best": r["rk"]})
    played = {r[0] for r in meta.execute("select distinct track_id from mix_plays")}

    # what the record sounds like, and where that sits against its own scene. Without these
    # a result can say what a record is but not how it compares, which is the half that is
    # not Shazam. The first index carried them; the rewrite for the full corpus lost it.
    feats = {}
    for r in meta.execute("select track_id, features from tracks where analyser_id='local'"):
        try: feats[r["track_id"]] = json.loads(r["features"])
        except Exception: pass
    home = collections.defaultdict(list)
    for r in meta.execute("select scene, track_id from track_scenes where week like '____-M__' and week <= '2025-M05'"):
        d = feats.get(r["track_id"])
        e = d.get("embedding") if d else None
        if e and len(e) == 45: home[r["scene"]].append(e)
    centre = {}
    for sc, es in home.items():
        if len(es) < 20: continue
        E = [np.array(e, float) for e in es]
        E = [e / (np.linalg.norm(e) or 1) for e in E]
        centre[sc] = np.mean(E, axis=0)
    print(f"measures for {len(feats)} records, scene centres for {len(centre)} scenes", flush=True)

    def measured(tid, sc):
        """The record's own numbers, and its distance from where its scene was in 2024."""
        d = feats.get(tid)
        if not d: return {}
        out = {}
        m = {k: d.get(k) for k in ("tempo", "drum_density", "drum_swing", "bass_weight", "vocal_presence")}
        if any(isinstance(v, (int, float)) for v in m.values()):
            out["measures"] = {k: (round(float(v), 3) if isinstance(v, (int, float)) else None)
                               for k, v in m.items()}
        cen = centre.get(sc)
        e = d.get("embedding")
        if cen is not None and e and len(e) == 45:
            v = np.array(e, float); v = v / (np.linalg.norm(v) or 1)
            out["dist_from_scene_2024"] = round(1 - float(v @ cen / ((np.linalg.norm(v) * np.linalg.norm(cen)) or 1)), 4)
        return out

    tracks, H, P = [], [], []
    kept = 0
    for i, row in enumerate(fp.execute("select track_id, n_hashes, hashes, frames from fp_tracks")):
        tid = row[0]
        if tid not in info: continue                      # a record we cannot name is no use in a result
        h = np.frombuffer(row[2], dtype="<u4")
        f = np.frombuffer(row[3], dtype="<u2")
        n = min(len(h), len(f), MAX_HASHES_PER_TRACK)
        if n < 50: continue
        ti = len(tracks)
        sc_info = scene.get(tid, {})
        d = {"track_id": tid, **info[tid], **sc_info, "played_in_sets": tid in played,
             **measured(tid, sc_info.get("scene"))}
        tracks.append(d)
        H.append(h[:n].astype("<u4"))
        P.append((np.full(n, ti, dtype="<u4") << 16) | np.minimum(f[:n], 0xFFFF).astype("<u4"))
        kept += n
        if len(tracks) % 500 == 0: print(f"  {len(tracks)} records, {kept:,} postings", flush=True)
    # additions: the targeted pass and the canon, both published as base64 blobs so the
    # gigabyte store never has to move
    import base64
    for extra in ["fp-extra.jsonl", "fp-canon.jsonl", "fp-history.jsonl"]:
        path = os.path.join(os.path.dirname(a.out), extra)
        if not os.path.exists(path): continue
        n_extra = 0
        for line in open(path):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("found") is False or "hashes" not in d: continue
            tid = d.get("track_id")
            if not tid: continue
            row = info.get(tid) or ({"name": d.get("name"), "artists": d.get("artists", []), "label": None}
                                    if d.get("canon") else None)
            if row is None: continue
            if any(t["track_id"] == tid for t in tracks[-3000:]): continue
            h = np.frombuffer(base64.b64decode(d["hashes"]), dtype="<u4")
            f = np.frombuffer(base64.b64decode(d["frames"]), dtype="<u2")
            n = min(len(h), len(f), MAX_HASHES_PER_TRACK)
            if n < 50: continue
            ti = len(tracks)
            own = {}
            if d.get("measures"):
                m = d["measures"]
                keep = {k: m.get(k) for k in ("tempo", "drum_density", "drum_swing", "bass_weight", "vocal_presence")
                        if isinstance(m.get(k), (int, float))}
                if keep: own["measures"] = keep
                cen = centre.get(d.get("scene"))
                e = m.get("embedding")
                if cen is not None and e and len(e) == 45:
                    v = np.array(e, float); v = v / (np.linalg.norm(v) or 1)
                    own["dist_from_scene_2024"] = round(1 - float(v @ cen / ((np.linalg.norm(v) * np.linalg.norm(cen)) or 1)), 4)
                if d.get("scene"): own["scene"] = d["scene"]
            tracks.append({"track_id": tid, **row, **scene.get(tid, {}),
                           **measured(tid, (scene.get(tid) or {}).get("scene")), **own,
                           "played_in_sets": tid in played, "canon": bool(d.get("canon")),
                           **({"preview": d["preview"]} if d.get("preview") else {})})
            H.append(h[:n].astype("<u4"))
            P.append((np.full(n, ti, dtype="<u4") << 16) | np.minimum(f[:n], 0xFFFF).astype("<u4"))
            n_extra += 1
        print(f"merged {n_extra} records from {extra}", flush=True)
    H = np.concatenate(H); P = np.concatenate(P)
    print(f"before pruning: {len(tracks)} records, {len(H):,} postings", flush=True)

    order = np.argsort(H, kind="stable")
    H, P = H[order], P[order]
    # drop hashes carried by too many records
    uniq, start, count = np.unique(H, return_index=True, return_counts=True)
    common = uniq[count > MAX_TRACKS_PER_HASH]
    if len(common):
        keep = ~np.isin(H, common)
        H, P = H[keep], P[keep]
    print(f"after pruning {len(common):,} over-common hashes: {len(H):,} postings", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # A distance of 0.014 tells a reader nothing. Rank each record against its own scene, so
    # the app can say "further out than four in five of its peers" instead of a number.
    by_scene = collections.defaultdict(list)
    for t in tracks:
        if t.get("scene") and t.get("dist_from_scene_2024") is not None:
            by_scene[t["scene"]].append(t)
    for sc, group in by_scene.items():
        group.sort(key=lambda x: x["dist_from_scene_2024"])
        n = len(group)
        for i, t in enumerate(group):
            t["dist_rank"] = round((i + 0.5) / n, 3)      # 0 is closest to the 2024 sound
    # and the same for each measurement, so a record can be placed ingredient by ingredient
    for key in ("tempo", "drum_density", "drum_swing", "bass_weight", "vocal_presence"):
        for sc, group in by_scene.items():
            vals = [(t["measures"].get(key), t) for t in group
                    if t.get("measures") and isinstance(t["measures"].get(key), (int, float))]
            if len(vals) < 20: continue
            vals.sort(key=lambda x: x[0])
            for i, (_, t) in enumerate(vals):
                t.setdefault("ranks", {})[key] = round((i + 0.5) / len(vals), 3)
    ranked = sum(1 for t in tracks if "dist_rank" in t)
    print(f"ranked {ranked} records against their own scene", flush=True)

    with open(a.out + ".bin", "wb") as f:
        f.write(struct.pack("<4sII", b"SFP1", len(H), len(tracks)))
        f.write(H.tobytes()); f.write(P.tobytes())
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "tracks": tracks, "postings": int(len(H)),
               "note": "hashes and postings are two parallel little-endian uint32 arrays after a 12-byte header, sorted by hash"},
              open(a.out + ".json", "w"), separators=(",", ":"))
    mb = os.path.getsize(a.out + ".bin") / 1e6
    print(json.dumps({"records": len(tracks), "postings": int(len(H)),
                      "bin_mb": round(mb, 1), "meta_mb": round(os.path.getsize(a.out + ".json") / 1e6, 1)}, indent=1))


if __name__ == "__main__":
    main()
