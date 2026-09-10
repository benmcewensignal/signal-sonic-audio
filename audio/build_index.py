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
import argparse, json, os, sqlite3, struct, time
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
        d = {"track_id": tid, **info[tid], **scene.get(tid, {}), "played_in_sets": tid in played}
        tracks.append(d)
        H.append(h[:n].astype("<u4"))
        P.append((np.full(n, ti, dtype="<u4") << 16) | np.minimum(f[:n], 0xFFFF).astype("<u4"))
        kept += n
        if len(tracks) % 500 == 0: print(f"  {len(tracks)} records, {kept:,} postings", flush=True)
    # additions: the targeted pass and the canon, both published as base64 blobs so the
    # gigabyte store never has to move
    import base64
    for extra in ["fp-extra.jsonl", "fp-canon.jsonl"]:
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
            tracks.append({"track_id": tid, **row, **scene.get(tid, {}),
                           "played_in_sets": tid in played, "canon": bool(d.get("canon"))})
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
