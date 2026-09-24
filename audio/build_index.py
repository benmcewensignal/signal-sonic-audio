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


# The similarity is a dot product, so each dimension's contribution to it is the product of the
# two records' values there. Summing those by family says what the two agree on: a pair that is
# close on tone is a different kind of neighbour from a pair that is close on how the sound
# moves, and "0.80 similar" says neither. The embedding carried here is twenty-six mel
# coefficients, thirteen means then thirteen deviations, followed by seven contrast bands.
# The names are checked against the corpus rather than chosen. Taking records extreme on one
# family and ordinary on the others:
#   mel means      high in raw techno, 140 and trance; low in bass house and tech house. That
#                  spread is tonal colour, from the extreme to the middle of the road.
#   mel deviations high in trance, amapiano and organic house, music with breakdowns and live
#                  playing, one extreme record by Marimba De Guatemala; low in bass house and
#                  garage, which loop. So it is how much the sound changes as it plays.
#   contrast       high in amapiano, deep house and afro house, with Guitar, Love & Music at
#                  the extreme; low in bass house and garage, which are dense. So it is how
#                  clearly you can pick the parts out.
_WHYNAME = {"tone": "the colour of the sound",
            "movement": "how much it changes as it plays",
            "texture": "how clearly the parts stand out"}


def _why(a, b):
    import numpy as _np
    c = _np.asarray(a) * _np.asarray(b)
    if c.size < 33:
        return None
    parts = {"tone": float(c[0:13].sum()),
             "movement": float(c[13:26].sum()),
             "texture": float(c[26:33].sum())}
    tot = sum(v for v in parts.values() if v > 0)
    if tot <= 0:
        return None
    top = max(parts, key=parts.get)
    if parts[top] <= 0:
        return None
    return {"on": _WHYNAME[top], "share": round(parts[top] / tot, 2)}


_STEMF = ["level", "crest", "dynamic_span", "centroid_hz", "rolloff_hz", "flatness",
          "onsets_per_s", "share_of_energy"]


def stem_neighbours(tracks, idx, glob_pat, k=8):
    """Nearest neighbours within each stem, so a record can be searched by one of its parts."""
    import glob as _g, json as _j
    import numpy as _np
    if not glob_pat:
        return 0
    S = {}
    for f in _g.glob(glob_pat):
        for line in open(f):
            try:
                d = _j.loads(line)
            except Exception:
                continue
            st = d.get("stems")
            if isinstance(st, dict):
                S[d.get("track_id")] = st
    if not S:
        print("no stem files matched; skipping the per-stem neighbours", flush=True)
        return 0
    written = 0
    for part in ("drums", "bass", "other", "vocals"):
        rows, keep = [], []
        for gi, ti in enumerate(idx):
            t = tracks[ti]
            st = (S.get(t.get("track_id")) or {}).get(part)
            if not st:
                continue
            v = [st.get(k2) for k2 in _STEMF]
            if any(not isinstance(x, (int, float)) for x in v):
                continue
            rows.append(v); keep.append(gi)
        if len(rows) < 50:
            continue
        X = _np.asarray(rows, dtype=float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        X /= (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        B = 1024
        for start in range(0, len(X), B):
            sim = X[start:start + B] @ X.T
            for row in range(sim.shape[0]):
                a = start + row
                sim[row, a] = -2
                top = _np.argpartition(-sim[row], k)[:k]
                top = top[_np.argsort(-sim[row][top])]
                t = tracks[idx[keep[a]]]
                t.setdefault("stem_near", {})[part] = [
                    {"id": tracks[idx[keep[int(b)]]]["track_id"],
                     "name": tracks[idx[keep[int(b)]]]["name"],
                     "sim": round(float(sim[row, int(b)]), 3)} for b in top]
                written += 1
        print(f"stem neighbours for {part}: {len(rows)} records", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="fingerprints.db"); ap.add_argument("--meta", default="sonic.db")
    ap.add_argument("--out", default="out/index")
    # Stem measures, one file per shard from the features pipeline. Searching by drums alone
    # finds the right scene at three times chance against three and a quarter for the whole
    # record, and shares 0.1 of eight neighbours with it: nearly as informative, and almost
    # entirely different records. That is a second way through the catalogue, and the only one
    # a producer hunting a specific part can use.
    ap.add_argument("--stems-glob", default=None)
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
        # bass_weight reads 1.000 for 99% of records and drum_swing clips for 15%: neither can
        # rank anything. sub_bass and pulse_clarity replace them where the record has been
        # measured on the current analyser; the old keys stay so older records still render.
        ed = d.get("edm") or {}
        m = {k: d.get(k) for k in ("tempo", "drum_density", "drum_swing", "bass_weight", "vocal_presence", "sub_bass", "pulse_clarity", "loudness", "how_played", "harmonic_weight")}
        m["sub_bass"] = ed.get("sub_bass")
        m["pulse_clarity"] = ed.get("pulse_clarity")
        # At scene level most measures collapse into one gradient; at record level nine stay
        # distinct. The card is about a record, so it carries the record-level set: loudness,
        # and two named groups of the unnamed dimensions.
        m["loudness"] = d.get("loudness")
        emb = d.get("embedding") or []
        if len(emb) == 45:
            m["how_played"] = float(sum(emb[40:44]) / 4.0)
            m["harmonic_weight"] = float(sum(emb[26:38]) / 12.0)
        if any(isinstance(v, (int, float)) for v in m.values()):
            out["measures"] = {k: (round(float(v), 3) if isinstance(v, (int, float)) else None)
                               for k, v in m.items()}
        e_all = d.get("embedding")
        if e_all and len(e_all) == 45:
            out["_emb"] = [e_all[i] for i in list(range(0, 26)) + list(range(38, 45))]
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
                keep = {k: m.get(k) for k in ("tempo", "drum_density", "drum_swing", "bass_weight", "vocal_presence", "sub_bass", "pulse_clarity", "loudness", "how_played", "harmonic_weight")
                        if isinstance(m.get(k), (int, float))}
                if keep: own["measures"] = keep
                em = m.get("embedding")
                if em and len(em) == 45:
                    own["_emb"] = [em[i] for i in list(range(0, 26)) + list(range(38, 45))]
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
    for key in ("tempo", "drum_density", "drum_swing", "bass_weight", "vocal_presence", "sub_bass", "pulse_clarity", "loudness", "how_played", "harmonic_weight"):
        for sc, group in by_scene.items():
            vals = [(t["measures"].get(key), t) for t in group
                    if t.get("measures") and isinstance(t["measures"].get(key), (int, float))]
            if len(vals) < 20: continue
            vals.sort(key=lambda x: x[0])
            for i, (_, t) in enumerate(vals):
                t.setdefault("ranks", {})[key] = round((i + 0.5) / len(vals), 3)
    ranked = sum(1 for t in tracks if "dist_rank" in t)
    print(f"ranked {ranked} records against their own scene", flush=True)

    idx = [i for i, t in enumerate(tracks) if t.get("measures") and t.get("_emb")]

    # Position on the two lines the whole site draws scenes on, so a record and its
    # neighbours sit in the same space as the genres rather than on an arbitrary circle.
    # The axes are the first two directions of the field, fixed here from the corpus.
    if len(idx) > 50:
        E = np.array([tracks[i]["_emb"] for i in idx], dtype=float)
        mu = E.mean(0)
        A = E - mu
        sub = A[np.random.default_rng(1).choice(len(A), min(8000, len(A)), replace=False)]
        _, _, Vt = np.linalg.svd(sub, full_matrices=False)
        # orient so that driving is positive: trance must sit above dubstep
        d1 = A @ Vt[0]
        tr = [d1[k] for k, i in enumerate(idx) if tracks[i].get("scene") == "trance-main-floor"]
        du = [d1[k] for k, i in enumerate(idx) if tracks[i].get("scene") == "140-deep-dubstep-grime"]
        sgn = 1.0 if (tr and du and np.mean(tr) > np.mean(du)) else -1.0
        d2 = A @ Vt[1]
        for k, i in enumerate(idx):
            tracks[i]["pos"] = {"driving": round(float(sgn * d1[k]), 4), "melodic": round(float(d2[k]), 4)}
        print(f"positions for {len(idx)} records", flush=True)

    # What else sounds like this? Computed once here rather than shipping 8,403 embeddings
    # to the phone. It is the question a person actually asks after "what is this", and
    # nobody else can answer it: Shazam knows the record, not its neighbours.
    if len(idx) > 50:
        E = np.array([tracks[i]["_emb"] for i in idx], dtype=float)
        # Centre first. Without it every record scores 0.99 against every other: the vectors
        # all point in nearly the same direction, so cosine measures how much a record is a
        # record rather than what kind. Subtracting the average makes the comparison about
        # how each one differs from the middle, which is the question being asked.
        E = E - E.mean(axis=0, keepdims=True)
        E = E / (E.std(axis=0, keepdims=True) + 1e-9)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        B = 512
        for start in range(0, len(idx), B):
            sim = E[start:start + B] @ E.T
            for row, gi in enumerate(range(start, min(start + B, len(idx)))):
                sim[row, gi] = -2.0                       # not itself
                near = np.argsort(-sim[row])[:8]
                tracks[idx[gi]]["near"] = [
                    {"id": tracks[idx[j]]["track_id"],
                     "name": tracks[idx[j]]["name"],
                     "pos": tracks[idx[j]].get("pos"),
                     "artists": (tracks[idx[j]].get("artists") or [])[:2],
                     "scene": tracks[idx[j]].get("scene"),
                     "sim": round(float(sim[row, j]), 3),
                     "why": _why(E[gi], E[j])} for j in near]
        # Which of a record's neighbours are neighbours of each other. The page draws these as
        # the edges of the graph, and without them the picture is a star rather than a network:
        # a centre and eight spokes, with no way to see that three of the eight belong together
        # and the other five do not. The api tried to work this out at request time by looking
        # up each neighbour in the summary index, but the neighbour lists live in the detail
        # file, so the set it checked against was always empty and no edge ever drew.
        own = {}
        for gi in range(len(idx)):
            own[tracks[idx[gi]]["track_id"]] = {n["id"] for n in tracks[idx[gi]].get("near", [])}
        linked = 0
        for gi in range(len(idx)):
            nb = tracks[idx[gi]].get("near", [])
            links = []
            for i2 in range(len(nb)):
                theirs = own.get(nb[i2]["id"], set())
                for j2 in range(i2 + 1, len(nb)):
                    if nb[j2]["id"] in theirs:
                        links.append([i2, j2])
            tracks[idx[gi]]["near_links"] = links
            if links:
                linked += 1
        try:
            n_stem = stem_neighbours(tracks, idx, a.stems_glob)
            if n_stem:
                print(f"per-stem neighbours written for {n_stem} record-stems", flush=True)
        except Exception as e:
            print(f"stem neighbours skipped: {type(e).__name__}: {str(e)[:90]}", flush=True)
        print(f"nearest neighbours for {len(idx)} records, "
              f"{linked} of them with edges between their neighbours", flush=True)
    # Walks: from any record, the nearest record that is meaningfully higher on one named
    # measure and as close as possible on everything else. Similarity keeps the step
    # coherent, the measure gives it a direction. Ten ids a record, and the catalogue
    # becomes traversable along an axis a person understands rather than along a genre.
    # walk along the measures that rank. bass_weight reads the same for ninety-nine per cent of
    # records and drum_swing clips for a sixth: a step along either lands nowhere in particular.
    # The card names eight measures and the walk stepped on four. This list was written when
    # four were all that ranked, and never grew when how played, harmonic weight, loudness
    # and tempo were added. drum_swing and bass_weight stay out: one is a coin flip between
    # techno and house, the other is ninety-nine per cent constant.
    AXES = ["sub_bass", "drum_density", "pulse_clarity", "vocal_presence",
            "how_played", "harmonic_weight", "loudness", "tempo"]
    if len(idx) > 50:
        E = np.array([tracks[i]["_emb"] for i in idx], dtype=float)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        vals = {}
        for ax in AXES:
            v = np.array([tracks[i]["measures"].get(ax) if isinstance(tracks[i]["measures"].get(ax), (int, float))
                          else np.nan for i in idx], dtype=float)
            vals[ax] = v
        B = 256
        for start in range(0, len(idx), B):
            sim = E[start:start + B] @ E.T
            for row, gi in enumerate(range(start, min(start + B, len(idx)))):
                walk = {}
                for ax in AXES:
                    v = vals[ax]
                    here = v[gi]
                    if not np.isfinite(here):
                        continue
                    sd = np.nanstd(v) or 1.0
                    step = {}
                    for name, mask in (("up", v > here + 0.25 * sd), ("down", v < here - 0.25 * sd)):
                        cand = np.where(mask & np.isfinite(v))[0]
                        if cand.size == 0:
                            continue
                        j = cand[np.argmax(sim[row, cand])]       # most alike, in that direction
                        step[name] = {"id": tracks[idx[j]]["track_id"],
                                      "name": tracks[idx[j]]["name"],
                                      "artists": (tracks[idx[j]].get("artists") or [])[:2],
                                      "scene": tracks[idx[j]].get("scene"),
                                      "v": round(float(v[j]), 3),
                                      "sim": round(float(sim[row, j]), 3)}
                    if step:
                        walk[ax] = step
                if walk:
                    tracks[idx[gi]]["walk"] = walk
        print(f"walks for {sum(1 for t in tracks if t.get('walk'))} records", flush=True)

    for t in tracks:
        t.pop("_emb", None)

    # Two files, not one. The matcher needs a name and a scene for eight thousand records;
    # the card needs neighbours and walks for exactly one. Carrying both in the file the API
    # loads on every cold start took it to twenty megabytes and the endpoint stopped
    # responding, which broke recognition itself to serve a feature nobody had tapped yet.
    rich = {}
    for t in tracks:
        extra = {}
        if t.get("near"): extra["near"] = t.pop("near")
        if t.get("walk"): extra["walk"] = t.pop("walk")
        # The per-stem neighbours are four more lists of eight per record. Left in the summary
        # they took it from 8.6 to 29.3 MB and the size guard stopped the build, which is the
        # guard doing exactly what the comment above it describes: this file is loaded on every
        # cold start and recognition itself depends on it staying small.
        if t.get("stem_near"): extra["stem_near"] = t.pop("stem_near")
        if extra: rich[t["track_id"]] = extra
    with open(a.out + "-detail.json", "w") as f:
        json.dump(rich, f, separators=(",", ":"))
    print(f"detail for {len(rich)} records written alongside the index", flush=True)

    with open(a.out + ".bin", "wb") as f:
        f.write(struct.pack("<4sII", b"SFP1", len(H), len(tracks)))
        f.write(H.tobytes()); f.write(P.tobytes())
    # measures and ranks as lists with their names once (mkeys), ranks as whole percentages: the
    # repeated names were 71% of this file, and above about twenty megabytes the API that loads it
    # on every cold start stops responding. The API expands them back into the same objects.
    mkeys = sorted({k for t in tracks for k in (t.get("measures") or {})} | {k for t in tracks for k in (t.get("ranks") or {})})
    for t in tracks:
        if isinstance(t.get("measures"), dict): t["measures"] = [t["measures"].get(k) for k in mkeys]
        if isinstance(t.get("ranks"), dict): t["ranks"] = [None if t["ranks"].get(k) is None else int(round(t["ranks"][k] * 100)) for k in mkeys]
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mkeys": mkeys,
               "tracks": tracks, "postings": int(len(H)),
               "note": "hashes and postings are two parallel little-endian uint32 arrays after a 12-byte header, sorted by hash"},
              open(a.out + ".json", "w"), separators=(",", ":"))
    mb = os.path.getsize(a.out + ".bin") / 1e6
    print(json.dumps({"records": len(tracks), "postings": int(len(H)),
                      "bin_mb": round(mb, 1), "meta_mb": round(os.path.getsize(a.out + ".json") / 1e6, 1)}, indent=1))


if __name__ == "__main__":
    main()
