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

# Which fingerprints each record keeps in the published index. The store holds every fingerprint of each record's two-minute
# preview, about 180 a second; the index takes the first 1,800, the first seven seconds or so (this used to say "about the
# first minute", which was wrong by the density), and of those keeps the KEEP_PER_RECORD rarest across the index, never one
# shared by more than MAX_SHARING postings. Until 17 Oct it instead dropped every hash shared by more than six records, a
# fixed number that pruned harder as the index grew: at 113,000 records it left a median of six fingerprints a record and
# 53% of records under the eight a match needs. Measured on those 113,000: the old rule kept 3.7M postings (30 MB) and
# 21,603 records with fifty or more fingerprints; this one keeps 11.1M (88 MB) and gives every record a hundred, with the
# kept hashes shared by a median of 15 postings and at most MAX_SHARING, so the API's scan of 200,000 postings a query is
# never near. A clip from elsewhere in a record still cannot match here: that is the catalogue-wide index on Modal
# (signal-sonic-features, worker/recognise_build_all.py), which the API asks when this one is not confident.
KEEP_PER_RECORD = 60    # 100 from 17 Oct 09:00 to 10:30; 60 since, with the cap on records lifted (CAP below)
MAX_SHARING = 500
MAX_HASHES_PER_TRACK = 1800  # the first seven seconds or so of a record, at about 180 fingerprints a second


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


def record_keys(tracks, idx, glob_pat):
    """Each record's key from its separated melody, bass and voice, with dance-music key templates; matches
    Beatport's key 77% of the time when clearest, 61% in the middle, 43% when least clear."""
    import glob as _g, json as _j
    import numpy as _np
    if not glob_pat: return 0
    K = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    MA = _np.array([.1652, .0475, .0829, .0669, .0999, .0927, .0529, .1316, .0522, .0744, .0694, .0643]); MI = _np.array([.1724, .0400, .0761, .1253, .0567, .0822, .0626, .1435, .0810, .0571, .0836, .0551])
    CAM = {"G#m": "1A", "D#m": "2A", "A#m": "3A", "Fm": "4A", "Cm": "5A", "Gm": "6A", "Dm": "7A", "Am": "8A", "Em": "9A", "Bm": "10A", "F#m": "11A", "C#m": "12A",
           "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B", "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B"}
    S = {}
    for f in _g.glob(glob_pat):
        for line in open(f):
            if '"embedding"' not in line: continue
            try: d = _j.loads(line)
            except Exception: continue
            st = d.get("stems")
            if isinstance(st, dict): S[d.get("track_id")] = st
    n = 0
    for ti in idx:
        st = S.get(tracks[ti].get("track_id"))
        if not st: continue
        def ch(p):
            e = (st.get(p) or {}).get("embedding")
            if not (isinstance(e, list) and len(e) == 45): return None
            v = _np.array(e[26:38], float); return (v - v.mean()) / (v.std() + 1e-9)
        o = ch("other")
        if o is None: continue
        v = o + sum(0.5 * x for x in (ch("bass"), ch("vocals")) if x is not None)
        sc = sorted(((float(_np.corrcoef(_np.roll(v, -i), pr)[0, 1]), K[i] + sf) for i in range(12) for pr, sf in ((MA, ""), (MI, "m"))), reverse=True)
        k, m = sc[0][1], sc[0][0] - sc[1][0]
        tracks[ti]["key"] = {"key": k, "camelot": CAM.get(k), "matches_beatport": 0.77 if m >= 0.108 else (0.61 if m >= 0.048 else 0.43)}; n += 1
    print(f"keys written for {n} records", flush=True)
    return n


def part_walks(tracks, idx, glob_pat, pool=40):
    """Walk by part: for each record and each part to keep, the record whose same part is among the
    40 closest while a chosen other part is as different as possible ("same drums, different bass")."""
    import glob as _g, json as _j
    import numpy as _np
    if not glob_pat: return 0
    PARTS = ("drums", "bass", "other", "vocals"); S = {}
    for f in _g.glob(glob_pat):
        for line in open(f):
            try: d = _j.loads(line)
            except Exception: continue
            st = d.get("stems")
            if isinstance(st, dict): S[d.get("track_id")] = st
    rows, keep = {p: [] for p in PARTS}, []
    for gi, ti in enumerate(idx):
        st = S.get(tracks[ti].get("track_id")) or {}
        vs = []
        for p in PARTS:
            v = [(st.get(p) or {}).get(k2) for k2 in _STEMF]
            if any(not isinstance(x, (int, float)) for x in v): break
            vs.append(v)
        if len(vs) < 4: continue
        keep.append(ti)
        for p, v in zip(PARTS, vs): rows[p].append(v)
    if len(keep) < 200: return 0
    Z = {}
    for p in PARTS:
        X = _np.asarray(rows[p], dtype=float); X = (X - X.mean(0)) / (X.std(0) + 1e-9); Z[p] = X / (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    # Mixable first: a DJ can only walk to a record at a tempo they can mix (within 6 per cent, or double or half).
    # Without this, 15 of 17 steps from a drum & bass record landed at 129 to 197 BPM. Fall back to the closest
    # regardless of tempo only when fewer than ten mixable records share the part closely, so no record loses its walks.
    Tm = _np.array([((tracks[ti].get("measures") or {}).get("tempo") or 0) for ti in keep], dtype=float)
    written = 0; B = 512
    for start in range(0, len(keep), B):
        sims = {p: Z[p][start:start + B] @ Z[p].T for p in PARTS}
        for r in range(sims["drums"].shape[0]):
            a = start + r; out = {}
            T0 = Tm[a]
            mixable = ((Tm > 0) & ((_np.abs(Tm - T0) <= 0.06 * T0) | (_np.abs(Tm * 2 - T0) <= 0.06 * T0) | (_np.abs(Tm / 2 - T0) <= 0.06 * T0))) if T0 > 0 else _np.ones(len(keep), bool)
            for A in PARTS:
                sa = sims[A][r].copy(); sa[a] = -2
                sm = _np.where(mixable, sa, -2.0)
                if int((sm > 0.5).sum()) >= 10: sa = sm   # enough mixable records share this part closely
                cand = _np.argpartition(-sa, pool)[:pool]
                for Bp in PARTS:
                    if Bp == A: continue
                    j = int(cand[_np.argmin(sims[Bp][r][cand])])
                    out[A + ">" + Bp] = tracks[keep[j]]["track_id"]
            tracks[keep[a]]["part_walk"] = out; written += 1
    print(f"walks by part written for {written} records", flush=True)
    return written


def detail_shard(tid):
    """Which of the 64 detail files holds a record: FNV-1a of its id, the same sum the api computes."""
    h = 0x811c9dc5
    for b in str(tid).encode("utf-8"):
        h ^= b; h = (h * 0x01000193) & 0xffffffff
    return h % 64


# Postings are written as SFP2: the record's place in the upper 19 bits and the frame in the lower 13 (api/listen.js reads
# both this and the old SFP1, 16 and 16, which could address only 65,536 records). A frame is 23 ms and the index keeps
# the first seconds of each record (MAX_HASHES_PER_TRACK), so 8,192 frames is plenty.
# The cap was 113,000 from 13 to 17 Oct, set by a 14 MB guard on index.json, which the API loads on every cold start; the
# guard came from a 20 MB version that stopped the endpoint when each record still carried its neighbours and walks. With
# the index filling on 15 Oct, every build swapped charted records (which carry a scene and a place on the map) for
# records one DJ plays (which do not), and records the Modal index recognised had no page: 52,000 by 17 Oct. The cap is
# now the format's, the guard 32 MB (about 330,000 records at 96 bytes each), and each record keeps 60 fingerprints
# rather than 100 to hold index.bin, which the API also fetches on a cold start, near 80 MB at 165,000 records.
PLACE_BITS, FRAME_MASK = 19, 0x1FFF
CAP = 1 << PLACE_BITS


def importance(meta):
    """How much a record matters to recognition, as a sort key (smaller first): records people name and test with (the
    canon and classics), then records with evidence of use, then the catalogue sample. Use is DJ sets (how many DJs
    play the record, from MixesDB tracklists and scanned mixes) and Beatport's charts (the best position reached).
      tier 0  canon or classic
      tier 1  played by two or more DJs, or a top-10 chart position
      tier 2  played by one DJ, or charted at all
      tier 3  the rest: records sampled from the catalogue by genre and month
    Within a tier: more DJs, then a higher chart position, then the newer record."""
    djs = collections.defaultdict(set)
    for dj, tid in meta.execute("select dj, track_id from tracklist_plays where track_id is not null and track_id != ''"):
        djs[tid].add(dj)
    for url, tid in meta.execute("select mix_url, track_id from mix_plays where track_id is not null"):
        djs[tid].add("mix:" + str(url))
    best = {r[0]: r[1] for r in meta.execute("select track_id, min(chart_rank) from track_scenes where chart_rank is not null group by track_id")}
    src, seen = {}, {}
    for tid, s_, fs in meta.execute("select track_id, source, first_seen from tracks"):
        src[tid] = s_ or ""; seen[tid] = str(fs or "")
    def key(t):
        tid = t["track_id"]; n = len(djs.get(tid, ())); rk = best.get(tid)
        if t.get("canon") or src.get(tid) in ("canon", "classic"): tier = 0
        elif n >= 2 or (rk is not None and rk <= 10): tier = 1
        elif n == 1 or rk is not None: tier = 2
        else: tier = 3
        newer = -int("".join(ch for ch in seen.get(tid, "")[:10] if ch.isdigit()) or 0)
        return (tier, -n, rk if rk is not None else 999, newer, tid)
    return key


def pick_within_cap(tracks, H, FR, meta, cap=CAP):
    """All the records if they fit; otherwise the most important that do. The ones left out are the catalogue sample
    first, so a full index is full of the records people play."""
    if len(tracks) <= cap:
        print(f"{len(tracks)} records, within the {cap:,} the index can address", flush=True)
        return tracks, H, FR
    key = importance(meta)
    order = sorted(range(len(tracks)), key=lambda i: key(tracks[i]))
    keep = sorted(order[:cap])            # the store's own order, so neighbours and walks are built as before
    tiers = collections.Counter(key(tracks[i])[0] for i in order[cap:])
    msg = (f"{len(tracks):,} fingerprinted records, {cap:,} fit: left out {len(tracks) - cap:,}, by tier "
           + ", ".join(f"{k}: {v:,}" for k, v in sorted(tiers.items())) + " (0 canon, 1 two or more DJs or top 10, 2 one DJ or charted, 3 catalogue)")
    print(msg, flush=True)
    return [tracks[i] for i in keep], [H[i] for i in keep], [FR[i] for i in keep]


def keep_rarest(H, P, shift, k=KEEP_PER_RECORD, cap=MAX_SHARING):
    """Each record's k rarest fingerprints (see KEEP_PER_RECORD above), with H left sorted by hash, as the API's binary
    search needs. Sharing is counted as postings per hash across the index, as the old rule counted it."""
    order = np.argsort(H, kind="stable")
    H, P = H[order], P[order]; del order
    starts = np.flatnonzero(np.concatenate(([True], H[1:] != H[:-1])))
    runlen = np.diff(np.append(starts, len(H))); del starts
    sharing = np.repeat(np.minimum(runlen, 2**31 - 1).astype(np.int32), runlen); del runlen
    rec = (P >> shift).astype(np.uint32)
    by_rec = np.argsort(rec, kind="stable")                    # positions grouped by record, hash order within each
    rs = rec[by_rec]
    b = np.flatnonzero(np.concatenate(([True], rs[1:] != rs[:-1]))); e = np.append(b[1:], len(rs)); del rs
    keep = np.zeros(len(H), dtype=bool)
    for s, t in zip(b, e):
        pos = by_rec[s:t]; sh = sharing[pos]
        ok = np.flatnonzero(sh <= cap)
        if len(ok) <= k:
            keep[pos[ok]] = True
        else:
            keep[pos[ok[np.argsort(sh[ok], kind="stable")[:k]]]] = True
    return H[keep], P[keep]


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
            # sonic.metadata stored "ERROR: HTTP Error 403: Forbidden" as the name of a record it could not fetch, and five
            # records went out under that name on 13 Oct
            for r in meta.execute("select track_id, name, artists, label from track_meta where name is not null and name not like 'ERROR:%'")}
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

    # Postings were packed with the record's place in the upper 16 bits (SFP1), so the index held at most 65,536 records:
    # it passed that on 10 Oct and the records beyond it were read as the records 65,536 places earlier (7,802 records
    # answered as someone else). Frames are kept per record and packed (SFP2, see CAP) once the records that fit are
    # chosen, most important first (pick_within_cap below).
    tracks, H, FR = [], [], []
    at_ = {}
    kept = 0
    for i, row in enumerate(fp.execute("select track_id, n_hashes, hashes, frames from fp_tracks")):
        tid = row[0]
        if tid not in info: continue                      # a record we cannot name is no use in a result
        if tid in at_: continue
        h = np.frombuffer(row[2], dtype="<u4")
        f = np.frombuffer(row[3], dtype="<u2")
        n = min(len(h), len(f), MAX_HASHES_PER_TRACK)
        if n < 50: continue
        ti = len(tracks)
        sc_info = scene.get(tid, {})
        d = {"track_id": tid, **info[tid], **sc_info, "played_in_sets": tid in played,
             **measured(tid, sc_info.get("scene"))}
        tracks.append(d); at_[tid] = ti
        H.append(h[:n].astype("<u4"))
        FR.append(np.minimum(f[:n], 0xFFFF).astype("<u4"))
        kept += n
        if len(tracks) % 500 == 0: print(f"  {len(tracks)} records, {kept:,} postings", flush=True)
    # additions: the targeted pass and the canon, both published as base64 blobs so the
    # gigabyte store never has to move
    import base64
    # each family may be split across files (fp-canon-<run>.jsonl beside fp-canon.jsonl): GitHub refuses any file over
    # 100 MB, and one canon run of 2,000 records appended about 300 MB, so its push failed every time from 28 September
    import glob as _gl
    _d = os.path.dirname(a.out)
    for path in [p_ for fam in ("fp-extra", "fp-canon", "fp-history") for p_ in sorted(_gl.glob(os.path.join(_d, fam + "*.jsonl")))]:
        extra = os.path.basename(path)   # the rest of the loop names the file by this
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
            if tid in at_:                                # once only (the old check looked back 3,000 records: 693 twice)
                if d.get("canon"): tracks[at_[tid]]["canon"] = True
                continue
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
            at_[tid] = ti
            H.append(h[:n].astype("<u4"))
            FR.append(np.minimum(f[:n], 0xFFFF).astype("<u4"))
            n_extra += 1
        print(f"merged {n_extra} records from {extra}", flush=True)
    tracks, H, FR = pick_within_cap(tracks, H, FR, meta)
    shift = np.uint32(32 - PLACE_BITS)                    # SFP2: the place above the 13 frame bits (api/listen.js: P >>> 13)
    P = [(np.full(len(f_), ti, dtype="<u4") << shift) | np.minimum(f_, FRAME_MASK).astype("<u4") for ti, f_ in enumerate(FR)]
    H = np.concatenate(H); P = np.concatenate(P)
    print(f"before pruning: {len(tracks)} records, {len(H):,} postings", flush=True)

    H, P = keep_rarest(H, P, shift)
    kept = np.bincount(P >> shift, minlength=len(tracks))
    print(f"after keeping each record's {KEEP_PER_RECORD} rarest fingerprints: {len(H):,} postings; records with fewer than 8: "
          f"{int((kept < 8).sum()):,}, with {KEEP_PER_RECORD}: {int((kept >= KEEP_PER_RECORD).sum()):,}", flush=True)

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
                # the eight closest, without sorting all the others: a full sort per record is n log n, and with every
                # fingerprinted record in the index (84,456 on 13 Oct) it made this loop the slowest part of the build
                near = np.argpartition(-sim[row], 8)[:8]
                near = near[np.argsort(-sim[row][near])]
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
            try:
                record_keys(tracks, idx, a.stems_glob)
            except Exception as e_:
                print("keys skipped:", type(e_).__name__, e_, flush=True)
            try:   # stay in key: the nearest sound neighbour whose key mixes harmonically (Camelot: same, one step, or relative)
                KB = {t["track_id"]: (t.get("key") or {}).get("camelot") for t in tracks if t.get("key")}
                def cam(c): return (int(c[:-1]), c[-1]) if c else None
                def fits(a_, b_):
                    if not a_ or not b_: return False
                    (na, la), (nb, lb) = a_, b_
                    return (na == nb) or (la == lb and (na - nb) % 12 in (1, 11))
                nk = 0
                for t in tracks:
                    me = cam(KB.get(t.get("track_id")))
                    for nb_ in (t.get("near") or []):
                        c_ = KB.get(nb_.get("id"))
                        if me and fits(me, cam(c_)):
                            t["near_key"] = {"id": nb_["id"], "name": nb_.get("name"), "artists": nb_.get("artists"), "camelot": c_}; nk += 1; break
                print(f"in-key steps written for {nk} records", flush=True)
            except Exception as e_:
                print("in-key steps skipped:", type(e_).__name__, e_, flush=True)
            try:
                part_walks(tracks, idx, a.stems_glob)
            except Exception as e_:
                print("walks by part skipped:", type(e_).__name__, e_, flush=True)
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
        # each measure's spread and which records carry it, once: worked out again for every record and every measure,
        # they cost more than the step itself
        sds = {ax: (np.nanstd(vals[ax]) or 1.0) for ax in AXES}
        fin = {ax: np.isfinite(vals[ax]) for ax in AXES}
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
                    sd = sds[ax]
                    step = {}
                    for name, mask in (("up", v > here + 0.25 * sd), ("down", v < here - 0.25 * sd)):
                        cand = np.where(mask & fin[ax])[0]
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
        if t.get("part_walk"): extra["part_walk"] = t.pop("part_walk")
        if t.get("key"): extra["key"] = t.pop("key")   # detail only
        if t.get("near_key"): extra["near_key"] = t.pop("near_key")   # detail only: the lean index stays small
        # The recognition index keeps only what matching and search need (id, title, artists, scene, chart and
        # DJ-play counts, how typical it is, a flag for a fully measured record and its tempo). Measures, ranks,
        # map position and neighbour links go to the record's detail file, which the api merges into every record
        # it returns: at 12 MB of a 14 MB limit, the index could not grow past about 45,000 records.
        FULL8 = ("how_played", "vocal_presence", "drum_density", "sub_bass", "pulse_clarity", "loudness", "harmonic_weight", "tempo")
        rk = t.get("ranks") or {}
        if isinstance(rk, dict) and all(isinstance(rk.get(k), (int, float)) for k in FULL8): t["f"] = 1
        ms = t.get("measures") or {}
        if isinstance(ms, dict) and isinstance(ms.get("tempo"), (int, float)): t["tm"] = round(float(ms["tempo"]), 1)
        for k in ("measures", "ranks", "pos", "near_links", "dist_from_scene_2024", "preview"):
            if t.get(k) is not None: extra[k] = t.pop(k)
        if extra: rich[t["track_id"]] = extra
    # Sixty-four small files, not one: the single file passed 100 MB, which GitHub will not store, so from
    # 23 September no build could save it and every record recognised since had no walks or neighbours.
    # Each record's details sit in the file its id hashes to, and the api fetches only that one.
    import os as _os
    _os.makedirs(_os.path.join(_os.path.dirname(a.out) or ".", "detail"), exist_ok=True)
    shards = [dict() for _ in range(64)]
    for tid, extra in rich.items(): shards[detail_shard(tid)][tid] = extra
    for k, sh in enumerate(shards):
        with open(_os.path.join(_os.path.dirname(a.out) or ".", "detail", "%02d.json" % k), "w") as f:
            json.dump(sh, f, separators=(",", ":"))
    print(f"detail for {len(rich)} records written alongside the index, in 64 files", flush=True)

    with open(a.out + ".bin", "wb") as f:
        f.write(struct.pack("<4sII", b"SFP2", len(H), len(tracks)))
        f.write(H.tobytes()); f.write(P.tobytes())
    # measures and ranks as lists with their names once (mkeys), ranks as whole percentages: the
    # repeated names were 71% of this file, and above about twenty megabytes the API that loads it
    # on every cold start stops responding. The API expands them back into the same objects.
    mkeys = sorted({k for t in tracks for k in (t.get("measures") or {})} | {k for t in tracks for k in (t.get("ranks") or {})})
    for t in tracks:
        if isinstance(t.get("measures"), dict): t["measures"] = [t["measures"].get(k) for k in mkeys]
        if isinstance(t.get("ranks"), dict): t["ranks"] = [None if t["ranks"].get(k) is None else int(round(t["ranks"][k] * 100)) for k in mkeys]
        # leaner still, so the index stays under the API's limit as the corpus grows (every full build
        # failed its 14 MB guard from 25 September, at 15.6 MB, and recognition stopped learning new
        # records): two decimals for measures and map positions, which are compared far more coarsely,
        # whole numbers for large measures such as tempo and loudness, and no label, which nothing reads
        if isinstance(t.get("measures"), list):
            t["measures"] = [None if v is None else (round(v, 1) if abs(v) >= 10 else round(v, 2)) for v in t["measures"]]
        if isinstance(t.get("pos"), dict):
            t["pos"] = {k: (round(v, 2) if isinstance(v, (int, float)) else v) for k, v in t["pos"].items()}
        if isinstance(t.get("dist_from_scene_2024"), (int, float)): t["dist_from_scene_2024"] = round(t["dist_from_scene_2024"], 2)
        t.pop("label", None)
        for k in [k for k, v in t.items() if k not in ("track_id", "measures", "ranks") and (v is None or v == 0 or v == [] or v == {} or v == "")]:
            t.pop(k)   # empty fields are read as missing anyway
    scenes_list = sorted({t["scene"] for t in tracks if isinstance(t.get("scene"), str)}); si_ = {x: i for i, x in enumerate(scenes_list)}
    for t in tracks:
        if isinstance(t.get("scene"), str): t["scene"] = si_[t["scene"]]   # one list of names; the API maps them back
    json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mkeys": mkeys, "scenes": scenes_list,
               "tracks": tracks, "postings": int(len(H)),
               "note": "hashes and postings are two parallel little-endian uint32 arrays after a 12-byte header, sorted by hash"},
              open(a.out + ".json", "w"), separators=(",", ":"))
    mb = os.path.getsize(a.out + ".bin") / 1e6
    print(json.dumps({"records": len(tracks), "postings": int(len(H)),
                      "bin_mb": round(mb, 1), "meta_mb": round(os.path.getsize(a.out + ".json") / 1e6, 1)}, indent=1))


if __name__ == "__main__":
    main()
