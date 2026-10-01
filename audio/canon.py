"""The records a scene is defined by, rather than the records selling this week.

Our corpus is Beatport's rolling chart, so it holds what is current and nothing older.
A recognition demo built on it misses everything anyone would call a classic. This is a
hand-compiled canon per scene, drawn from the standard published lists, searched against
Beatport by artist and title, and fingerprinted where a match exists.

Two honest limits, both reported rather than hidden:

  it is curated, not measured    someone chose these; no instrument did
  much of it will not be there   Beatport is a store for current dance records, and a
                                 1986 Chicago house record may never have been listed.
                                 The job reports the hit rate so we know what we hold.

  python -m audio.canon --out out/fp-canon.jsonl --limit 200
"""
import argparse, base64, json, os, tempfile, time, urllib.parse, urllib.request
import numpy as np
from . import fingerprints as FP
try:
    from sonic.analyser_local import LocalAnalyser
except Exception as _e:
    LocalAnalyser = None
    _ANALYSER_ERROR = repr(_e)[:120]
else:
    _ANALYSER_ERROR = None
from .beatport import get_token, _get

CANON = {
"house": ["Mr Fingers Can You Feel It","Marshall Jefferson Move Your Body","Joe Smooth Promised Land",
 "Frankie Knuckles Your Love","Robin S Show Me Love","CeCe Peniston Finally","Crystal Waters Gypsy Woman",
 "Ce Ce Rogers Someday","Todd Terry Something Goin On","Armand Van Helden You Dont Know Me",
 "Daft Punk Around The World","Stardust Music Sounds Better With You","Green Velvet La La Land",
 "Jaydee Plastic Dreams","Alison Limerick Where Love Lives","Inner City Good Life","Chip E Time To Jack"],
"deep-house": ["Larry Heard Mysteries Of Love","Kerri Chandler Rain","Moodymann Shades Of Jae",
 "Ron Trent Altered States","St Germain Rose Rouge","Fred P Just Beginning","Theo Parrish Falling Up",
 "Blaze Lovelee Dae","Glenn Underground Jazzin","MK Always"],
"tech-house": ["Green Velvet Flash","Jamie Jones Hungry For The Power","Hot Natured Benediction",
 "Fisher Losing It","Camelphat Cola","Solardo XTC","Patrick Topping Be Sharp Say Nowt",
 "Michael Bibi Hanging Tree","Chris Lake Turn Off The Lights"],
"techno-peak-time": ["Jeff Mills The Bells","Underground Resistance Jaguar","Adam Beyer Your Mind",
 "Chris Liebing Auf Und Ab","Charlotte de Witte Sgadi Li Mi","Amelie Lens Contradiction",
 "Umek Gatex","Joris Voorn Ringo"],
"techno-raw-deep-hypnotic": ["Derrick May Strings Of Life","Robert Hood Minus","Surgeon Magnese",
 "Model 500 No UFOs","Cybotron Clear","Carl Craig Bug In The Bassbin","Basic Channel Phylyps Trak",
 "Donato Dozzy Menta","Oscar Mulero Nano"],
"hard-techno": ["Sara Landry Pray For Me","999999999 300000003","I Hate Models Daydream",
 "Perc Look What Your Love Has Done","Klangkuenstler Dreh Dich Nicht Um"],
"trance-main-floor": ["Robert Miles Children","ATB 9pm Till I Come","Paul van Dyk For An Angel",
 "Binary Finary 1998","Chicane Saltwater","Above and Beyond Sun and Moon","Armin van Buuren Communication",
 "Tiesto Adagio For Strings","Ferry Corsten Out Of The Blue","System F Out Of The Blue"],
"psy-trance": ["Infected Mushroom Becoming Insane","Astrix Deep Jungle Walk","Vini Vici Great Spirit",
 "Hallucinogen LSD","Ace Ventura Presence"],
"drum-and-bass": ["Goldie Inner City Life","LTJ Bukem Horizons","Roni Size Brown Paper Bag",
 "Shy FX Original Nuttah","Dillinja Valve Sound","Pendulum Tarantula","Andy C Body Rock",
 "High Contrast Return Of Forever","Sub Focus Timewarp","Netsky Iron Heart","Chase and Status Blind Faith",
 "Dimension Whip Slap","Wilkinson Afterglow"],
"uk-garage-speed-garage": ["MJ Cole Sincere","Artful Dodger Re Rewind","Double 99 Ripgroove",
 "Todd Edwards Saved My Life","Sunship Try Me Out","DJ Luck and MC Neat A Little Bit Of Luck",
 "Wookie Battle","Zed Bias Neighbourhood","Interplanetary Criminal B2B","Conducta Rain"],
"140-deep-dubstep-grime": ["Skream Midnight Request Line","Benga Night","Digital Mystikz Anti War Dub",
 "Mala Changes","Coki Spongebob","Loefah Mud","Kode9 9 Samurai","Burial Archangel","Joker Purple City"],
"breaks-breakbeat-uk-bass": ["The Prodigy Out Of Space","Chemical Brothers Block Rockin Beats",
 "Fatboy Slim Right Here Right Now","Krafty Kuts Gimme The Breaks","Stanton Warriors Da Antidote",
 "Plump DJs Big Groovy"],
"bass-house": ["Jauz Feel The Volume","AC Slater Bass Inside","Habstrakt Chicken Soup",
 "Malaa Notorious","Chris Lorenzo Perfect Storm"],
"afro-house": ["Black Coffee Superman","Culoe De Song Webaba","Da Capo Ubuntu","Black Motion Rainbow",
 "Caiiro The Akan","Themba Modimo","Enoo Napa Drums Of Peace"],
"amapiano": ["Kabza De Small Sponono","DJ Maphorisa Izolo","Focalistic Ke Star","Musa Keys Selina",
 "Uncle Waffles Tanzania","Tyler ICU Mnike","Young Stunna Adiwele"],
"uk-funky-gqom": ["Crazy Cousinz Do You Mind","Roska Elevated Levels","Distal Ancient Mystic",
 "DJ Lag Ice Drop","Rudeboyz Gqom","Citizen Boy Ghost"],
"melodic-house-techno": ["Tale Of Us Nova","Adriatique X","Artbat Return","Mind Against Walk The Line",
 "Stephan Bodzin Singularity","Agents Of Time Miracle"],
"progressive-house": ["Sasha Xpander","John Digweed Heaven Scent","Eric Prydz Opus","Deadmau5 Strobe",
 "Guy J Save Me","Hernan Cattaneo Sundown"],
"organic-house": ["Bedouin Melodia","Acid Pauli Mst","Nicola Cruz Colibria","Bora Uzer Kite",
 "Sabo Kalimba"],
"indie-dance": ["LCD Soundsystem Dance Yrself Clean","Hot Chip Over And Over","Roisin Murphy Murphys Law",
 "Moderat A New Error","Trentemoller Moan"],
}


import re as _re, unicodedata as _ud, zlib as _zlib
def _toks(s):
    s = _ud.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return [w for w in _re.sub(r"[^a-z0-9]+", " ", s).split() if len(w) >= 3 and w not in ("feat", "the", "and", "original", "mix", "remix", "edit", "extended", "radio", "version", "remastered", "remaster", "club", "vocal", "dub")]
def _fits(term, hit):
    """the hit is this record: an artist of the hit appears in the query, and the query's title words appear in the hit's name.
    (A first search hit was taken as found before: "Kassian - Metro" came back as another artist's "Metro".)"""
    q = set(_toks(term)); arts = set(w for a_ in hit.get("artists") or [] for w in _toks(a_)); name = set(_toks(hit.get("name")))
    if not (arts & q): return False
    title = q - arts
    return not title or len(title & name) >= max(1, (len(title) + 1) // 2)

def _same_recording(hit):
    """versions worth fingerprinting for recognising a classic: the same recording at another length or point (original,
    extended, radio edit, remaster, dub), not an acapella or another artist's remix, which are different recordings"""
    nm = str(hit.get("name") or "").lower()
    if any(w in nm for w in ("acapella", "a cappella", "acappella")): return False
    if "remix" in nm and not any(w in nm for w in ("original", "remaster")): return False
    return True

def search(term, token, per_page=20):
    q = urllib.parse.quote(term)
    d = _get(f"/catalog/search/?q={q}&type=tracks&per_page={per_page}", token)
    rows = (d.get("tracks") or d.get("results") or []) if isinstance(d, dict) else []
    out = []
    for t in rows:
        tid = "bp:" + str(t.get("id"))
        name = t.get("name") or ""
        arts = [a.get("name") for a in (t.get("artists") or []) if a.get("name")]
        url = t.get("sample_url") or ((t.get("preview") or {}).get("mp3") or {}).get("url")
        out.append({"track_id": tid, "name": (name + (" (" + t.get("mix_name") + ")" if t.get("mix_name") else "")).strip(), "artists": arts, "preview": url,
                    "sample_start_ms": t.get("sample_start_ms"), "sample_end_ms": t.get("sample_end_ms"), "length_ms": t.get("length_ms")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/fp-canon.jsonl"); ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--list", default="", help="a further list to find: the records DJs play most that Sonic cannot yet recognise")
    ap.add_argument("--budget-minutes", type=int, default=80)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--versions", type=int, default=4, help="every version Beatport holds of a classic, up to this many: each preview sits at a different point")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    have = set(); have_ids = set()
    import glob as _gl   # every canon file counts as done: runs now write their own file (fp-canon-<run>.jsonl)
    for path_ in set(_gl.glob(os.path.join(os.path.dirname(a.out) or ".", "fp-canon*.jsonl")) + ([a.out] if os.path.exists(a.out) else [])):
        for line in open(path_):
            try:
                r_ = json.loads(line)
                if r_.get("pass") == "v2": have.add(r_.get("query"))   # the first pass took unchecked first hits: every query is searched again once
                if r_.get("track_id"): have_ids.add(r_["track_id"])
            except Exception: pass
    if LocalAnalyser is None:
        print(f"WARNING: no analyser, records will be fingerprinted but not measured ({_ANALYSER_ERROR})", flush=True)
    token = get_token()
    t0 = time.time(); found = missing = err = measured_n = 0; tried = 0; first_errors = []; first_measure_error = []
    report = []
    mode = "a" if os.path.exists(a.out) else "w"
    with open(a.out, mode) as out:
        todo = dict(CANON)
        if getattr(a, "list", ""):
            for lp_ in [x for x in a.list.split(",") if x]:
                try:
                    for q in json.load(open(lp_))["queue"]: todo.setdefault(q.get("scene") or "unknown", []).append(q["term"])
                except Exception as e:
                    print("list not read:", lp_, type(e).__name__, e, flush=True)
        for scene, titles in todo.items():
            for term in titles:
                if term in have: continue
                if a.of > 1 and _zlib.crc32(term.encode()) % a.of != a.shard: continue
                if (time.time() - t0) / 60 > a.budget_minutes:
                    print("budget reached", flush=True); break
                if tried >= a.limit:
                    break
                tried += 1
                p = None
                try:
                    hits = [h for h in search(term, token) if h.get("preview") and _fits(term, h) and _same_recording(h)][:a.versions]
                    if not hits:
                        missing += 1; report.append({"query": term, "scene": scene, "found": False})
                        out.write(json.dumps({"query": term, "scene": scene, "found": False, "pass": "v2"}) + "\n")
                        continue
                    for hit in hits:
                      if hit["track_id"] in have_ids: continue
                      have_ids.add(hit["track_id"])
                      fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
                      req = urllib.request.Request(hit["preview"], headers={"User-Agent": "signal-sonic-audio/canon"})
                      with urllib.request.urlopen(req, timeout=30) as r, open(p, "wb") as f: f.write(r.read())
                      y = FP.load_audio(p, max_seconds=150)   # the whole preview (it was the first 90 seconds)
                      hs = FP.hashes(y)
                      measures = {}
                      if LocalAnalyser is not None:
                          try:
                              fv = LocalAnalyser().analyse(p)
                              d = fv.__dict__ if hasattr(fv, "__dict__") else {}
                              measures = {k: (round(float(d[k]), 4) if isinstance(d.get(k), (int, float)) else None)
                                          for k in ("tempo", "drum_density", "drum_swing",
                                                    "bass_weight", "vocal_presence")}
                              measures["embedding"] = [round(float(x), 5) for x in (d.get("embedding") or [])]
                          except Exception as e:
                              import traceback
                              measures = {"error": f"{type(e).__name__}: {str(e)[:180]}",
                                          "where": traceback.format_exc().strip().split(chr(10))[-3][:160]}
                              print(f"    {term}: could not measure -> {measures['error']}", flush=True)
                              if not first_measure_error: first_measure_error.append(measures["error"])
                      if len(hs) < 50: raise ValueError("too few fingerprints")
                      H = np.array([h for h, _ in hs], dtype="<u4")
                      Fr = np.minimum(np.array([f for _, f in hs]), 0xFFFF).astype("<u2")
                      out.write(json.dumps({"query": term, "scene": scene, "found": True,
                                            "track_id": hit["track_id"], "name": hit["name"], "artists": hit["artists"],
                                            "preview": hit["preview"],
                                            "canon": True, "pass": "v2", "sample_start_ms": hit.get("sample_start_ms"), "sample_end_ms": hit.get("sample_end_ms"), "length_ms": hit.get("length_ms"),
                                          "measures": measures, "n": int(len(H)),
                                            "hashes": base64.b64encode(H.tobytes()).decode(),
                                            "frames": base64.b64encode(Fr.tobytes()).decode()}) + "\n")
                      found += 1
                      if measures.get("tempo"): measured_n += 1
                      if found % 20 == 0: out.flush(); print(f"  {found} fingerprinted, {missing} not on Beatport", flush=True)
                except Exception as e:
                    err += 1
                    if err <= 4: print(f"  {term}: {type(e).__name__}: {str(e)[:60]}", flush=True)
                    if len(first_errors) < 3: first_errors.append(f"{type(e).__name__}: {str(e)[:90]}")
                finally:
                    if p:
                        try: os.unlink(p)
                        except OSError: pass
    total = sum(len(v) for v in CANON.values())
    summ = {"canon_size": total, "tried": tried, "fingerprinted": found, "measured": measured_n, "not_on_beatport": missing,
            "errors": err, "hit_rate": round(found / max(1, found + missing), 2), "first_errors": first_errors,
            "analyser": ("not available: " + _ANALYSER_ERROR) if _ANALYSER_ERROR else "available", "first_measure_error": first_measure_error[:1]}
    print(json.dumps(summ, indent=1), flush=True)
    # the run logs cannot be read from where the pipeline is managed, so the outcome is an annotation on the run
    print("::notice title=canon::" + json.dumps(summ), flush=True)
    if tried and found + missing == 0:
        print("::error title=canon::every lookup failed, so nothing was recorded: " + "; ".join(first_errors), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
