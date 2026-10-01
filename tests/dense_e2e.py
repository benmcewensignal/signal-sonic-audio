"""Recognition in a room, end to end through the live site: room-degraded 25-second clips of classics (in the dense index)
and of recent records (in the main index only), fingerprinted as the phone does and posted to /api/listen. Writes
data/dense-e2e.json.   python -m tests.dense_e2e"""
import glob, json, os, random, sys, tempfile, urllib.request
import numpy as np
sys.path.insert(0, os.getcwd())
from audio import fingerprints as FP
from audio.rehearse import degrade
SITE = os.environ.get("SITE", "https://www.earlysignal.live").rstrip("/")
rng = np.random.default_rng(3); pick = random.Random(3)
def load(pattern, n):
    out, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            try: d = json.loads(line)
            except Exception: continue
            if d.get("found") is not False and d.get("preview") and d.get("track_id") and d["track_id"] not in seen: seen.add(d["track_id"]); out.append(d)
    pick.shuffle(out); return out[:n]
classics = load("out/fp-canon-v2-*.jsonl", 40)
# records that are not classics: from the site's DJ index, with previews, outside the classics set and sharing no title with a classic
import re as _re
canon_ids, canon_titles = set(), set()
for f in glob.glob("out/fp-canon*.jsonl"):
    for line in open(f):
        try:
            d_ = json.loads(line)
            if d_.get("track_id"): canon_ids.add(d_["track_id"])
            canon_titles.add(_re.sub(r"[^a-z0-9]+", " ", str(d_.get("name") or "").lower()).split(" (")[0].strip())
        except Exception: pass
def _get(path): return json.load(urllib.request.urlopen(urllib.request.Request(SITE + path, headers={"User-Agent": "sonic-dense-e2e"}), timeout=120))
PV = _get("/data/previews.json").get("u", {}); DI = _get("/data/dj-index.json"); DN = _get("/data/dj-names.json"); names_ = DN.get("t") or DN.get("n") or []
pool = [(t, (names_[k] if k < len(names_) else "")) for k, t in enumerate(DI["ids"]) if t in PV and t not in canon_ids]
pick.shuffle(pool)
canon_titles.discard("")
recent = [{"track_id": t, "preview": PV[t]} for t, nm in pool if not nm or _re.sub(r"[^a-z0-9]+", " ", str(nm).lower()).strip() not in canon_titles][:20]
print(f"outside records: {len(recent)}", flush=True)
cq = {}
for f in glob.glob("out/fp-canon*.jsonl"):
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("track_id"): cq[d["track_id"]] = d.get("query")
def ask(d):
    fd, p = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
    try:
        urllib.request.urlretrieve(d["preview"], p); y = FP.load_audio(p, max_seconds=None)
    finally: os.remove(p)
    L = int(FP.SR * 25); st = pick.randint(0, max(0, len(y) - L - 1)); clip = degrade(y[st:st + L], "room", rng).astype(np.float32)
    hs = [[int(h), int(f)] for h, f in FP.hashes(clip)]
    req = urllib.request.Request(SITE + "/api/listen", data=json.dumps({"hashes": hs[:20000]}).encode(), headers={"Content-Type": "application/json", "User-Agent": "sonic-dense-e2e"})
    return json.load(urllib.request.urlopen(req, timeout=60))
res = {"classics": [], "recent": []}
for kind, L_ in (("classics", classics), ("recent", recent)):
    for d in L_:
        try: r = ask(d)
        except Exception as e: res[kind].append({"id": d["track_id"], "error": type(e).__name__}); continue
        tr_ = r.get("track") or {}; tid = tr_.get("track_id")
        def _g(name, arts):   # the service's grouping: first artist's first word and the title before any bracket or feat.
            import re as _r
            n_ = lambda x: _r.sub(r"\bu\b", "you", _r.sub(r"[^a-z0-9]+", " ", str(x or "").replace("'", "").lower())).strip()
            return (n_((arts or [""])[0]).split(" ")[0] if arts else "") + "|" + n_(_r.split(r"\s*[\(\[]|\s+feat\.?\s+|\s+ft\.?\s+", str(name or ""), 1)[0])
        same = tid == d["track_id"] or (cq.get(tid) and cq.get(tid) == cq.get(d["track_id"])) or (tid and _g(tr_.get("name"), tr_.get("artists")) == _g(d.get("name"), d.get("artists")))
        res[kind].append({"id": d["track_id"], "found": bool(r.get("found")), "right": bool(r.get("found") and same), "wrong": bool(r.get("found") and not same), "via": r.get("via"), "dense": r.get("dense"), "local_hits": r.get("hashes_in_index"), "sent": r.get("hashes_sent"), "answer": [tid, tr_.get("name"), (tr_.get("artists") or [None])[0]] if r.get("found") else None, "asked": [d.get("name"), (d.get("artists") or [None])[0]]})
def rate(L_, k): return f"{sum(1 for x in L_ if x.get(k))} of {len(L_)}"
summ = {"classics_right": rate(res["classics"], "right"), "classics_via_dense": f"{sum(1 for x in res['classics'] if x.get('via') == 'classics')} of {len(res['classics'])}",
        "classics_wrong": rate(res["classics"], "wrong"), "recent_right": rate(res["recent"], "right"), "recent_wrong": rate(res["recent"], "wrong"),
        "errors": sum(1 for k in res for x in res[k] if x.get("error"))}
json.dump({"site": SITE, "summary": summ, "results": res}, open("data/dense-e2e.json", "w"), indent=1)
print("::notice title=dense e2e::" + json.dumps(summ))
