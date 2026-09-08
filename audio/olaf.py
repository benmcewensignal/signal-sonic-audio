"""Olaf adapter: store previews, query a mix, parse the hits.

Olaf (Joren Six, Ghent) is a C fingerprinter built for matching under the conditions a
DJ set imposes: time stretch, pitch shift, other audio underneath. Ours is not. This
wraps its binary so the set layer can be benchmarked on it, and if it wins, run on it.

Audio in: any file librosa can read. Olaf wants 16 kHz mono float32 raw; we convert.
Results out: (track_id, seconds into the query, match score) per hit.

  python -m audio.olaf build                       compile the binary into bin/
  python -m audio.olaf store  <id> <audio>         index one record under an id
  python -m audio.olaf query  <audio>              hits for a mix
  python -m audio.olaf bench  --db sonic.db        precision/recall against NTS truth
"""
import argparse, json, os, re, sqlite3, subprocess, sys, tempfile, urllib.request
import numpy as np

BIN = os.path.join(os.path.dirname(__file__), "..", "bin", "olaf_core")
SR = 16000


def build():
    """Clone and compile Olaf with gcc; zig is not needed for the core."""
    if os.path.exists(BIN): return BIN
    d = tempfile.mkdtemp()
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/JorenSix/Olaf.git", d], check=True)
    subprocess.run(["make", "compile_core"], cwd=d, check=True, capture_output=True)
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    subprocess.run(["cp", os.path.join(d, "bin", "olaf_core"), BIN], check=True)
    os.makedirs(os.path.expanduser("~/.olaf/db"), exist_ok=True)
    return BIN


def to_raw(audio_path):
    """librosa cannot decode webm or m4a, which is what yt-dlp returns for most mixes:
    transcode with ffmpeg first, as the main pipeline does."""
    import librosa
    if os.path.splitext(audio_path)[1].lower() not in (".wav", ".flac", ".aiff", ".aif"):
        wav = os.path.splitext(audio_path)[0] + ".olaf.wav"
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", audio_path, "-ac", "1", "-ar", str(SR), wav],
                           capture_output=True, text=True)
        if os.path.exists(wav) and os.path.getsize(wav) > 1000: audio_path = wav
        else: raise RuntimeError("ffmpeg: " + (r.stderr.strip().splitlines()[-1][:160] if r.stderr.strip() else "no output"))
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    fd, raw = tempfile.mkstemp(suffix=".raw"); os.close(fd)
    y.astype(np.float32).tofile(raw)
    return raw, len(y) / SR


def _run(*args):
    r = subprocess.run([build(), *args], capture_output=True, text=True)
    return r.stdout + r.stderr


def store(track_id, audio_path):
    raw, dur = to_raw(audio_path)
    # olaf derives its internal id from the filename, so give it the track id as the name
    named = os.path.join(tempfile.mkdtemp(), f"{track_id.replace(':', '_')}.raw")
    os.rename(raw, named)
    out = _run("store", named, named)
    os.unlink(named)
    m = re.search(r"Stored (\d+) fp", out)
    return int(m.group(1)) if m else 0


def query(audio_path):
    raw, dur = to_raw(audio_path)
    out = _run("query", raw, raw)
    os.unlink(raw)
    hits = []
    for line in out.splitlines():
        # olaf prints: count, query_start, query_stop, ref_path, ref_id, ref_start, ref_stop
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7 and parts[0].isdigit() and int(parts[0]) > 0 and parts[3]:
            name = os.path.basename(parts[3]).replace(".raw", "").replace("_", ":", 1)
            hits.append({"track_id": name, "count": int(parts[0]), "q_start": float(parts[1]), "q_stop": float(parts[2])})
    return hits, dur


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("build")
    s = sub.add_parser("store"); s.add_argument("track_id"); s.add_argument("audio")
    q = sub.add_parser("query"); q.add_argument("audio")
    a = ap.parse_args()
    if a.cmd == "build": print(build())
    elif a.cmd == "store": print(store(a.track_id, a.audio), "fingerprints")
    elif a.cmd == "query":
        hits, dur = query(a.audio); print(json.dumps({"duration_s": round(dur, 1), "hits": hits}, indent=1))


if __name__ == "__main__":
    main()
