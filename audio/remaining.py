"""How many records still lack a rhythm measurement. Prints a single number."""
import json, os, sqlite3, sys

have = set()
if os.path.isdir("out"):
    for f in os.listdir("out"):
        if f.startswith("rhythm-") and f.endswith(".jsonl"):
            for line in open(os.path.join("out", f)):
                try: have.add(json.loads(line)["track_id"])
                except Exception: pass
try:
    c = sqlite3.connect("sonic.db")
    n = sum(1 for (t,) in c.execute("select track_id from tracks where analyser_id='local' and track_id like 'bp:%'") if t not in have)
except Exception as e:
    print(0); sys.exit(0)
print(n)
