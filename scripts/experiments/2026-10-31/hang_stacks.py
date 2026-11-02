"""Name the stacks of a hung WASM run (hang_stress.sh): every report.*.json under <dir> (Node report: the main thread and
each worker = pthread; a worker blocked in a wait answers "Unavailable"), the wasm frames ($funcN or wasm-function[N],
N = function index with the imports) named from the snapshot's orbslam3_run.js.symbols (N:name, \\XX escapes).
Usage: hang_stacks.py <dir> <orbslam3_run.js.symbols>"""
import sys, re, json
from pathlib import Path

names = {}
for l in Path(sys.argv[2]).read_text().splitlines():
    i, _, n = l.partition(":")
    if i.isdigit(): names[int(i)] = re.sub(r"\\([0-9a-f]{2})", lambda m: chr(int(m.group(1), 16)), n)


def show(label, st):
    print(f"--- {label}: {st.get('message', '')}")
    for f in st.get("stack", []):
        m = re.search(r"\$func(\d+)|wasm-function\[(\d+)\]", f)
        print("   ", names.get(int(m.group(1) or m.group(2)), f) if m else f.strip())


for rep in sorted(Path(sys.argv[1]).rglob("report.*.json")):
    r = json.loads(rep.read_text())
    print(f"=== {rep}")
    show("main (Node)", r["javascriptStack"])
    print("    native:", " <- ".join(f.get("symbol", "?").split("(")[0].strip() for f in r.get("nativeStack", [])[:8]))
    for k, w in enumerate(r.get("workers", [])):
        show(f"worker {k}", w.get("javascriptStack", {}))
