"""Name the stacks of a hung WASM run (hang_stress.sh): every report.*.json under <dir> (Node report: the main thread and
each worker = pthread), wasm-function[N] frames named from the snapshot's orbslam3_run.js.symbols (N:name) and
demangled with c++filt when it is installed. Usage: hang_stacks.py <dir> <orbslam3_run.js.symbols>"""
import sys, re, json, shutil, subprocess
from pathlib import Path

names = {}
for l in Path(sys.argv[2]).read_text().splitlines():
    i, _, n = l.partition(":")
    if i.isdigit(): names[int(i)] = n
if shutil.which("c++filt"):
    keys = list(names)
    out = subprocess.run(["c++filt"], input="\n".join(names[k] for k in keys), capture_output=True, text=True).stdout.splitlines()
    if len(out) == len(keys): names = dict(zip(keys, out))


def show(label, st):
    print(f"--- {label}: {st.get('message', '')}")
    for f in st.get("stack", []):
        m = re.search(r"wasm-function\[(\d+)\]", f)
        print("   ", names.get(int(m.group(1)), f) if m else f.strip())


for rep in sorted(Path(sys.argv[1]).rglob("report.*.json")):
    r = json.loads(rep.read_text())
    print(f"=== {rep}")
    show("main (Node)", r["javascriptStack"])
    for k, w in enumerate(r.get("workers", [])):
        show(f"worker {k}", w.get("javascriptStack", {}))
