"""Name the stacks of a hung WASM run (hang_stress.sh), for every run folder under <dir>:
- gdb.txt (hang_gdb.py: every thread's native + JIT frames) with perf.map (V8's map of JIT code, node --perf-basic-prof):
  a JIT pc is looked up as pc-1 in the latest perf-map entry that holds it (code addresses get reused); wasm functions
  (wasm-function[N]-N-liftoff/turbofan) are named from the snapshot's orbslam3_run.js.symbols;
- report.*.json (Node report, SIGUSR2): only written when Node's main thread gets back to its event loop; frames $funcN
  or wasm-function[N] named the same way.
symbols: N:name, N = function index with the imports, names with \\XX escapes.
Usage: hang_stacks.py <dir> <orbslam3_run.js.symbols>"""
import sys, re, json
from pathlib import Path

names = {}
for l in Path(sys.argv[2]).read_text().splitlines():
    i, _, n = l.partition(":")
    if i.isdigit(): names[int(i)] = re.sub(r"\\([0-9a-f]{2})", lambda m: chr(int(m.group(1), 16)), n)


def wasm_name(s):
    m = re.search(r"\$func(\d+)|wasm-function\[(\d+)\]", s)
    return names.get(int(m.group(1) or m.group(2)), s) if m else s


def show(label, st):
    print(f"--- {label}: {st.get('message', '')}")
    for f in st.get("stack", []):
        print("   ", wasm_name(f).strip())


def gdb_stacks(gdbtxt):
    perf = []
    pm = gdbtxt.parent / "perf.map"
    if pm.exists():
        for l in pm.read_text(errors="replace").splitlines():
            p = l.split(" ", 2)
            if len(p) == 3:
                a = int(p[0], 16)
                perf.append((a, a + int(p[1], 16), p[2]))

    def jit(pc):
        hit = None
        for a, b, n in perf:  # the latest entry wins
            if a <= pc - 1 < b: hit = n
        return wasm_name(hit) if hit else "?"

    print(f"=== {gdbtxt}")
    for l in gdbtxt.read_text(errors="replace").splitlines():
        m = re.match(r"\s+(0x[0-9a-f]+) (.*)", l)
        if m and m.group(2) == "?":
            print(f"    {jit(int(m.group(1), 16))}")
        elif m:
            print(f"    {m.group(2)[:120]}")
        elif l.startswith("==="):
            print(l)


root = Path(sys.argv[1])
for g in sorted(root.rglob("gdb.txt")):
    if "=== thread" in g.read_text(errors="replace"):
        gdb_stacks(g)
for rep in sorted(root.rglob("report.*.json")):
    r = json.loads(rep.read_text())
    print(f"=== {rep}")
    show("main (Node)", r["javascriptStack"])
    print("    native:", " <- ".join(f.get("symbol", "?").split("(")[0].strip() for f in r.get("nativeStack", [])[:8]))
    for k, w in enumerate(r.get("workers", [])):
        show(f"worker {k}", w.get("javascriptStack", {}))
