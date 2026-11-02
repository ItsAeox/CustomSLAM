"""V8 CPU profiles (node --cpu-prof, one .cpuprofile per thread) of a WASM run: per profile, the functions with the most
self time and the most total time (in the stack), wasm frames ($funcN / wasm-function[N]) named from the build's
orbslam3_run.js.symbols (N:name with the imports, \\XX escapes). Threads that mostly wait show the wait function on top.
Usage: cpuprof_top.py <cpuprof dir> <orbslam3_run.js.symbols> [top=25]"""
import sys, re, json
from pathlib import Path
from collections import defaultdict

names = {}
for l in Path(sys.argv[2]).read_text().splitlines():
    i, _, n = l.partition(":")
    if i.isdigit(): names[int(i)] = re.sub(r"\\([0-9a-f]{2})", lambda m: chr(int(m.group(1), 16)), n)
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 25


def fname(cf):
    f = cf.get("functionName") or "(anonymous)"
    m = re.search(r"\$func(\d+)|wasm-function\[(\d+)\]", f)
    return names.get(int(m.group(1) or m.group(2)), f) if m else f


for p in sorted(Path(sys.argv[1]).glob("*.cpuprofile")):
    prof = json.loads(p.read_text())
    nodes = {n["id"]: n for n in prof["nodes"]}
    parent = {c: n["id"] for n in prof["nodes"] for c in n.get("children", [])}
    self_t, tot_t = defaultdict(float), defaultdict(float)
    for sid, dt in zip(prof["samples"], prof["timeDeltas"]):
        dt = max(dt, 0) / 1000.0
        self_t[fname(nodes[sid]["callFrame"])] += dt
        seen, k = set(), sid
        while k is not None:
            n = fname(nodes[k]["callFrame"])
            if n not in seen:
                seen.add(n); tot_t[n] += dt
            k = parent.get(k)
    total = sum(self_t.values())
    print(f"## {p.name}: {total / 1000:.1f} s sampled\n")
    print("| Self ms | Self % | Function | | Total ms | Total % | Function |")
    print("|---|---|---|---|---|---|---|")
    a = sorted(self_t.items(), key=lambda x: -x[1])[:TOP]
    b = sorted(tot_t.items(), key=lambda x: -x[1])[:TOP]
    for (n1, v1), (n2, v2) in zip(a, b):
        print(f"| {v1:.0f} | {100 * v1 / total:.1f} | {n1[:90]} | | {v2:.0f} | {100 * v2 / total:.1f} | {n2[:90]} |")
    print()
