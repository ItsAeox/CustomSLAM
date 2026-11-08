"""phone_bench_1107: the benchmark arms of main_1103 / ext_1103 in the Pixel 9's Chrome (web/viewer.html), user 7 Nov:
"run the EuRoC and the synthetic sets on the phone, just like the desktop".

Page module web/build-phone (md5 682eaccb...): the frozen 84c1f17 SLAM sources + one read-only binding entry, the
current map id (build_phone_web_1107.sh; reverting it rebuilds the frozen module, md5 c4608c77..., byte for byte), same settings files as the PC batches
(main_1103/settings, ext_1103/settings, ORB-SLAM3's EuRoC_mono.yaml / the synthetic settings.yaml for stock), same
pacing as orbslam3_run --rt-wall (viewer pacing=rt: every frame, never before its time, no drops), the dataset gyro fed
as orbslam3_run --gyro for next / beta, no drawing (viz=0; frames decoded on the page's main thread two ahead, as the
runner's reader thread). Frames are stored on the phone per sequence (Cache API, cache=1); every run first deletes
other sequences' frames (clearcache=1) and the sequence's last run deletes its own (uncache=1): the phone has ~6 GB free, EuRoC is ~9.7 GB.

Phases (each sequence block: fill the cache, every arm x seed in shuffled order, free the cache):
  1 EuRoC (11 sequences) seeds 1-2; 2 synthetic core set (42) seeds 1-2; 3 EuRoC seeds 3-5.
Results: results/browser/ph1107_<arm>_<seq>_viewer_seed<k>_{final.txt,live.txt,timing.csv,info.json};
log: results/phone_1107/runs.log (one line per run + battery / temperature).
Usage: python phone_bench_1107.py --serial 192.168.1.7:32781 [--phases 1,2,3] [--dry]"""
import argparse, json, random, re, subprocess, time
from pathlib import Path

ADB = r"C:\Users\Aeox\AppData\Local\Android\Sdk\platform-tools\adb.exe"
R = Path("H:/Research")
OUT = R / "results/browser"
LOG = R / "results/phone_1107/runs.log"
EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium",
         "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
SYNTH = [f"{k}{i}" for k in ("ar", "gesture", "orbit", "rotate", "shake", "slide", "walk") for i in range(6)]
ARMS = ["stock", "fast", "next", "beta"]
GYRO = {"next", "beta"}
FRESH = True  # no run launched yet by this process
DRY = False  # --dry: print the URLs, write nothing to the run log


def settings(arm, ds, seq):
    if ds == "euroc":
        return "/orbsettings/EuRoC_mono.yaml" if arm == "stock" else f"/settings/phone_1107/EuRoC_mono_page-{arm}.yaml"
    return f"/datasets/synth/{seq}/settings.yaml" if arm == "stock" else f"/settings/phone_1107/synth/{seq}/{seq}_page-{arm}.yaml"


def adb(serial, *a, timeout=30):
    return subprocess.run([ADB, "-s", serial, *a], capture_output=True, text=True, timeout=timeout).stdout


def battery(serial):
    s = adb(serial, "shell", "dumpsys", "battery")
    g = lambda k: (re.search(rf"{k}: (\S+)", s) or [None, None])[1]
    return {"level": int(g("level") or -1), "temp_c": int(g("temperature") or 0) / 10,
            "charging": any(g(k) == "true" for k in ("AC powered", "USB powered", "Wireless powered"))}


def log(line):
    if DRY:
        print(line, flush=True); return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    print(line, flush=True)


PAGE_JS = """JSON.stringify({status: (document.getElementById('status') || {}).textContent,
  frame: (document.getElementById('sFrame') || {}).textContent, hidden: document.hidden, isolated: self.crossOriginIsolated,
  log: ((document.getElementById('log') || {}).textContent || '').slice(-800)})"""


def page_state(serial):  # read-only look at the benchmark tab when a run times out (Chrome remote debugging over adb)
    try:
        import urllib.request, websocket
        adb(serial, "forward", "tcp:9333", "localabstract:chrome_devtools_remote")
        tabs = json.load(urllib.request.urlopen("http://localhost:9333/json", timeout=10))
        tab = next((t for t in tabs if "localhost:8010/web/viewer.html" in t.get("url", "")), None)
        if tab is None:
            return "no viewer tab"
        ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=15, suppress_origin=True)
        ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {"expression": PAGE_JS, "returnByValue": True}}))
        r = json.loads(ws.recv()); ws.close()
        return r.get("result", {}).get("result", {}).get("value") or json.dumps(r)[:500]
    except Exception as e:
        return f"page state unavailable: {e}"


def mem_avail(serial):
    m = re.search(r"MemAvailable:\s+(\d+)", adb(serial, "shell", "cat", "/proc/meminfo"))
    return f"{int(m[1]) // 1024} MB" if m else "?"


def wait_ok(serial):  # battery and temperature guard: pause, never kill a run
    while True:
        b = battery(serial)
        if b["level"] >= 0 and b["level"] < 20 and not b["charging"]:
            log(f"PAUSE battery {b}"); time.sleep(300); continue
        if b["temp_c"] >= 45:
            log(f"PAUSE hot {b}"); time.sleep(180); continue
        return b


def run(serial, arm, ds, seq, seed, first, last, seq_s, dry):
    tag = f"ph1107_{arm}"
    info = OUT / f"{tag}_{seq}_viewer_seed{seed}_info.json"
    if info.exists():
        if json.loads(info.read_text()).get("hidden_n", 0) == 0:
            log(f"skip {tag} {seq} seed{seed} (done)"); return True
        for f in OUT.glob(f"{tag}_{seq}_viewer_seed{seed}_*"):  # finished while the page was hidden: repeat it
            f.unlink()
        log(f"HIDDEN {tag} {seq} seed{seed}: saved while the page was hidden, files removed, repeated")
    q = (f"seq={ds}/{seq}&pacing=rt&seed={seed}&settings={settings(arm, ds, seq)}&viz=0&tag={tag}&save=1&cache=1&clearcache=1&build=phone"
         f"{'&gyro=1' if arm in GYRO else ''}{'&uncache=1' if last else ''}&autorun=1")
    url = f"http://localhost:8010/web/viewer.html?{q}"
    if dry:
        print(url); return True
    b = wait_ok(serial)
    t0 = time.time()
    # fresh page per run (one tab, reused: application_id); the first run of a block also fills the frame cache
    adb(serial, "shell", f"am start -a android.intent.action.VIEW -n com.android.chrome/com.google.android.apps.chrome.Main "
                         f"--es com.android.browser.application_id webvio_bench -d '{url}'")
    # the first run of a block (or of this driver process) may also fill the frame cache (EuRoC: up to 1.3 GB over
    # Wi-Fi); later runs find the frames on the phone (8 Nov: two hung runs cost 32 min each with the long limit)
    global FRESH
    limit = 3 * seq_s + (1800 if first or FRESH else 300)
    FRESH = False
    while time.time() - t0 < limit and not info.exists():
        time.sleep(5)
    ok = info.exists()
    extra, word = "", "ok" if ok else "TIMEOUT"
    if not ok:
        extra = f"page {page_state(serial)} | MemAvailable {mem_avail(serial)}"
    if ok:
        j = json.loads(info.read_text())
        extra = f"rate {j['effective_rate']:.3f} tracked {j['tracked']}/{j['processed']} maps {j['maps']}"
        if j.get("hidden_n", 0) > 0:  # Chrome was not in front during the run (throttled): discard and repeat it
            extra += f" | page hidden {j['hidden_n']}x {j.get('hidden_s', 0):.0f} s: files removed, repeated"
            for f in OUT.glob(f"{tag}_{seq}_viewer_seed{seed}_*"):
                f.unlink()
            ok, word = False, "HIDDEN"
    log(f"{word} {tag} {seq} seed{seed} {time.time() - t0:.0f} s {extra} | battery {b['level']}% "
        f"{'charging' if b['charging'] else 'not charging'} {b['temp_c']} C")
    return ok


def seq_seconds(ds, seq):
    csv = R / f"datasets/{'euroc' if ds == 'euroc' else 'synth'}/{seq}/mav0/cam0/data.csv"
    ts = [int(l.split(",")[0]) for l in csv.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return (ts[-1] - ts[0]) * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True)
    ap.add_argument("--phases", default="1,2,3")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    global DRY
    DRY = a.dry
    rng = random.Random(11071)
    phases = {"1": ("euroc", EUROC, [1, 2]), "2": ("synth", SYNTH, [1, 2]), "3": ("euroc", EUROC, [3, 4, 5])}
    log(f"=== start phases {a.phases}")
    for ph in a.phases.split(","):
        ds, seqs, seeds = phases[ph]
        order = seqs[:]; rng.shuffle(order)
        log(f"--- phase {ph}: {ds} {len(order)} sequences, seeds {seeds}")
        for seq in order:
            jobs = [(arm, s) for s in seeds for arm in ARMS]
            rng.shuffle(jobs)
            ss = seq_seconds(ds, seq)
            for k, (arm, seed) in enumerate(jobs):
                for attempt in (1, 2):
                    if run(a.serial, arm, ds, seq, seed, k == 0, k == len(jobs) - 1, ss, a.dry):
                        break
    log("=== done")


if __name__ == "__main__":
    main()
