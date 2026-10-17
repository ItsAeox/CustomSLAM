#!/usr/bin/env python3
"""Development server for the web SLAM demo and the dataset player.

- Sends the COOP/COEP headers that browsers require for SharedArrayBuffer (WASM threads).
- Serves only whitelisted directories under the research root:
    /web/...       this folder (pages, JS, built wasm in web/build/)
    /datasets/...  dataset sequences (read-only)
    /vocab/...     ORBvoc.bin
    /settings/...  generated ORB-SLAM3 settings files
- GET  /api/sequence?dataset=tumvie|euroc|tumvi&name=<seq>  -> JSON manifest (image URLs + timestamps, as orbslam3_run)
- GET  /api/sequences                          -> all runnable sequences with settings + GT URLs (viewer)
- GET  /api/imu?dataset=euroc|tumvi&name=<seq>  -> accelerometer on the camera timeline + camera<-IMU rotation
                                                  (ar.html dataset test mode: gravity instead of DeviceMotion)
- /orbsettings/... ORB-SLAM3's official settings (baselines/orbslam3/settings)
- /gt/...          ground truth, TUM format, camera frame (<root>/results/gt/<dataset>_<name>.txt)
- POST /api/result?name=<file>[&append=1]      -> saves (or appends) the body to <results>/browser/<file>
- POST /api/record?name=<rec>&file=<f>         -> recorder (record.html): stores <root>/datasets/phone/<rec>/ in the
                                                  EuRoC layout (frame_<n>.jpg, data.csv, imu.csv, touches.csv, ...)
- Optional HTTPS (for phones on the LAN): --cert cert.pem --key key.pem --host 0.0.0.0

Usage: python serve.py [--root H:/Research] [--port 8000] [--host localhost] [--cert C --key K]
"""
import argparse
import json
import re
import socket
import ssl
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

WEB_DIR = Path(__file__).resolve().parent


def tumvie_manifest(seq: Path) -> dict:
    """Same frames/timestamps as vio_run / orbslam3_run: image index from the file name,
    t = image_timestamps_left[idx] * 1e-6 + cam_time_offset_ns * 1e-9, rebased to the first timestamp."""
    calib = json.loads((seq / "camera-calibration.json").read_text())
    calib = calib.get("value0", calib)
    off = float(calib.get("cam_time_offset_ns", 0)) * 1e-9
    ts = [float(l) * 1e-6 + off for l in (seq / "left_images" / "image_timestamps_left.txt").read_text().splitlines()
          if l.strip() and not l.startswith("#")]
    t0 = ts[0]
    frames = []
    for f in sorted((seq / "left_images").glob("*.jpg")):
        k = int(re.sub(r"\D", "", f.stem))
        if 0 <= k < len(ts):
            frames.append({"idx": k, "t": ts[k] - t0, "url": f"/datasets/{seq.name}/left_images/{f.name}"})
    return {"dataset": "tumvie", "name": seq.name, "tBase": t0, "clahe": True, "frames": frames,
            "missing": len(ts) - len(frames)}


def asl_manifest(seq: Path, dataset: str, url_prefix: str) -> dict:
    """EuRoC / TUM-VI EuRoC-format export: frames from mav0/cam0/data.csv, t = (ns - ns0) * 1e-9 (same as orbslam3_run).
    CLAHE for TUM-VI only, as ORB-SLAM3's mono_tum_vi / mono_euroc examples."""
    rows = []
    for l in (seq / "mav0" / "cam0" / "data.csv").read_text().splitlines():
        if not l.strip() or l.startswith("#"):
            continue
        ns, name = l.split(",", 1)
        rows.append((int(ns), name.strip()))
    ns0 = rows[0][0]
    frames = [{"idx": i, "t": (ns - ns0) * 1e-9, "url": f"{url_prefix}/{seq.name}/mav0/cam0/data/{name}"}
              for i, (ns, name) in enumerate(rows) if (seq / "mav0" / "cam0" / "data" / name).exists()]
    return {"dataset": dataset, "name": seq.name, "tBase": ns0 * 1e-9, "clahe": dataset == "tumvi",
            "frames": frames, "missing": len(rows) - len(frames)}


# Camera -> IMU rotation for TUM-VI (512x512, cam0): IMU.T_b_c1 from ORB-SLAM3's
# Examples/Monocular-Inertial/TUM-VI.yaml (the IMU is the body frame). EuRoC: read from mav0/cam0/sensor.yaml (T_BS).
TUMVI_R_BC = [[-0.9995250378696743, 0.0075019185074052044, -0.02989013031643309],
              [0.029615343885863205, -0.03439736061393144, -0.998969345370175],
              [-0.008522328211654736, -0.9993800792498829, 0.03415885127385616]]


PHONE_R_BC = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]

# recorder uploads: allowed file names -> path inside datasets/phone/<rec>/
RECORD_FILES = {"data.csv": "mav0/cam0/data.csv", "imu.csv": "mav0/imu0/data.csv", "touches.csv": "touches.csv",
                "gyro_gs.csv": "gyro_gs.csv", "accel_gs.csv": "accel_gs.csv", "meta.json": "meta.json",
                "settings.yaml": "settings.yaml"}


def record_path(name: str) -> str | None:
    if re.fullmatch(r"frame_\d{1,7}\.jpg", name):
        return f"mav0/cam0/data/{name}"
    return RECORD_FILES.get(name)


def imu_raw(seq: Path) -> dict:
    """Every IMU sample (EuRoC-format imu0, IMU body frame) on the camera timeline: t (s), acc (m/s^2), gyro (rad/s).
    For mono-inertial runs in ar.html's dataset test mode (?inertial=1)."""
    ns0 = None
    for l in (seq / "mav0" / "cam0" / "data.csv").read_text().splitlines():
        if l.strip() and not l.startswith("#"):
            ns0 = int(l.split(",", 1)[0])
            break
    t, acc, gyro = [], [], []
    for l in (seq / "mav0" / "imu0" / "data.csv").read_text().splitlines():
        if not l.strip() or l.startswith("#"):
            continue
        f = l.split(",")
        t.append(round((int(f[0]) - ns0) * 1e-9, 6))
        gyro.append([float(f[1]), float(f[2]), float(f[3])])
        acc.append([float(f[4]), float(f[5]), float(f[6])])
    return {"t": t, "acc": acc, "gyro": gyro}


def imu_manifest(seq: Path, dataset: str, hz: float = 50.0) -> dict:
    """Accelerometer (specific force, m/s^2, IMU frame; at rest it points up) averaged into 1/hz bins, with t on the
    camera timeline (seconds since the first cam0 frame, as asl_manifest), and R_cb (camera <- IMU, row-major)."""
    ns0 = None
    for l in (seq / "mav0" / "cam0" / "data.csv").read_text().splitlines():
        if l.strip() and not l.startswith("#"):
            ns0 = int(l.split(",", 1)[0])
            break
    if dataset == "phone":  # recorder (record.js): DeviceMotion device axes, rear camera portrait: R_bc = diag(1, -1, -1)
        r_bc = PHONE_R_BC
    elif dataset == "euroc":  # EuRoC: imu0 is the body frame; cam0 T_BS maps camera -> body
        txt = (seq / "mav0" / "cam0" / "sensor.yaml").read_text()
        m = re.search(r"T_BS:.*?data:\s*\[([^\]]*)\]", txt, re.S)
        v = [float(x) for x in m.group(1).replace("\n", " ").split(",")]
        r_bc = [v[0:3], v[4:7], v[8:11]]
    else:
        r_bc = TUMVI_R_BC
    r_cb = [[r_bc[j][i] for j in range(3)] for i in range(3)]
    bins = {}
    for l in (seq / "mav0" / "imu0" / "data.csv").read_text().splitlines():
        if not l.strip() or l.startswith("#"):
            continue
        f = l.split(",")
        t = (int(f[0]) - ns0) * 1e-9
        b = bins.setdefault(int(t * hz // 1), [0.0, 0.0, 0.0, 0.0, 0])
        b[0] += t; b[1] += float(f[4]); b[2] += float(f[5]); b[3] += float(f[6]); b[4] += 1
    ts, acc = [], []
    for k in sorted(bins):
        b = bins[k]
        ts.append(round(b[0] / b[4], 4))
        acc.append([round(b[1] / b[4], 4), round(b[2] / b[4], 4), round(b[3] / b[4], 4)])
    return {"t": ts, "acc": acc, "R_cb": r_cb, "hz": hz}


def list_sequences(root: Path) -> list:
    """All sequences the player/viewer can run, with their ORB-SLAM3 settings and ground truth (if available).
    GT files: <root>/results/gt/<dataset>_<name>.txt (TUM format, camera frame; see eval/gt_*.py)."""
    out = []
    ds_root = root / "datasets"
    gt_dir = root / "results" / "gt"

    def add(dataset, name, settings, clahe):
        gt = gt_dir / f"{dataset}_{name}.txt"
        out.append({"dataset": dataset, "name": name, "settings": settings, "clahe": clahe,
                    "gt": f"/gt/{gt.name}" if gt.exists() else None})

    for sub, settings in (("euroc", "/orbsettings/EuRoC_mono.yaml"), ("tumvi", "/orbsettings/TUM-VI_mono.yaml")):
        for seq in sorted((ds_root / sub).glob("*")):
            if (seq / "mav0" / "cam0" / "data.csv").exists():
                add(sub, seq.name, settings, sub == "tumvi")
    for seq in sorted((ds_root / "phone").glob("*")):  # recorder sequences (settings written by record.js)
        if (seq / "mav0" / "cam0" / "data.csv").exists() and (seq / "settings.yaml").exists():
            add("phone", seq.name, f"/datasets/phone/{seq.name}/settings.yaml", False)
    for seq in sorted(ds_root.glob("*")):
        if (seq / "camera-calibration.json").exists() and (seq / "left_images").is_dir():
            s = root / "results" / "scratch" / "settings" / f"tumvie_{seq.name}.yaml"
            if s.exists():
                add("tumvie", seq.name, f"/settings/{s.name}", True)
    return out


class Handler(SimpleHTTPRequestHandler):
    root: Path = Path(".")
    results: Path = Path(".")

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def translate_path(self, path):
        p = unquote(urlparse(path).path)
        mounts = {"/web/": WEB_DIR, "/datasets/": self.root / "datasets", "/vocab/": self.root / "vocab",
                  "/settings/": self.root / "results" / "scratch" / "settings",
                  "/orbsettings/": WEB_DIR.parent / "baselines" / "orbslam3" / "settings",
                  "/gt/": self.root / "results" / "gt"}
        for prefix, base in mounts.items():
            if p.startswith(prefix):
                target = (base / p[len(prefix):]).resolve()
                if base.resolve() in target.parents or target == base.resolve():
                    return str(target)
        return str(WEB_DIR / "__forbidden__")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self.send_response(302)
            self.send_header("Location", "/web/index.html")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if u.path == "/api/sequences":
            return self.send_json(list_sequences(self.root))
        if u.path == "/api/sequence":
            q = parse_qs(u.query)
            name = q.get("name", [""])[0]
            dataset = q.get("dataset", ["tumvie"])[0]
            sub = {"euroc": "euroc", "tumvi": "tumvi", "phone": "phone"}.get(dataset, "")
            base = (self.root / "datasets" / sub).resolve()
            seq = (base / name).resolve()
            if not name or base not in seq.parents or not seq.is_dir():
                return self.send_error(404, "unknown sequence")
            manifest = (asl_manifest(seq, dataset, f"/datasets/{sub}") if sub else tumvie_manifest(seq))
            return self.send_json(manifest)
        if u.path == "/api/imu":
            q = parse_qs(u.query)
            name = q.get("name", [""])[0]
            dataset = q.get("dataset", [""])[0]
            sub = {"euroc": "euroc", "tumvi": "tumvi", "phone": "phone"}.get(dataset, "")
            base = (self.root / "datasets" / sub).resolve()
            seq = (base / name).resolve()
            if not sub or not name or base not in seq.parents or not (seq / "mav0" / "imu0" / "data.csv").exists():
                return self.send_error(404, "no IMU for this sequence")
            return self.send_json(imu_raw(seq) if q.get("raw", ["0"])[0] == "1" else imu_manifest(seq, dataset))
        return super().do_GET()

    def send_json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/record":
            return self.record(parse_qs(u.query))
        if u.path != "/api/result":
            return self.send_error(404)
        q = parse_qs(u.query)
        name = q.get("name", [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            return self.send_error(400, "bad name")
        n = int(self.headers.get("Content-Length", 0))
        out = self.results / "browser" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        body = self.rfile.read(n)
        if q.get("append", ["0"])[0] == "1":
            with out.open("ab") as f:
                f.write(body)
        else:
            out.write_bytes(body)
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()
        print(f"saved {out}", file=sys.stderr)


    def record(self, q):
        name, file = q.get("name", [""])[0], q.get("file", [""])[0]
        rel = record_path(file)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or not rel:
            return self.send_error(400, "bad recording or file name")
        out = self.root / "datasets" / "phone" / name / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        n = int(self.headers.get("Content-Length", 0))
        out.write_bytes(self.rfile.read(n))
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()
        if not file.startswith("frame_"):
            print(f"recording {name}: saved {rel}", file=sys.stderr)

    def log_message(self, fmt, *args):  # quiet: one line per recorded frame would flood the log
        if "/api/record" in (self.path or "") and " 204 " in (fmt % args):
            return
        super().log_message(fmt, *args)


class V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(WEB_DIR.parents[1]), help="research root (contains datasets/, vocab/, results/)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="localhost",
                    help="'localhost' (default) listens on both 127.0.0.1 and ::1; anything else is bound as given")
    ap.add_argument("--cert")
    ap.add_argument("--key")
    a = ap.parse_args()
    Handler.root = Path(a.root)
    Handler.results = Path(a.root) / "results"
    Handler.extensions_map.update({".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript"})
    # Browsers resolve "localhost" to ::1 first. With an IPv4-only server, Windows takes 0.1-0.3 s to refuse every
    # new IPv6 connection before falling back to IPv4, which capped the dataset player at ~5 fps. Hence both loopbacks,
    # plus HTTP/1.1 keep-alive so connections are reused.
    Handler.protocol_version = "HTTP/1.1"
    if a.host == "localhost":
        servers = [ThreadingHTTPServer(("127.0.0.1", a.port), Handler)]
        try:
            servers.append(V6Server(("::1", a.port), Handler))
        except OSError as e:
            print(f"IPv6 loopback not available ({e}); IPv4 only", file=sys.stderr)
    else:
        servers = [(V6Server if ":" in a.host else ThreadingHTTPServer)((a.host, a.port), Handler)]
    scheme = "http"
    if a.cert and a.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.cert, a.key)
        for httpd in servers:
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print(f"serving {scheme}://{a.host}:{a.port}/  (root {a.root})", file=sys.stderr)
    for httpd in servers[1:]:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
