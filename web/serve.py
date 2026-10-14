#!/usr/bin/env python3
"""Development server for the web SLAM demo and the dataset player.

- Sends the COOP/COEP headers that browsers require for SharedArrayBuffer (WASM threads).
- Serves only whitelisted directories under the research root:
    /web/...       this folder (pages, JS, built wasm in web/build/)
    /datasets/...  dataset sequences (read-only)
    /vocab/...     ORBvoc.bin
    /settings/...  generated ORB-SLAM3 settings files
- GET  /api/sequence?dataset=tumvie|euroc|tumvi&name=<seq>  -> JSON manifest (image URLs + timestamps, as orbslam3_run)
- /orbsettings/... ORB-SLAM3's official settings (baselines/orbslam3/settings)
- POST /api/result?name=<file>                 -> saves the body to <results>/browser/<file>
- Optional HTTPS (for phones on the LAN): --cert cert.pem --key key.pem --host 0.0.0.0

Usage: python serve.py [--root H:/Research] [--port 8000] [--host 127.0.0.1] [--cert C --key K]
"""
import argparse
import json
import re
import ssl
import sys
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
                  "/orbsettings/": WEB_DIR.parent / "baselines" / "orbslam3" / "settings"}
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
            self.end_headers()
            return
        if u.path == "/api/sequence":
            q = parse_qs(u.query)
            name = q.get("name", [""])[0]
            dataset = q.get("dataset", ["tumvie"])[0]
            sub = {"euroc": "euroc", "tumvi": "tumvi"}.get(dataset, "")
            base = (self.root / "datasets" / sub).resolve()
            seq = (base / name).resolve()
            if not name or base not in seq.parents or not seq.is_dir():
                return self.send_error(404, "unknown sequence")
            manifest = (asl_manifest(seq, dataset, f"/datasets/{sub}") if sub else tumvie_manifest(seq))
            body = json.dumps(manifest).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/result":
            return self.send_error(404)
        name = parse_qs(u.query).get("name", [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            return self.send_error(400, "bad name")
        n = int(self.headers.get("Content-Length", 0))
        out = self.results / "browser" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.rfile.read(n))
        self.send_response(204)
        self.end_headers()
        print(f"saved {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(WEB_DIR.parents[1]), help="research root (contains datasets/, vocab/, results/)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cert")
    ap.add_argument("--key")
    a = ap.parse_args()
    Handler.root = Path(a.root)
    Handler.results = Path(a.root) / "results"
    Handler.extensions_map.update({".wasm": "application/wasm", ".js": "text/javascript", ".mjs": "text/javascript"})
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    scheme = "http"
    if a.cert and a.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(a.cert, a.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print(f"serving {scheme}://{a.host}:{a.port}/  (root {a.root})", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
