#!/usr/bin/env python3
"""Fake a phone recording from an EuRoC sequence and upload it through the recorder's endpoint (web/serve.py POST
/api/record), to test the record -> replay chain without a phone: frames as JPEG, IMU rows, two taps, settings.yaml.
Usage: make_test_recording.py <euroc seq dir> <name> [--seconds 15] [--server http://127.0.0.1:8010]"""
import argparse, io, json, re, urllib.request
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("name")
ap.add_argument("--seconds", type=float, default=15); ap.add_argument("--server", default="http://127.0.0.1:8010")
ap.add_argument("--settings", default=str(Path(__file__).resolve().parents[1] / "baselines/orbslam3/settings/EuRoC_mono.yaml"))
ap.add_argument("--inertial", default=str(Path(__file__).resolve().parents[1] / "baselines/orbslam3/settings/EuRoC_mono_inertial.yaml"))
a = ap.parse_args()

def put(file, body):
    req = urllib.request.Request(f"{a.server}/api/record?name={a.name}&file={file}", data=body, method="POST")
    urllib.request.urlopen(req).read()

seq = Path(a.seq) / "mav0"
rows = [l.split(",") for l in (seq / "cam0/data.csv").read_text().splitlines() if l and not l.startswith("#")]
ns0 = int(rows[0][0]); rows = [r for r in rows if int(r[0]) - ns0 <= a.seconds * 1e9]
cam = ["#timestamp [ns],filename"]
for i, (ns, f) in enumerate(rows):
    buf = io.BytesIO(); Image.open(seq / "cam0/data" / f.strip()).convert("L").save(buf, "JPEG", quality=92)
    put(f"frame_{i}.jpg", buf.getvalue()); cam.append(f"{ns},frame_{i}.jpg")
imu = [l for l in (seq / "imu0/data.csv").read_text().splitlines() if l.startswith("#") or (l and ns0 - 1e8 <= int(l.split(",")[0]) <= int(rows[-1][0]))]
put("data.csv", ("\n".join(cam) + "\n").encode()); put("imu.csv", ("\n".join(imu) + "\n").encode())
put("touches.csv", f"#timestamp [ns],u [px],v [px]\n{ns0 + int(6e9)},376,330\n{ns0 + int(12e9)},376,330\n".encode())
tbc = re.search(r"IMU\.T_b_c1:\s*!!opencv-matrix.*?\]", Path(a.inertial).read_text(), re.S).group(0)
put("settings.yaml", (Path(a.settings).read_text().rstrip("\n") + "\n\n" + tbc + "\n").encode())
put("meta.json", json.dumps({"name": a.name, "notes": f"test: first {a.seconds} s of {Path(a.seq).name}, taps at 6 s and 12 s"}).encode())
print(f"uploaded {a.name}: {len(rows)} frames, {len(imu) - 1} IMU rows")
