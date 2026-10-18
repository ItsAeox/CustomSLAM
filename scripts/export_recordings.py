#!/usr/bin/env python3
"""Phone recordings (datasets/phone/rec_*) as static files for GitHub Pages, so ar.html can replay them without the
local server: <out>/recordings/<name>/ with the frames as gray JPEG (quality 80: the tracker uses gray; ~22 KB instead
of ~36 KB per frame), manifest.json / imu.json / imu_raw.json exactly as web/serve.py serves /api/sequence and /api/imu
(frame URLs rewritten to the static files), settings.yaml and touches.csv; and <out>/recordings/index.json (one entry
per recording, as /api/sequences, with its label from labels.txt) for index.html and ar.html.

  export_recordings.py <out dir, e.g. docs> <name> [<name> ...]
"""
import io
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "web"))
import serve  # noqa: E402  (asl_manifest, imu_manifest, imu_raw)

PHONE = REPO.parent / "datasets" / "phone"


def labels():
    out = {}
    p = PHONE / "labels.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                f = line.split()
                out[f[0]] = f"{f[1]}: {' '.join(f[2:]).split(' / ')[0].rsplit(' ', 1)[0]}"
    return out


def main():
    out, names = Path(sys.argv[1]), sys.argv[2:]
    lab = labels()
    index = []
    for name in names:
        seq = PHONE / name
        od = out / "recordings" / name
        (od / "frames").mkdir(parents=True, exist_ok=True)
        man = serve.asl_manifest(seq, "phone", "")
        size = 0
        for f in man["frames"]:
            fn = f["url"].rsplit("/", 1)[1]
            dst = od / "frames" / fn
            if not dst.exists():
                b = io.BytesIO()
                Image.open(seq / "mav0" / "cam0" / "data" / fn).convert("L").save(b, "JPEG", quality=80)
                dst.write_bytes(b.getvalue())
            size += dst.stat().st_size
            f["url"] = f"recordings/{name}/frames/{fn}"
        (od / "manifest.json").write_text(json.dumps(man))
        (od / "imu.json").write_text(json.dumps(serve.imu_manifest(seq, "phone")))
        (od / "imu_raw.json").write_text(json.dumps(serve.imu_raw(seq)))
        for fn in ("settings.yaml", "touches.csv"):
            if (seq / fn).exists():
                shutil.copy(seq / fn, od / fn)
        index.append({"dataset": "phone", "name": name, "settings": f"recordings/{name}/settings.yaml", "clahe": False,
                      "gt": None, "label": lab.get(name, "recording"), "frames": len(man["frames"])})
        print(f"{name}: {len(man['frames'])} frames, {size / 1e6:.1f} MB")
    (out / "recordings" / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
