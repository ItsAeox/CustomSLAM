"""Decode AR page snapshots (data URLs POSTed to results/browser/snap_*.txt) into JPEG files next to them."""
import base64
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "H:/Research/results/browser")
for f in sorted(d.glob("snap_*.txt")):
    data = f.read_text().strip()
    if not data.startswith("data:image/jpeg;base64,"):
        continue
    out = f.with_suffix(".jpg")
    out.write_bytes(base64.b64decode(data.split(",", 1)[1]))
    f.unlink()
    print(out)
