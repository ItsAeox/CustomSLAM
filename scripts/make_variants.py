#!/usr/bin/env python3
"""Generate settings variants of ORB-SLAM3's official mono settings for speed/accuracy ablations.

Variants (applied to baselines/orbslam3/settings/{EuRoC,TUM-VI}_mono.yaml):
  stock      unchanged
  light-orb  ORBextractor.nFeatures x0.6, nLevels 8 -> 6
  light-map  Web.LBA.MaxOptKFs 20, Web.LBA.MaxFixedKFs 20, Web.Mapping.Neighbors 15   (see include/WebConfig.h)
  light-all  light-orb + light-map
  klt        Web.KLT.Enable 1 (hybrid ORB+KLT tracker, defaults in include/WebConfig.h)
  klt-orb    klt + light-orb
  tsm        Web.Track.TimeScaledMotion 1 (motion prediction rescaled to the frame interval; for --live runs)

Usage: make_variants.py <out_dir> [variant ...]   (default: all)
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "baselines" / "orbslam3" / "settings"

ORB = {"feat_scale": 0.6, "levels": 6}
MAP = {"Web.LBA.MaxOptKFs": 20, "Web.LBA.MaxFixedKFs": 20, "Web.Mapping.Neighbors": 15}


def light_orb(text):
    n = int(re.search(r"ORBextractor\.nFeatures:\s*(\d+)", text).group(1))
    text = re.sub(r"(ORBextractor\.nFeatures:\s*)\d+", lambda m: f"{m.group(1)}{round(n * ORB['feat_scale'])}", text)
    return re.sub(r"(ORBextractor\.nLevels:\s*)\d+", lambda m: f"{m.group(1)}{ORB['levels']}", text)


def light_map(text):
    return text.rstrip("\n") + "\n\n# web port tuning (include/WebConfig.h)\n" + \
        "".join(f"{k}: {v}\n" for k, v in MAP.items())


def klt(text):
    return text.rstrip("\n") + "\n\n# web port: KLT tracking frames (include/WebConfig.h)\nWeb.KLT.Enable: 1\n"


def tsm(text):
    return text.rstrip("\n") + "\n\n# web port: time-scaled motion model (include/WebConfig.h)\nWeb.Track.TimeScaledMotion: 1\n"


VARIANTS = {"stock": lambda t: t, "light-orb": light_orb, "light-map": light_map,
            "light-all": lambda t: light_map(light_orb(t)), "klt": klt, "klt-orb": lambda t: klt(light_orb(t)), "tsm": tsm}


def main():
    out = Path(sys.argv[1])
    only = set(sys.argv[2:])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("EuRoC_mono", "TUM-VI_mono"):
        base = (SRC / f"{name}.yaml").read_text()
        for v, f in VARIANTS.items():
            if only and v not in only:
                continue
            p = out / f"{name}_{v}.yaml"
            p.write_text(f(base), newline="\n")
            print(p)


if __name__ == "__main__":
    main()
