#!/usr/bin/env python3
"""Generate settings variants of ORB-SLAM3's official mono settings for speed/accuracy ablations.

Variants (applied to baselines/orbslam3/settings/{EuRoC,TUM-VI}_mono.yaml):
  stock      unchanged
  light-orb  ORBextractor.nFeatures x0.6, nLevels 8 -> 6
  light-map  Web.LBA.MaxOptKFs 20, Web.LBA.MaxFixedKFs 20, Web.Mapping.Neighbors 15   (see include/WebConfig.h)
  light-all  light-orb + light-map
  klt        Web.KLT.Enable 1 (hybrid ORB+KLT tracker, defaults in include/WebConfig.h)
  klt-orb    klt + light-orb
  lba5       Web.LBA.Iterations 5 (local BA Levenberg-Marquardt iterations, stock 10)
  tsm        Web.Track.TimeScaledMotion 1 (motion prediction rescaled to the frame interval; for --live runs)
  bow-lu3/bow-lu2  Web.BoW.LevelsUp 3 / 2 (for L5 / L4 vocabularies: BoW direct index at the same depth as stock L6)
  kf080/kf070      Web.KF.RefRatio 0.8 / 0.7 (keyframe insertion threshold; stock mono 0.9 -> fewer keyframes)
  cull080/cull070  Web.Mapping.CullRedundancy 0.8 / 0.7 (keyframe culling; stock 0.9 -> more culling)
  preset-*   the AR page presets (web/options.js PRESETS), visual-only part:
               preset-stock   stock
               preset-hybrid  klt + Web.Track.RelocTimeout 10
               preset-fast    klt with Web.KLT.MaxConsecutive 4 + tsm + light-map + RelocTimeout 10
               preset-light   light-orb + RelocTimeout 10 (features x0.6 of the dataset's setting; the phone uses 600)
  <v>-gyro   v in stock / preset-hybrid / preset-fast, plus Web.Gyro.Prior 1 and the dataset's IMU.T_b_c1 (gyro rotation
             prior; run with orbslam3_run --gyro so the IMU samples are passed in mono mode)
  fast-<m>, fast-gyro-<m>   round 2 (2026-10-17): preset-fast (and preset-fast-gyro) plus one change m from FAST_MODS
             (coast / coast2 = loss recovery (prior search / + local-map PnP), fastpose / posecheck (+w: warm-started
             rounds) = g2o-free pose solver / compare mode, init = lower two-view thresholds, orbpar = parallel ORB
             levels, norm = exposure-normalised KLT, ginit = gyro two-view
             initialisation (fast-gyro-ginit only), ssc = SSC keypoint selection, speed = orbpar + fastposew, nofb = no backward LK
             pass, kf080 = fewer keyframes, kltpar = optical flow on 4 threads (kltparc: serial vs 4 threads compare
             mode), speed2 = speed + kltpar, rot = far points for pure rotation (prototype), rot2 = the same as points at infinity in the pose solve)

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
            "light-all": lambda t: light_map(light_orb(t)), "klt": klt, "klt-orb": lambda t: klt(light_orb(t)), "tsm": tsm,
            "lba5": lambda t: t.rstrip("\n") + "\n\n# web port: local BA iterations (include/WebConfig.h)\nWeb.LBA.Iterations: 5\n"}


def add(text, *lines):
    return text.rstrip("\n") + "\n\n# web port (include/WebConfig.h)\n" + "".join(f"{l}\n" for l in lines)


def gyro(text, name):
    """Web.Gyro.Prior 1 + the camera -> IMU extrinsics block copied from ORB-SLAM3's inertial settings of the dataset."""
    inertial = (SRC / f"{name}_inertial.yaml").read_text()
    tbc = re.search(r"IMU\.T_b_c1:\s*!!opencv-matrix.*?\]", inertial, re.S).group(0)
    return add(text, "Web.Gyro.Prior: 1", tbc)


RELOC10 = "Web.Track.RelocTimeout: 10"
VARIANTS.update({
    # keyframe policy study (include/WebConfig.h: Web.KF.RefRatio, Web.Mapping.CullRedundancy; stock mono 0.9 / 0.9)
    # smaller vocabularies index BoW at the same depth (include/WebConfig.h Web.BoW.LevelsUp; stock 4 for L = 6)
    "bow-lu3": lambda t: add(t, "Web.BoW.LevelsUp: 3"),
    "bow-lu2": lambda t: add(t, "Web.BoW.LevelsUp: 2"),
    "kf080": lambda t: add(t, "Web.KF.RefRatio: 0.8"),
    "kf070": lambda t: add(t, "Web.KF.RefRatio: 0.7"),
    "cull080": lambda t: add(t, "Web.Mapping.CullRedundancy: 0.8"),
    "cull070": lambda t: add(t, "Web.Mapping.CullRedundancy: 0.7"),
    "preset-stock": lambda t: t,
    "preset-hybrid": lambda t: add(klt(t), RELOC10),
    "preset-fast": lambda t: add(tsm(light_map(klt(t))), "Web.KLT.MaxConsecutive: 4", RELOC10),
    "preset-light": lambda t: add(light_orb(t), RELOC10),
})

# round 2 (2026-10-17): one change at a time on top of "Hybrid, fastest" (include/WebConfig.h)
FAST_MODS = {
    "coast": ["Web.Track.Coast: 1"],
    "fastpose": ["Web.Track.FastPose: 1"],
    "posecheck": ["Web.Track.FastPose: 2"],
    "nofb": ["Web.KLT.FBThreshold: 0"],
    "kf080": ["Web.KF.RefRatio: 0.8"],
    "coast2": ["Web.Track.Coast: 2"],
    "fastposew": ["Web.Track.FastPose: 1", "Web.Track.FastPoseWarm: 1"],
    "posecheckw": ["Web.Track.FastPose: 2", "Web.Track.FastPoseWarm: 1"],
    "init": ["Web.Init.MinMatches: 60", "Web.Init.MinParallax: 0.5", "Web.Init.MinTriangulated: 40"],
    "orbpar": ["Web.ORB.Parallel: 1"],
    "norm": ["Web.KLT.Normalize: 1"],
    "ginit": ["Web.Init.Gyro: 1"],
    "ssc": ["Web.ORB.SSC: 1"],
    "speed": ["Web.ORB.Parallel: 1", "Web.Track.FastPose: 1"],
    "kltpar": ["Web.KLT.Parallel: 1"],
    "kltparc": ["Web.KLT.Parallel: 2"],
    "speed2": ["Web.ORB.Parallel: 1", "Web.Track.FastPose: 1", "Web.KLT.Parallel: 1"],
    "rot": ["Web.Rot.FarPoints: 1"],
    "rot2": ["Web.Rot.FarPoints: 2"],
    "rot3": ["Web.Rot.FarPoints: 3"],
    "cov": ["Web.KF.Coverage: 0.4"],
    "cov2": ["Web.KF.Coverage: 0.4", "Web.Map.MinBaseline: 0.005", "Web.Map.MaxCosParallax: 0.99995"],
}
def focal(text, k):  # wrong focal length (uncalibrated phone): Camera1.fx / fy scaled by k
    return re.sub(r"(Camera1\.f[xy]:\s*)([0-9.eE+-]+)", lambda m: f"{m.group(1)}{float(m.group(2)) * k:.6f}", text)


VARIANTS["fast-fx090"] = lambda t: focal(VARIANTS["preset-fast"](t), 0.9)  # focal-from-gyro check (estimate_time_offset --focal)
VARIANTS["fast-fx110"] = lambda t: focal(VARIANTS["preset-fast"](t), 1.1)
for m, lines in FAST_MODS.items():
    VARIANTS[f"fast-{m}"] = (lambda ls: lambda t: add(VARIANTS["preset-fast"](t), *ls))(lines)


def main():
    out = Path(sys.argv[1])
    only = set(sys.argv[2:])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("EuRoC_mono", "TUM-VI_mono"):
        base = (SRC / f"{name}.yaml").read_text()
        variants = dict(VARIANTS)
        for v in ("stock", "preset-hybrid", "preset-fast"):  # *-gyro: the same with the gyro rotation prior
            variants[f"{v}-gyro"] = (lambda f: lambda t: gyro(f(t), name))(VARIANTS[v])
        for m, lines in FAST_MODS.items():
            variants[f"fast-gyro-{m}"] = (lambda ls: lambda t: add(gyro(VARIANTS["preset-fast"](t), name), *ls))(lines)
        for v, f in variants.items():
            if only and v not in only:
                continue
            p = out / f"{name}_{v}.yaml"
            p.write_text(f(base), newline="\n")
            print(p)


if __name__ == "__main__":
    main()
