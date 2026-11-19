p = "H:/Research/claude/ISSUES.md"
s = open(p, encoding="utf-8").read()
start = s.index("| H1 | **SUSPECTED")
end = s.index("\n", start)
new = ("| H1 | **CONFIRMED: the `--realtime` sweeps played well below real time.** The runner sleeps `T − ttrack`, the "
       "protocol of ORB-SLAM3's examples, which counts only `TrackMonocular`, not image loading. Clean measurement "
       "(native, MH01, frames 200→400 = 10 s of data, during sweeps): **19.6 s and 16.8 s on `/mnt/h` (0.51–0.60×), "
       "13.8 s on WSL ext4 (0.73×)**. Part of it is the read pattern: `cv::imread`'s small reads cost 23.5 ms per PNG "
       "on the 9P mount against 7.1 ms with 1 MB reads. Sweep wall times agree (klt-orb EuRoC: ~262 s per run for "
       "~123 s sequences). **Impact:** comparisons *within* a sweep are fair (same conditions per arm). But no result so "
       "far is at real-time speed: local mapping had more time than live, so earlier \"real time\" wording (and G1's "
       "\"the WASM residual is mapping speed\") must be read as \"at ~0.5–0.7× playback\". The first `--live` pilot was "
       "I/O-dominated: discarded. | FIX WRITTEN (not yet built: WSL hung, see H2): `orbslam3_run` reads whole files + "
       "`imdecode`; a background FrameSource thread loads the next frame; `--rt-wall` = every frame on a wall-clock "
       "schedule; `--live` = camera thread; `run_info` records `wall_s`, `seq_s`, `effective_rate`. The remaining "
       "queued sweeps should move to ext4 + `--rt-wall` (each has its own baseline arm) |\n"
       "| H2 | **WSL froze (memory, 13:07).** Three Magistrale WASM runs (>1.2 GB each, growing) plus a compile pushed the "
       "7.8 GB VM into thrashing (`vmmemWSL` 7.7 GB, host 0.7 GB free). New `wsl` connections time out "
       "(Wsl/Service/0x8007274c), no progress since 13:07. Terminating the distro was refused by the permission "
       "classifier, so I'm waiting for it to recover. | OPEN. Rule: no builds while Magistrale-length WASM runs are "
       "going; Magistrale at most 2 jobs |")
s = s[:start] + new + s[end:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
