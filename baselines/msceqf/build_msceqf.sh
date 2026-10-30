#!/usr/bin/env bash
# Native MSCEqF reference (Multi State Constraint Equivariant Filter, Fornasier et al., RA-L 2023; aau-cns/MSCEqF 2cb653a,
# Apache-2.0) + msceqf_run (this folder). Sources on H: (/mnt/h/Research/ext); its FetchContent dependencies pinned to
# local clones there: Lie-plusplus ec459c3 (main), yaml-cpp c5e6536 (master), Eigen 3.4.0 (3147391). apt OpenCV 4.5
# (as EqVIO; /opt/slam has no highgui), apt Boost. Build in /root/build/msceqf (WSL disk, H:\WSL). -j3 (7 GB VM). MSCEqF's own Release flags
# (-O3 -flto -march=native) are kept.
set -euo pipefail
E=/mnt/h/Research/ext
cmake -S /mnt/h/Research/CustomSLAM/baselines/msceqf -B /root/build/msceqf -DCMAKE_BUILD_TYPE=Release -DMSCEQF_SRC=$E/MSCEqF \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_SOURCE_DIR_LIEPLUSPLUS=$E/Lie-plusplus \
  -DFETCHCONTENT_SOURCE_DIR_YAML-CPP=$E/yaml-cpp -DFETCHCONTENT_SOURCE_DIR_EIGEN=$E/eigen-3.4.0 \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF \
  -DBUILD_TESTING=OFF > /root/build/msceqf_cmake.log 2>&1 || { tail -30 /root/build/msceqf_cmake.log; exit 1; }
nice -n 10 cmake --build /root/build/msceqf -j${J:-3} --target msceqf_run 2>&1 | { grep -E "error|Error" || true; } | head -30
ls -la --time-style=+%T /root/build/msceqf/msceqf_run
echo "=== done $(date +%T)"
