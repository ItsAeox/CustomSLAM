# Vendored: ORB-SLAM3

- Upstream: https://github.com/UZ-SLAMLab/ORB_SLAM3
- Commit: 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4 (v1.0, 2022-02-10)
- Licence: GPL-3.0 (see LICENSE). Authors: Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez,
  José M. M. Montiel, Juan D. Tardós (University of Zaragoza).
- Contents: include/, src/, Thirdparty/{DBoW2, g2o, Sophus}, LICENSE, README.md, Dependencies.md, Changelog.md
  (Examples, Vocabulary and evaluation tools are not vendored).

This first commit is the unmodified upstream snapshot. Every later change for the web port is a separate
commit and is guarded by `ORBSLAM3_WEB` where possible; `git log -- slam/orbslam3` lists them.
