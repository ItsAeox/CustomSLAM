#!/usr/bin/env bash
# Moved to vocab_eval2.sh (run by queue7 on snap-kf1): ORB-SLAM3 builds BoW feature vectors 4 levels up from the leaves
# (Frame/KeyFrame::ComputeBoW, transform(..., 4)), so with an L5 / L4 vocabulary the direct index has 10 / 1 node(s) and
# SearchByBoW becomes near brute force -> vocabulary size would be confounded with matching cost. vocab_eval2 sets
# Web.BoW.LevelsUp = L - 2 so every vocabulary indexes at the same depth (100 nodes).
echo "vocab_eval.sh: moved to vocab_eval2.sh (queue7) $(date +%T)"
