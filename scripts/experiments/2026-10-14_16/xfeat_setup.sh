set -e
cd /root
[ -d accelerated_features ] || git clone -q --depth 1 https://github.com/verlab/accelerated_features.git
cd accelerated_features && git log --oneline -1 && ls weights && du -sh weights
[ -d /opt/slam/venv-xfeat ] || python3 -m venv /opt/slam/venv-xfeat
nice -n 19 /opt/slam/venv-xfeat/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch 2>&1 | tail -2
nice -n 19 /opt/slam/venv-xfeat/bin/pip install -q onnx onnxruntime opencv-python-headless 2>&1 | tail -2
/opt/slam/venv-xfeat/bin/python -c "import torch, onnx, onnxruntime; print('torch', torch.__version__, 'onnx', onnx.__version__, 'ort', onnxruntime.__version__)"
