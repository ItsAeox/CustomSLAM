"""Export XFeat (verlab/accelerated_features, Apache-2.0) to ONNX for a browser feasibility test.

Usage (WSL, venv with torch + onnx + onnxruntime): python scripts/xfeat_export.py /root/xfeat   -> /root/xfeat_{640x480,480x640}.onnx
Requires the official repo at /root/accelerated_features (weights/xfeat.pt).

Graph: image (1,1,H,W float, H and W divisible by 32) ->
  desc  (1,64,H/8,W/8)  L2-normalised dense descriptors
  score (1,1,H,W)       keypoint heatmap after 5x5 non-maximum suppression (0 where not a local maximum)
  rel   (1,1,H/8,W/8)   reliability map
Top-k selection and descriptor interpolation stay in JavaScript (cheap). Mirrors XFeat.detectAndCompute.
"""
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/accelerated_features")
from modules.model import XFeatModel  # noqa: E402


class XFeatExport(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        M1, K1, H1 = self.net(x)
        M1 = F.normalize(M1, dim=1)
        # XFeat.get_kpts_heatmap: softmax over 65 bins, drop the last, 8x8 depth-to-space. The depth-to-space is a
        # stride-8 transposed convolution with one-hot kernels (plain ops; the 6-D reshape version failed on WebGPU).
        probs = F.softmax(K1, 1)[:, :64]
        w = torch.zeros(64, 1, 8, 8)
        for k in range(64):
            w[k, 0, k // 8, k % 8] = 1.0
        heat = F.conv_transpose2d(probs, w, stride=8)
        local_max = F.max_pool2d(heat, kernel_size=5, stride=1, padding=2)  # XFeat.NMS (5x5)
        score = heat * (heat >= local_max).float()
        return M1, score, H1


net = XFeatModel()


def unfold_as_conv(x, ws=8):
    """Same result as XFeatModel._unfold2d (channel c*ws*ws + i*ws + j <- pixel (i, j) of each ws x ws block), written
    as a stride-ws convolution with one-hot kernels: exports to a plain Conv instead of 6-D reshapes/transposes, which
    ONNX Runtime Web's WebGPU backend computed wrongly."""
    C = x.shape[1]
    w = torch.zeros(C * ws * ws, 1, ws, ws, dtype=x.dtype)
    for c in range(C):
        for i in range(ws):
            for j in range(ws):
                w[c * ws * ws + i * ws + j, 0, i, j] = 1.0
    return F.conv2d(x, w, stride=ws, groups=C) if C == 1 else F.conv2d(x, w, stride=ws, groups=C)


_orig_unfold = XFeatModel._unfold2d
t = torch.rand(1, 1, 64, 48)
assert torch.equal(_orig_unfold(None, t, ws=8), unfold_as_conv(t, 8)), "unfold rewrite differs"
XFeatModel._unfold2d = lambda self, x, ws=2: unfold_as_conv(x, ws)
net.load_state_dict(torch.load("/root/accelerated_features/weights/xfeat.pt", map_location="cpu"))
net.eval()
model = XFeatExport(net).eval()
out = sys.argv[1] if len(sys.argv) > 1 else "/root/xfeat"
# fixed shapes: Tensor.unfold (8x8 cells) cannot be exported with dynamic H/W
for (h, w) in ((640, 480), (480, 640)):
    torch.onnx.export(model, torch.randn(1, 1, h, w), f"{out}_{h}x{w}.onnx", input_names=["image"],
                      output_names=["desc", "score", "rel"], opset_version=17, dynamo=False)
print("exported", out + "_640x480.onnx", out + "_480x640.onnx")

# check against the PyTorch graph with onnxruntime (CPU)
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(out + "_640x480.onnx", providers=["CPUExecutionProvider"])
x = torch.rand(1, 1, 640, 480) * 255
with torch.no_grad():
    ref = model(x)
got = sess.run(None, {"image": x.numpy()})
for name, a, b in zip(["desc", "score", "rel"], ref, got):
    print(name, tuple(b.shape), "max abs diff", float(np.abs(a.numpy() - b).max()))
import time  # noqa: E402
t = time.perf_counter()
for _ in range(10):
    sess.run(None, {"image": x.numpy()})
print(f"onnxruntime CPU (WSL, all threads): {(time.perf_counter() - t) / 10 * 1000:.1f} ms per 480x640 frame")
