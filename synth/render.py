"""Ray-casting renderer for synth/scene.py rooms: exact geometry, pinhole camera (OpenCV convention: x right, y down,
z forward, pixel centres at integer coordinates), mip-mapped texture sampling (no aliasing on oblique floors),
Lambertian lighting, rolling shutter (every image row at its own time) and motion blur (sub-samples over the exposure),
vignetting, auto exposure, shot + read noise, gamma and JPEG like a phone frame.

Time convention of a frame: `t` is the middle row at the middle of the exposure; row r is read at
t + (r / (H - 1) - 0.5) * readout and integrates light over [row time - exposure / 2, row time + exposure / 2].
The ground truth pose of a frame is the pose at `t`.
"""
import cv2
import numpy as np

NEAR = 0.02


class Renderer:
    def __init__(self, room, K, W, H, readout=0.0, exposure=0.0, noise=True, seed=0, blur_px_per_sample=1.0, max_samples=16,
                 blur_mode="warp"):
        self.room, self.K, self.W, self.H = room, np.asarray(K, float), W, H
        self.readout, self.exposure, self.noise = readout, exposure, noise
        self.rng = np.random.default_rng(seed)
        self.blur_px, self.max_samples, self.blur_mode = blur_px_per_sample, max_samples, blur_mode
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        self.dc = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], -1).astype(np.float32)  # z = 1: t = depth
        rr = ((xs - cx) / W) ** 2 + ((ys - cy) / H) ** 2
        self.vignette = (1.0 - 0.6 * rr).astype(np.float32)
        self.f = 0.5 * (fx + fy)
        # mip pyramids (Gaussian pyrDown) per face
        self.mips = []
        for fc in room.faces:
            lv = [fc.tex.astype(np.float32)]
            while min(lv[-1].shape) > 8:
                lv.append(cv2.pyrDown(lv[-1]))
            self.mips.append(lv)
        self.gain = None  # auto exposure state
        self.near_depth = None  # for the motion-blur sample count

    # ------------------------------------------------------------------ one instant (per-row times)
    def _cast(self, R, o):
        """R (H,3,3) camera->world per row, o (H,3) camera centre per row -> depth, face id, a, b (texture coords)"""
        H, W = self.H, self.W
        rays = np.matmul(self.dc, R.astype(np.float32).transpose(0, 2, 1))  # world directions, z_cam = 1
        depth = np.full((H, W), np.inf, np.float32)
        fid = np.full((H, W), -1, np.int16)
        A = np.zeros((H, W), np.float32); B = np.zeros((H, W), np.float32)
        for k, fc in enumerate(self.room.faces):
            box = self._bbox(fc, R, o)
            if box is None:
                continue
            r0, r1, c0, c1 = box
            rr = rays[r0:r1, c0:c1]
            oc = (o[r0:r1] - fc.c).astype(np.float32)  # (h,3)
            den = _dot(rr, fc.n)
            num = -(oc @ fc.n.astype(np.float32))[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num / den
            d = depth[r0:r1, c0:c1]
            ok = (t > NEAR) & (t < d)
            if not ok.any():
                continue
            a = ((oc @ fc.u.astype(np.float32))[:, None] + t * _dot(rr, fc.u)) * np.float32(1 / fc.lu ** 2)
            b = ((oc @ fc.v.astype(np.float32))[:, None] + t * _dot(rr, fc.v)) * np.float32(1 / fc.lv ** 2)
            ok &= (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
            if not ok.any():
                continue
            d[ok] = t[ok]
            fid[r0:r1, c0:c1][ok] = k
            A[r0:r1, c0:c1][ok] = a[ok]; B[r0:r1, c0:c1][ok] = b[ok]
        return depth, fid, A, B, rays

    def _bbox(self, fc, R, o):
        """image rows / cols that the face can cover under the poses of the first, middle and last row"""
        H, W = self.H, self.W
        cor = fc.corners()
        us, vs = [], []
        for h in (0, H // 2, H - 1):
            pc = (cor - o[h]) @ R[h]  # camera coordinates (R is camera->world)
            if (pc[:, 2] <= NEAR).all():
                continue
            if (pc[:, 2] <= NEAR).any():
                return 0, H, 0, W  # crosses the camera plane: test everything
            us += list(self.K[0, 0] * pc[:, 0] / pc[:, 2] + self.K[0, 2])
            vs += list(self.K[1, 1] * pc[:, 1] / pc[:, 2] + self.K[1, 2])
        if not us:
            return None
        c0, c1 = int(np.floor(min(us))) - 2, int(np.ceil(max(us))) + 3
        r0, r1 = int(np.floor(min(vs))) - 2, int(np.ceil(max(vs))) + 3
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, W), min(r1, H)
        if c0 >= c1 or r0 >= r1:
            return None
        return r0, r1, c0, c1

    def _shade(self, depth, fid, A, B, rays, o):
        """albedo (mip-mapped texture) x irradiance (ambient + ceiling point lights, two-sided Lambert) per pixel"""
        H, W = self.H, self.W
        img = np.full((H, W), 0.3, np.float32)
        orow = np.broadcast_to(o[:, None, :], (H, W, 3))
        for k in np.unique(fid):
            if k < 0:
                continue
            fc, mips = self.room.faces[k], self.mips[k]
            m = fid == k
            a, b, t, r = A[m], B[m], depth[m], rays[m]
            rn = np.linalg.norm(r, axis=1)
            cos = np.abs(r @ fc.n.astype(np.float32)) / rn
            # texels per pixel along the most stretched direction (trilinear-style isotropic LOD)
            tw = mips[0].shape[1] / fc.lu
            tpp = (t * rn / self.f) / np.maximum(cos, 0.08) * tw
            lod = np.clip(np.log2(np.maximum(tpp, 1.0)), 0, len(mips) - 1)
            l0 = np.floor(lod).astype(np.int32); fr = (lod - l0).astype(np.float32)
            l1 = np.minimum(l0 + 1, len(mips) - 1)
            alb = np.zeros(len(a), np.float32)
            for lv in np.unique(np.concatenate([l0, l1])):
                s0, s1 = l0 == lv, l1 == lv
                if s0.any():
                    alb[s0] += (1 - fr[s0]) * _bilinear(mips[lv], a[s0], b[s0])
                if s1.any():
                    alb[s1] += fr[s1] * _bilinear(mips[lv], a[s1], b[s1])
            pts = orow[m].astype(np.float32) + r * t[:, None]
            lit = np.full(len(a), 0.45, np.float32)
            for L in self.room.lights:
                dl = L.astype(np.float32) - pts
                dist = np.linalg.norm(dl, axis=1)
                lit += 0.9 * np.abs(dl @ fc.n.astype(np.float32)) / dist / (1 + 0.15 * dist ** 2)
            img[m] = alb * lit
        return img

    def instant(self, R, o):
        depth, fid, A, B, rays = self._cast(R, o)
        return self._shade(depth, fid, A, B, rays, o), depth

    # ------------------------------------------------------------------ a frame
    def n_samples(self, pose_fn, t):
        """motion-blur sub-samples: ~one per `blur_px` of image motion during the exposure"""
        if self.exposure <= 0:
            return 1
        e = self.exposure
        R0, o0 = pose_fn(np.array([t - e / 2])); R1, o1 = pose_fn(np.array([t + e / 2]))
        dR = R0[0].T @ R1[0]
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        near = self.near_depth if self.near_depth else 0.5  # 10th percentile of the previous frame's depth
        px = ang * self.f + np.linalg.norm(o1[0] - o0[0]) * self.f / max(near, 0.2)
        return int(np.clip(np.ceil(px / self.blur_px), 1, self.max_samples))

    def frame(self, pose_fn, t, return_depth=False):
        """one frame at time t; motion blur = the central instant rendered exactly, re-projected (with its per-pixel
        depth) to every sub-sample pose of the exposure and averaged (blur_mode 'warp', default), or every sub-sample
        rendered in full ('full', slow; the two agree up to depth edges)"""
        H, W = self.H, self.W
        rows = (np.arange(H) / (H - 1) - 0.5) * self.readout
        S = self.n_samples(pose_fn, t)
        Rc, oc = pose_fn(t + rows)
        img_c, depth = self.instant(Rc, oc)
        if S == 1:
            acc = img_c
        elif self.blur_mode == "full":
            acc = np.zeros((H, W), np.float32)
            for s in range(S):
                R, o = pose_fn(t + rows + ((s + 0.5) / S - 0.5) * self.exposure)
                acc += self.instant(R, o)[0]
            acc /= S
        else:
            rays = np.matmul(self.dc, Rc.astype(np.float32).transpose(0, 2, 1))
            d = np.where(np.isfinite(depth), depth, 1e3).astype(np.float32)
            P = oc[:, None, :].astype(np.float32) + rays * d[..., None]  # world point of each centre pixel
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            acc = np.zeros((H, W), np.float32)
            K = self.K
            for s in range(S):
                R, o = pose_fn(t + rows + ((s + 0.5) / S - 0.5) * self.exposure)
                pc = np.matmul(P - o[:, None, :].astype(np.float32), R.astype(np.float32))  # R^T (P - o) per row
                z = np.maximum(pc[..., 2], 1e-3)
                u = K[0, 0] * pc[..., 0] / z + K[0, 2]; v = K[1, 1] * pc[..., 1] / z + K[1, 2]
                # backward map: the sub-sample pixel x shows the centre pixel x - flow(x) (small, smooth flow)
                mx = (2 * xs - u).astype(np.float32); my = (2 * ys - v).astype(np.float32)
                acc += cv2.remap(img_c, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            acc /= S
        img = acc * self.vignette
        fin = depth[np.isfinite(depth)]
        self.near_depth = float(np.percentile(fin, 10)) if fin.size else None
        # auto exposure: gain towards mean 0.42 with a ~0.3 s time constant (frames at ~30 fps)
        target = 0.42 / max(float(np.mean(img)), 1e-3)
        self.gain = target if self.gain is None else self.gain * 0.9 + target * 0.1
        lin = np.clip(img * self.gain, 0, 1)
        if self.noise:
            sig = np.sqrt(0.0006 * lin + (1.5 / 255) ** 2)
            lin = np.clip(lin + sig * self.rng.standard_normal(lin.shape).astype(np.float32), 0, 1)
        out = (255.0 * lin ** (1 / 2.2) + 0.5).astype(np.uint8)
        return (out, depth, S) if return_depth else (out, S)


def _bilinear(img, a, b):
    """img sampled at texture coords a (columns, 0..1) and b (rows, 0..1), clamped at the edges"""
    h, w = img.shape
    x = a * w - 0.5; y = b * h - 0.5
    x0 = np.clip(np.floor(x).astype(int), 0, w - 1); y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(x - x0, 0, 1).astype(np.float32); fy = np.clip(y - y0, 0, 1).astype(np.float32)
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def _dot(rr, vec):
    """rays (h, w, 3) . vec; a single-axis vector (axis-aligned faces) costs one multiply"""
    nz = np.flatnonzero(np.abs(vec) > 1e-12)
    if len(nz) == 1:
        return rr[..., nz[0]] * np.float32(vec[nz[0]])
    return rr @ vec.astype(np.float32)
