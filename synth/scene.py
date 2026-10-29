"""Synthetic rooms for the WebVIO dataset (synth/README.md): rectangles in metres with procedural grayscale textures.

World frame: x, y horizontal, z up (gravity -z), origin at a floor corner of the room. Every surface is a textured
rectangle `Face(c, u, v, tex, ...)`: corner c, edge vectors u (texture columns) and v (texture rows), texture image
covering the whole rectangle, so its physical scale is exact (texels per metre = width_px / |u|). Boxes (furniture)
are five or six faces. Decals (posters, rugs, the calibration board) are faces lifted 2 mm off their surface.
Textures are drawn by code only (noise, shapes, text glyphs): reproducible from the seed and free to publish.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

PX_PER_M = 400  # texture resolution (2.5 mm texels); mip-mapped at render time


@dataclass
class Face:
    c: np.ndarray            # corner (3,)
    u: np.ndarray            # edge along texture columns (3,)
    v: np.ndarray            # edge along texture rows (3,)
    tex: np.ndarray          # float32 HxW in [0, 1] (albedo)
    name: str = ""
    n: np.ndarray = field(init=False)
    lu: float = field(init=False)
    lv: float = field(init=False)

    def __post_init__(self):
        self.c, self.u, self.v = (np.asarray(x, np.float64) for x in (self.c, self.u, self.v))
        self.lu, self.lv = float(np.linalg.norm(self.u)), float(np.linalg.norm(self.v))
        n = np.cross(self.u, self.v)
        self.n = n / np.linalg.norm(n)

    def corners(self):
        return np.array([self.c, self.c + self.u, self.c + self.u + self.v, self.c + self.v])


# ------------------------------------------------------------------------------------------------ textures
def value_noise(h, w, rng, scales=(4, 16, 64, 256), weights=(1.0, 0.6, 0.35, 0.2)):
    """multi-octave value noise in [0, 1]: random grids of `s` px cells upsampled (bicubic) and summed"""
    out = np.zeros((h, w), np.float32)
    for s, wt in zip(scales, weights):
        gh, gw = max(2, h // s + 2), max(2, w // s + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        out += wt * cv2.resize(g, (gw * s, gh * s), interpolation=cv2.INTER_CUBIC)[:h, :w]
    out -= out.min()
    return out / max(out.max(), 1e-6)


def tex_size(w_m, h_m):
    return max(8, int(round(h_m * PX_PER_M))), max(8, int(round(w_m * PX_PER_M)))


def tiles(w_m, h_m, rng, tile=0.6, grout=0.005, base=0.7, var=0.05):
    """stone floor tiles (tile x tile m) with grout lines and per-tile mottling"""
    h, w = tex_size(w_m, h_m)
    t = 0.5 + 0.35 * (value_noise(h, w, rng, (6, 24, 96), (0.5, 0.35, 0.25)) - 0.5)
    tp = int(tile * PX_PER_M)
    out = np.empty((h, w), np.float32)
    for i in range(0, h, tp):
        for j in range(0, w, tp):
            out[i:i + tp, j:j + tp] = base + var * rng.standard_normal() + 0.25 * (t[i:i + tp, j:j + tp] - 0.5)
    g = max(1, int(grout * PX_PER_M))
    for i in range(0, h, tp):
        out[i:i + g, :] = 0.28
    for j in range(0, w, tp):
        out[:, j:j + g] = 0.28
    return np.clip(out, 0, 1)


def planks(w_m, h_m, rng, width=0.12, base=0.45):
    """wooden floor / table top: planks along u with grain"""
    h, w = tex_size(w_m, h_m)
    pw = max(4, int(width * PX_PER_M))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = value_noise(h, w, rng, (8, 32, 128), (0.5, 0.3, 0.2))
    out = np.empty((h, w), np.float32)
    for i in range(0, h, pw):
        b = base + 0.1 * rng.standard_normal()
        f = rng.uniform(20, 60)
        sl = slice(i, i + pw)
        out[sl] = b + 0.06 * np.sin(xx[sl] / f + 6 * n[sl]) + 0.12 * (n[sl] - 0.5)
        out[i:i + 2] = 0.22
        # plank ends at random positions
        for x0 in np.cumsum(rng.uniform(0.6, 1.8, 8) * PX_PER_M).astype(int):
            if x0 < w:
                out[sl, x0:x0 + 2] = 0.24
    return np.clip(out, 0, 1)


def plaster(w_m, h_m, rng, base=0.78):
    """painted wall: weak low-frequency texture (real walls give few corners)"""
    h, w = tex_size(w_m, h_m)
    return np.clip(base + 0.06 * (value_noise(h, w, rng, (16, 64, 256), (0.3, 0.4, 0.3)) - 0.5), 0, 1).astype(np.float32)


def fabric(w_m, h_m, rng, base=0.35):
    h, w = tex_size(w_m, h_m)
    n = value_noise(h, w, rng, (3, 12, 48), (0.5, 0.3, 0.2))
    weave = 0.04 * np.sin(np.arange(w)[None, :] * 1.3) * np.sin(np.arange(h)[:, None] * 1.3)
    return np.clip(base + 0.18 * (n - 0.5) + weave, 0, 1).astype(np.float32)


def poster(w_m, h_m, rng):
    """picture / poster: random shapes, lines and text glyphs on a mottled background (rich corners)"""
    h, w = tex_size(w_m, h_m)
    img = (0.25 + 0.5 * value_noise(h, w, rng, (32, 128), (0.6, 0.4))).astype(np.float32)
    k = (h * w) / (PX_PER_M ** 2)  # area in m^2
    for _ in range(int(25 * k) + 6):
        col = float(rng.uniform(0, 1))
        kind = rng.integers(0, 4)
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        s = int(rng.uniform(0.02, 0.15) * PX_PER_M)
        if kind == 0:
            cv2.circle(img, p, s, col, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(img, p, (p[0] + s, p[1] + int(s * rng.uniform(0.4, 2))), col, -1, cv2.LINE_AA)
        elif kind == 2:
            pts = (np.array(p) + rng.integers(-s, s, (int(rng.integers(3, 6)), 2))).astype(np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
        else:
            cv2.line(img, p, (p[0] + int(rng.integers(-2 * s, 2 * s)), p[1] + int(rng.integers(-2 * s, 2 * s))), col,
                     max(1, s // 10), cv2.LINE_AA)
    text(img, rng, n=int(6 * k) + 2)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), 0.1, max(2, int(0.01 * PX_PER_M)))
    return np.clip(img, 0, 1)


def text(img, rng, n=4):
    h, w = img.shape
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    for _ in range(n):
        s = "".join(rng.choice(list(alphabet), int(rng.integers(4, 14))))
        scale = float(rng.uniform(0.6, 2.2)) * PX_PER_M / 400
        cv2.putText(img, s, (int(rng.integers(0, max(1, w - 50))), int(rng.integers(20, max(21, h)))),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, float(rng.choice([0.05, 0.95])), max(1, int(2 * scale)), cv2.LINE_AA)


def books(w_m, h_m, rng):
    """book shelf front: vertical spines of random widths / heights, shelf boards"""
    h, w = tex_size(w_m, h_m)
    img = np.full((h, w), 0.15, np.float32)
    shelf_h = int(0.36 * PX_PER_M)
    for y0 in range(0, h, shelf_h):
        x = 0
        while x < w:
            bw = int(rng.uniform(0.02, 0.05) * PX_PER_M)
            bh = int(rng.uniform(0.6, 0.95) * shelf_h)
            img[min(h, y0 + shelf_h - bh):y0 + shelf_h, x:x + bw] = rng.uniform(0.2, 0.9)
            x += bw + int(rng.integers(0, 3))
        img[y0:y0 + int(0.02 * PX_PER_M)] = 0.55
    img += 0.05 * (value_noise(h, w, rng, (4, 16), (0.6, 0.4)) - 0.5)
    return np.clip(img, 0, 1)


def checkerboard(w_m, h_m, square=0.1):
    """calibration board of known scale: square x square m squares (white border of one square)"""
    h, w = tex_size(w_m, h_m)
    yy, xx = np.mgrid[0:h, 0:w]
    sq = square * PX_PER_M
    b = ((xx // sq).astype(int) + (yy // sq).astype(int)) % 2
    img = np.where(b == 0, 0.92, 0.08).astype(np.float32)
    bd = int(sq)
    img[:bd, :] = img[-bd:, :] = 0.92
    img[:, :bd] = img[:, -bd:] = 0.92
    return img


def rug(w_m, h_m, rng):
    h, w = tex_size(w_m, h_m)
    img = (0.3 + 0.3 * value_noise(h, w, rng, (8, 32), (0.5, 0.5))).astype(np.float32)
    cy, cx = h / 2, w / 2
    for r in np.linspace(0.45, 0.05, 6):
        cv2.ellipse(img, (int(cx), int(cy)), (int(r * w), int(r * h)), 0, 0, 360, float(rng.uniform(0.1, 0.9)),
                    max(2, int(0.02 * PX_PER_M)), cv2.LINE_AA)
    for _ in range(40):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.1, 0.45)
        cv2.circle(img, (int(cx + rr * w * np.cos(a)), int(cy + rr * h * np.sin(a))), int(0.03 * PX_PER_M),
                   float(rng.uniform(0, 1)), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), 0.15, int(0.04 * PX_PER_M))
    return np.clip(img, 0, 1)


def stripes(w_m, h_m, rng, base=0.5):
    """striped / patterned upholstery: bands of random widths and tones, small woven motif"""
    h, w = tex_size(w_m, h_m)
    img = np.empty((h, w), np.float32)
    y = 0
    while y < h:
        bh = int(rng.uniform(0.01, 0.05) * PX_PER_M)
        img[y:y + bh] = float(rng.choice([0.12, 0.3, 0.55, 0.8, 0.9]))
        y += bh
    xx = np.arange(w)[None, :]
    img += 0.08 * np.sign(np.sin(xx * 2 * np.pi / max(4, 0.012 * PX_PER_M)))
    img += 0.08 * (value_noise(h, w, rng, (3, 12), (0.6, 0.4)) - 0.5)
    return np.clip(img, 0, 1)


def slats(w_m, h_m, rng, base=0.22):
    """dark wooden furniture side: horizontal slats with gaps and grain (high contrast against a light floor)"""
    h, w = tex_size(w_m, h_m)
    img = np.empty((h, w), np.float32)
    n = value_noise(h, w, rng, (4, 16, 64), (0.4, 0.4, 0.2))
    sl = max(6, int(rng.uniform(0.06, 0.1) * PX_PER_M))
    for y in range(0, h, sl):
        img[y:y + sl] = base + 0.06 * rng.standard_normal() + 0.15 * (n[y:y + sl] - 0.5)
        img[y:y + max(2, sl // 10)] = 0.06
    fr = max(4, int(0.05 * PX_PER_M))
    img[:, :fr] = img[:, -fr:] = base * 0.8
    img[:fr] = img[-fr:] = base * 0.8
    return np.clip(img, 0, 1)


def curtain(w_m, h_m, rng):
    """lace curtain: light cloth with folds and repeated ornaments"""
    h, w = tex_size(w_m, h_m)
    xx = np.arange(w)[None, :].astype(np.float32)
    img = 0.75 + 0.1 * np.sin(xx / (0.06 * PX_PER_M) * 2 * np.pi) + np.zeros((h, 1), np.float32)
    img += 0.05 * (value_noise(h, w, rng, (8, 32), (0.5, 0.5)) - 0.5)
    step = int(0.12 * PX_PER_M)
    for y in range(step // 2, h, step):
        for x in range(step // 2 + (y // step % 2) * step // 2, w, step):
            cv2.ellipse(img, (x, y), (step // 4, step // 3), float(rng.uniform(0, 180)), 0, 360, 0.45, 2, cv2.LINE_AA)
            cv2.circle(img, (x, y), step // 10, 0.4, -1, cv2.LINE_AA)
    return np.clip(img, 0, 1)


def screen(w_m, h_m, rng):
    """TV / monitor: dark glossy panel with a thin frame and a faint reflection"""
    h, w = tex_size(w_m, h_m)
    img = np.full((h, w), 0.07, np.float32)
    img += 0.05 * np.linspace(0, 1, w, dtype=np.float32)[None, :] * np.linspace(1, 0, h, dtype=np.float32)[:, None]
    f = max(3, int(0.015 * PX_PER_M))
    img[:f] = img[-f:] = 0.02; img[:, :f] = img[:, -f:] = 0.02
    return img


# ------------------------------------------------------------------------------------------------ geometry
def box(x0, y0, z0, sx, sy, sz, texfn, rng, name, top=None, bottom=False):
    """axis-aligned box [x0, x0+sx] x [y0, y0+sy] x [z0, z0+sz] as outward-facing rectangles"""
    X, Y, Z = np.eye(3)
    p = np.array([x0, y0, z0], float)
    f = [
        Face(p, X * sx, Z * sz, texfn(sx, sz, rng), name + ".front"),                    # y = y0 (normal -y)
        Face(p + Y * sy + X * sx, -X * sx, Z * sz, texfn(sx, sz, rng), name + ".back"),  # y = y0 + sy
        Face(p + Y * sy, -Y * sy, Z * sz, texfn(sy, sz, rng), name + ".left"),           # x = x0
        Face(p + X * sx, Y * sy, Z * sz, texfn(sy, sz, rng), name + ".right"),           # x = x0 + sx
        Face(p + Z * sz, X * sx, Y * sy, (top or texfn)(sx, sy, rng), name + ".top"),     # z = z0 + sz
    ]
    if bottom:
        f.append(Face(p + Y * sy, X * sx, -Y * sy, texfn(sx, sy, rng), name + ".bottom"))
    # the box's 6th face (bottom) normally rests on the floor
    return f


@dataclass
class Room:
    name: str
    size: tuple                 # (sx, sy, sz) m
    faces: list
    lights: np.ndarray          # (L, 3) point lights
    free: tuple                 # (x0, y0, x1, y1): open floor area for trajectories
    targets: dict               # named points of interest (3,)
    known: dict                 # objects of exactly known size (for the paper / checks)
    boxes: list = field(default_factory=list)  # (name, x0, y0, z0, sx, sy, sz): furniture, for collision checks


def make_room(kind: str, seed: int) -> Room:
    """three layouts; 'living' (default), 'bedroom', 'office'"""
    rng = np.random.default_rng(seed)
    X, Y, Z = np.eye(3)
    if kind == "living":
        sx, sy, sz = 6.0, 5.0, 2.6
    elif kind == "bedroom":
        sx, sy, sz = 4.5, 4.0, 2.5
    else:
        sx, sy, sz = 5.0, 4.0, 2.7
    faces = []
    boxes = []

    def B(x0, y0, z0, bx, by, bz, texfn, rng_, name, top=None):
        boxes.append((name, x0, y0, z0, bx, by, bz))
        return box(x0, y0, z0, bx, by, bz, texfn, rng_, name, top=top)
    floor_tex = tiles if kind != "bedroom" else planks
    # floor (normal +z), ceiling (normal -z), walls facing inward
    faces.append(Face([0, 0, 0], X * sx, Y * sy, floor_tex(sx, sy, rng), "floor"))
    faces.append(Face([0, sy, sz], X * sx, -Y * sy, plaster(sx, sy, rng, 0.85), "ceiling"))
    faces.append(Face([0, 0, sz], X * sx, -Z * sz, plaster(sx, sz, rng), "wall.y0"))          # y = 0, normal +y
    faces.append(Face([sx, sy, sz], -X * sx, -Z * sz, plaster(sx, sz, rng, 0.74), "wall.y1"))  # y = sy, normal -y
    faces.append(Face([0, sy, sz], -Y * sy, -Z * sz, plaster(sy, sz, rng, 0.8), "wall.x0"))                  # x = 0, normal +x
    faces.append(Face([sx, 0, sz], Y * sy, -Z * sz, plaster(sy, sz, rng, 0.76), "wall.x1"))   # x = sx, normal -x
    eps = 0.002
    # decals: posters on every wall (rich texture), a rug, the 1 m calibration board on the floor
    def wall_decal(wall, a0, z0, w, h, fn, name):  # u = "right" as seen from inside the room, v = down: upright, unmirrored
        if wall == "y0":
            return Face([a0 + w, eps, z0 + h], -X * w, -Z * h, fn(w, h, rng), name)
        if wall == "y1":
            return Face([a0, sy - eps, z0 + h], X * w, -Z * h, fn(w, h, rng), name)
        if wall == "x0":
            return Face([eps, a0, z0 + h], Y * w, -Z * h, fn(w, h, rng), name)
        return Face([sx - eps, a0 + w, z0 + h], -Y * w, -Z * h, fn(w, h, rng), name)
    for wall, length in (("y0", sx), ("y1", sx), ("x0", sy), ("x1", sy)):
        a = 0.3
        while a < length - 0.6:
            w = float(rng.uniform(0.5, 1.4)); h = float(rng.uniform(0.4, 1.0))
            if a + w > length - 0.2:
                break
            faces.append(wall_decal(wall, a, float(rng.uniform(0.9, 1.6)), w, h, poster, f"poster.{wall}.{a:.1f}"))
            a += w + float(rng.uniform(0.15, 0.5))
    known = {}
    # furniture along the walls (keeps the centre free)
    if kind == "living":
        faces += B(0.3, sy - 1.0, 0, 2.2, 0.9, 0.45, slats, rng, "sofa", top=stripes)
        faces += B(0.3, sy - 0.35, 0.45, 2.2, 0.25, 0.45, stripes, rng, "sofa_back")
        faces += B(0.2, 1.2, 0, 0.8, 0.8, 0.45, slats, rng, "armchair", top=stripes)
        faces += B(0.2, 1.2, 0.45, 0.2, 0.8, 0.45, stripes, rng, "armchair_back")
        faces += B(sx - 0.45, 0.6, 0, 0.4, 1.6, 1.8, books, rng, "shelf")
        faces += B(sx - 0.5, 2.6, 0, 0.45, 1.6, 0.6, slats, rng, "sideboard", top=planks)
        faces += B(2.4, 1.6, 0, 1.2, 0.6, 0.45, planks, rng, "coffee_table", top=poster)
        faces += B(2.6, 1.7, 0.45, 0.3, 0.22, 0.04, poster, rng, "magazine")
        faces += B(sx - 0.42, 3.0, 0.6, 0.3, 0.3, 0.25, poster, rng, "box_on_sideboard")
        faces += B(4.9, 0.2, 0, 0.35, 0.35, 0.4, fabric, rng, "pouf")
        faces.append(Face([3.0, sy - 0.003, 2.2], X * 1.6, -Z * 1.9, curtain(1.6, 1.9, rng), "curtain"))
        faces.append(Face([2.0, 0.003, 1.6], -X * 1.1, -Z * 0.65, screen(1.1, 0.65, rng), "tv"))
        faces += B(0.8, 0.05, 0, 1.5, 0.4, 0.5, slats, rng, "tv_stand", top=planks)
        faces.append(Face([1.0, 0.8, eps], X * 2.0, Y * 1.4, rug(2.0, 1.4, rng), "rug"))
        faces.append(Face([3.9, 2.9, eps], X * 1.0, Y * 1.0, checkerboard(1.0, 1.0), "board_1m"))
        known["board_1m"] = {"corner": [3.9, 2.9, eps], "size_m": [1.0, 1.0], "square_m": 0.1}
        known["coffee_table"] = {"corner": [2.4, 1.6, 0], "size_m": [1.2, 0.6, 0.45]}
        free = (1.0, 0.8, 4.8, 3.7)
        targets = {"table": np.array([3.0, 1.9, 0.45]), "board": np.array([4.4, 3.4, 0]), "centre": np.array([3.0, 2.5, 0.5])}
    elif kind == "bedroom":
        faces += B(0.2, sy - 2.2, 0, 1.6, 2.0, 0.55, slats, rng, "bed", top=stripes)
        faces += B(0.2, sy - 0.15, 0, 1.6, 0.1, 1.1, slats, rng, "headboard")
        faces += B(1.85, sy - 0.5, 0, 0.45, 0.4, 0.5, slats, rng, "nightstand", top=poster)
        faces += B(sx - 0.6, 0.3, 0, 0.5, 1.2, 0.75, planks, rng, "desk", top=poster)
        faces += B(sx - 0.55, 0.5, 0.75, 0.3, 0.4, 0.2, books, rng, "desk_books")
        faces += B(2.0, 0.2, 0, 1.0, 0.5, 2.0, books, rng, "wardrobe")
        faces += B(3.3, 0.15, 0, 0.4, 0.4, 0.35, stripes, rng, "laundry_box")
        faces.append(Face([sx - 0.003, 1.8, 2.2], -Y * 0 + Y * 1.4, -Z * 1.8, curtain(1.4, 1.8, rng), "curtain"))
        faces.append(Face([0.3, 0.6, eps], X * 1.4, Y * 1.0, rug(1.4, 1.0, rng), "rug"))
        faces.append(Face([2.2, 1.5, eps], X * 1.0, Y * 1.0, checkerboard(1.0, 1.0), "board_1m"))
        known["board_1m"] = {"corner": [2.2, 1.5, eps], "size_m": [1.0, 1.0], "square_m": 0.1}
        known["desk"] = {"corner": [sx - 0.6, 0.3, 0], "size_m": [0.5, 1.2, 0.75]}
        free = (1.9, 0.9, 3.8, 3.2)
        targets = {"bed": np.array([1.0, 2.8, 0.55]), "board": np.array([2.7, 2.0, 0]), "centre": np.array([2.8, 2.0, 0.5])}
    else:
        faces += B(0.3, 0.3, 0, 1.6, 0.8, 0.75, planks, rng, "desk", top=poster)
        faces.append(Face([0.55, 0.32, 1.25], X * 0.6, -Z * 0.38, screen(0.6, 0.38, rng), "monitor"))
        faces += B(0.7, 0.32, 0.75, 0.2, 0.05, 0.12, slats, rng, "monitor_stand")
        faces += B(1.2, 0.9, 0, 0.5, 0.5, 0.45, stripes, rng, "chair", top=fabric)
        faces += B(sx - 0.5, 0.5, 0, 0.4, 2.5, 2.0, books, rng, "shelf")
        faces += B(1.8, sy - 0.7, 0, 1.4, 0.6, 1.1, books, rng, "cabinet", top=planks)
        faces += B(2.0, sy - 0.6, 1.1, 0.35, 0.3, 0.3, poster, rng, "box_on_cabinet")
        faces += B(0.2, sy - 1.2, 0, 0.5, 0.5, 0.6, slats, rng, "bin")
        faces.append(Face([0.003, 2.8, 2.3], -Y * 0 - Y * 1.2, -Z * 1.9, curtain(1.2, 1.9, rng), "curtain"))
        faces.append(Face([2.0, 1.4, eps], X * 1.0, Y * 1.0, checkerboard(1.0, 1.0), "board_1m"))
        known["board_1m"] = {"corner": [2.0, 1.4, eps], "size_m": [1.0, 1.0], "square_m": 0.1}
        known["desk"] = {"corner": [0.3, 0.3, 0], "size_m": [1.6, 0.8, 0.75]}
        free = (1.2, 1.3, 4.0, 3.0)
        targets = {"desk": np.array([1.1, 0.7, 0.75]), "board": np.array([2.5, 1.9, 0]), "centre": np.array([2.6, 2.0, 0.5])}
    lights = np.array([[sx * 0.33, sy * 0.4, sz - 0.05], [sx * 0.7, sy * 0.6, sz - 0.05]])
    return Room(kind, (sx, sy, sz), faces, lights, free, targets, known, boxes)
