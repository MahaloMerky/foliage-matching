"""Synthetic layered canopy with parallax and wind, for testing without real photos.

Geometry (nadir camera at altitude A over flat ground, units = ground pixels):
a point at height h with ground position X projects to  (X - cam) * s(h) + c,
where s(h) = A / (A - h). Taller things move more between photos (parallax),
and the true A->B mapping for a point at height h is  pA + (camA - camB) * s(h).
Wind displaces leaves per photo and is deliberately NOT in the ground truth.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class View:
    image: np.ndarray       # BGR uint8
    height: np.ndarray      # float32 surface height per pixel (ground truth)
    cam: np.ndarray         # true camera position (x, y)
    cam_gps: np.ndarray     # noisy position, as GPS would report it


class CanopyScene:
    def __init__(self, world=(3000, 1100), altitude=100.0, n_crowns=260, tree_height=(8, 28),
                 crown_radius=(40, 100), leaves_per_crown=600, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W, self.H = world
        self.A = altitude
        rng = self.rng
        # Ground: low-frequency soil/grass texture plus a few rigid rocks.
        noise = rng.normal(0, 1, (self.H // 16 + 2, self.W // 16 + 2)).astype(np.float32)
        base = cv2.resize(noise, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        fine = cv2.GaussianBlur(rng.normal(0, 1, (self.H, self.W)).astype(np.float32), (0, 0), 1.5)
        g = (base * 18 + fine * 10)
        self.ground = np.dstack([45 + g * 0.6, 70 + g, 85 + g]).clip(0, 255).astype(np.uint8)
        for _ in range(60):
            c = (int(rng.uniform(0, self.W)), int(rng.uniform(0, self.H)))
            cv2.circle(self.ground, c, int(rng.uniform(3, 8)), (120, 125, 130), -1, cv2.LINE_AA)

        # Crowns -> leaves (x, y, height, half-axes, angle, colour, crown id).
        cx = rng.uniform(0, self.W, n_crowns)
        cy = rng.uniform(0, self.H, n_crowns)
        R = rng.uniform(*crown_radius, n_crowns)
        top = rng.uniform(*tree_height, n_crowns)
        leaves = []
        for i in range(n_crowns):
            n = leaves_per_crown
            r = R[i] * np.sqrt(rng.uniform(0, 1, n))
            t = rng.uniform(0, 2 * np.pi, n)
            dome = 1 - 0.45 * (r / R[i]) ** 2  # crowns are domes: centre highest
            h = top[i] * dome * rng.uniform(0.8, 1.0, n)
            shade = 0.45 + 0.55 * (h / top[i])  # lower leaves are in shadow
            base_col = rng.uniform([25, 85, 30], [60, 150, 75])
            col = (base_col * rng.uniform(0.85, 1.15, (n, 3))) * shade[:, None]
            leaves.append(np.column_stack([
                cx[i] + r * np.cos(t), cy[i] + r * np.sin(t), h,
                rng.uniform(3, 7, n), rng.uniform(1.5, 3.5, n), rng.uniform(0, 180, n),
                col, np.full(n, i)]))
        L = np.vstack(leaves)
        self.leaves = L[np.argsort(L[:, 2])]  # draw low -> high (painter's algorithm)
        self.n_crowns = n_crowns

    def s(self, h):
        return self.A / (self.A - h)

    def render(self, cam, size=(800, 600), wind_px=1.5, gps_sigma=3.0, seed=None) -> View:
        rng = np.random.default_rng(seed)
        w, h = size
        c = np.array([w / 2, h / 2])
        cam = np.asarray(cam, float)
        # Ground: subpixel translation of the world texture.
        M = np.float32([[1, 0, c[0] - cam[0]], [0, 1, c[1] - cam[1]]])
        img = cv2.warpAffine(self.ground, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        hmap = np.zeros((h, w), np.float32)

        L = self.leaves
        sc = self.s(L[:, 2])
        crown_sway = rng.normal(0, wind_px * 1.5, (self.n_crowns, 2))
        jitter = rng.normal(0, wind_px, (len(L), 2)) + crown_sway[L[:, 9].astype(int)]
        px = (L[:, :2] - cam) * sc[:, None] + c + jitter * sc[:, None]
        vis = (px[:, 0] > -20) & (px[:, 0] < w + 20) & (px[:, 1] > -20) & (px[:, 1] < h + 20)
        sh = 4  # fixed-point subpixel drawing
        for (x, y), ax, ay, ang, b, g, r, hh, s in zip(
                px[vis], L[vis, 3], L[vis, 4], L[vis, 5] + rng.normal(0, 8, vis.sum()),
                L[vis, 6], L[vis, 7], L[vis, 8], L[vis, 2], sc[vis]):
            ctr = (int(x * 16), int(y * 16))
            axes = (int(ax * s * 16), int(ay * s * 16))
            cv2.ellipse(img, ctr, axes, ang, 0, 360, (b, g, r), -1, cv2.LINE_AA, sh)
            cv2.ellipse(hmap, ctr, axes, ang, 0, 360, float(hh), -1, cv2.LINE_8, sh)

        gain = rng.uniform(0.9, 1.1)  # exposure change between shots
        img = (img.astype(np.float32) * gain + rng.normal(0, 3, img.shape)).clip(0, 255)
        gps = cam + rng.normal(0, gps_sigma, 2)
        return View(img.astype(np.uint8), hmap, cam, gps)

    def flight_strip(self, n_views=5, size=(800, 600), overlap=0.75, wind_px=1.5,
                     gps_sigma=3.0, seed=0) -> list[View]:
        step = (1 - overlap) * size[0]
        x0 = size[0] / 2 + 60
        y = self.H / 2
        return [self.render((x0 + i * step, y + 4 * np.sin(i)), size, wind_px, gps_sigma,
                            seed=seed * 1000 + i) for i in range(n_views)]

    def gt_fn(self, va: View, vb: View):
        """True A->B mapping using A's height map (ignores wind by design)."""
        def f(pa):
            xi = pa[:, 0].astype(int).clip(0, va.height.shape[1] - 1)
            yi = pa[:, 1].astype(int).clip(0, va.height.shape[0] - 1)
            s = self.s(va.height[yi, xi])[:, None]
            return pa + (va.cam - vb.cam) * s
        return f

    def prior_homography(self, va: View, vb: View, plane_height: float) -> np.ndarray:
        """What drone metadata gives you: a translation from noisy GPS at an assumed plane height."""
        d = (va.cam_gps - vb.cam_gps) * self.s(plane_height)
        return np.array([[1, 0, d[0]], [0, 1, d[1]], [0, 0, 1]], np.float64)
