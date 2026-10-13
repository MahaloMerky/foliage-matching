"""Keypoint matchers: SIFT baseline and optional learned matchers (GPU-accelerated)."""
import cv2
import numpy as np

from .device import torch_device

_EMPTY = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))


def sift_matches(gray_a, gray_b, mask_a=None, mask_b=None, ratio=0.8, nfeatures=8000):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    ka, da = sift.detectAndCompute(gray_a, mask_a)
    kb, db = sift.detectAndCompute(gray_b, mask_b)
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return _EMPTY
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    good = [m for m, n in (p for p in flann.knnMatch(da, db, k=2) if len(p) == 2)
            if m.distance < ratio * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pa, pb


class LearnedMatcher:
    """DISK+LightGlue or LoFTR via kornia. Uses CUDA + fp16 autocast when available.

    Weights download on first use. Images are resized so the long side is
    `max_side` (cost grows with pixel count), and keypoints are scaled back.
    """

    def __init__(self, kind: str = "lightglue", device: str = "auto", max_side: int = 1024,
                 num_features: int = 4096):
        import torch
        import kornia.feature as KF
        self.torch, self.KF = torch, KF
        self.kind, self.max_side, self.num_features = kind, max_side, num_features
        self.device = torch_device(device)
        if kind == "lightglue":
            self.disk = KF.DISK.from_pretrained("depth").eval().to(self.device)
            self.lg = KF.LightGlueMatcher("disk").eval().to(self.device)
        elif kind == "loftr":
            self.loftr = KF.LoFTR(pretrained="outdoor").eval().to(self.device)
        else:
            raise ValueError(kind)

    def _prep(self, bgr):
        h, w = bgr.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        nh, nw = int(h * s) // 8 * 8, int(w * s) // 8 * 8  # LoFTR needs /8
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = w / nw, h / nh
        t = self.torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float()[None] / 255
        return t.to(self.device), np.float32([sx, sy])

    def match(self, bgr_a, bgr_b):
        torch = self.torch
        ta, sa = self._prep(bgr_a)
        tb, sb = self._prep(bgr_b)
        amp = torch.autocast("cuda", dtype=torch.float16) if self.device.type == "cuda" \
            else torch.autocast("cpu", enabled=False)
        with torch.inference_mode(), amp:
            if self.kind == "loftr":
                gray = lambda t: (0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3])
                out = self.loftr({"image0": gray(ta), "image1": gray(tb)})
                ka, kb = out["keypoints0"], out["keypoints1"]
            else:
                fa = self.disk(ta, self.num_features, pad_if_not_divisible=True)[0]
                fb = self.disk(tb, self.num_features, pad_if_not_divisible=True)[0]
                laf = lambda k: self.KF.laf_from_center_scale_ori(
                    k[None], torch.ones(1, len(k), 1, 1, device=self.device))
                hw = lambda t: torch.tensor(t.shape[2:], device=self.device)
                _, idx = self.lg(fa.descriptors, fb.descriptors, laf(fa.keypoints),
                                 laf(fb.keypoints), hw1=hw(ta), hw2=hw(tb))
                ka, kb = fa.keypoints[idx[:, 0]], fb.keypoints[idx[:, 1]]
        pa = ka.float().cpu().numpy() * sa
        pb = kb.float().cpu().numpy() * sb
        return pa.astype(np.float32), pb.astype(np.float32)


def filter_by_mask(pa, pb, mask_a=None, mask_b=None):
    """Keep matches whose endpoints fall inside the given masks (e.g. stable regions)."""
    keep = np.ones(len(pa), bool)
    for pts, m in ((pa, mask_a), (pb, mask_b)):
        if m is None or not len(pts):
            continue
        xi = pts[:, 0].astype(int).clip(0, m.shape[1] - 1)
        yi = pts[:, 1].astype(int).clip(0, m.shape[0] - 1)
        keep &= m[yi, xi] > 0
    return pa[keep], pb[keep]


def gate_by_prior(pa, pb, H_ab, max_err_px):
    """Drop matches far from where the drone-pose prior predicts (GPS-guided matching)."""
    if H_ab is None or not len(pa):
        return pa, pb
    pred = cv2.perspectiveTransform(pa[None], H_ab)[0]
    keep = np.linalg.norm(pred - pb, axis=1) <= max_err_px
    return pa[keep], pb[keep]
