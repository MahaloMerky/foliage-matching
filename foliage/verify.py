"""Geometric verification and evaluation metrics."""
import cv2
import numpy as np

_USAC = getattr(cv2, "USAC_MAGSAC", None)


def ransac_inliers(pa, pb, model: str = "fundamental", thresh: float = 1.5) -> np.ndarray:
    """Boolean inlier mask. Fundamental matrix handles parallax; homography assumes a plane."""
    if len(pa) < 8:
        return np.zeros(len(pa), bool)
    if model == "fundamental":
        _, m = cv2.findFundamentalMat(pa, pb, _USAC if _USAC is not None else cv2.FM_RANSAC,
                                      thresh, 0.999, 10000)
    else:
        _, m = cv2.findHomography(pa, pb, _USAC if _USAC is not None else cv2.RANSAC,
                                  thresh * 2, maxIters=10000, confidence=0.999)
    return np.zeros(len(pa), bool) if m is None else m.ravel().astype(bool)


def coverage(pts: np.ndarray, shape, grid: int = 8) -> float:
    """Fraction of grid cells containing at least one point (spread across the image)."""
    if not len(pts):
        return 0.0
    h, w = shape[:2]
    gx = (pts[:, 0] / w * grid).astype(int).clip(0, grid - 1)
    gy = (pts[:, 1] / h * grid).astype(int).clip(0, grid - 1)
    return len(set(zip(gx.tolist(), gy.tolist()))) / grid ** 2


def gt_correct(pa, pb, gt_fn, tol: float = 5.0) -> np.ndarray:
    """Ground-truth check (synthetic data only): gt_fn(pa) -> expected pb."""
    if not len(pa):
        return np.zeros(0, bool)
    return np.linalg.norm(gt_fn(pa) - pb, axis=1) <= tol


def draw_matches(a, b, pa, pb, inliers, max_draw: int = 400):
    h = max(a.shape[0], b.shape[0])
    canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b
    idx = np.random.default_rng(0).permutation(len(pa))[:max_draw]
    for i in idx:
        col = (60, 220, 60) if inliers[i] else (60, 60, 230)  # green inlier, red outlier
        p = tuple(int(v) for v in pa[i])
        q = (int(pb[i, 0]) + a.shape[1], int(pb[i, 1]))
        cv2.line(canvas, p, q, col, 1, cv2.LINE_AA)
    return canvas
