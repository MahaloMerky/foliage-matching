"""Binarize at coarse scale, describe crown/gap blobs, and match them."""
from dataclasses import dataclass

import cv2
import numpy as np


def binarize(img: np.ndarray, method: str = "otsu", block: int = 151) -> np.ndarray:
    if method == "otsu":
        _, m = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        m = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, block | 1, 0)
    else:
        raise ValueError(method)
    return m


@dataclass
class Blobs:
    centroids: np.ndarray    # (N, 2) float32, x/y
    descriptors: np.ndarray  # (N, D) float32
    polarity: np.ndarray     # (N,) 1 = canopy, 0 = gap


def _shape_features(stats, labels, idx) -> np.ndarray:
    x, y, w, h, area = stats[idx]
    comp = (labels[y:y + h, x:x + w] == idx).astype(np.uint8)
    hu = cv2.HuMoments(cv2.moments(comp, binaryImage=True)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # log-scale, rotation invariant
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = cv2.contourArea(cv2.convexHull(contours[0])) if contours else area
    solidity = area / max(hull_area, 1)
    return np.concatenate([[np.log(area)], hu[:4], [solidity]])


def extract_blobs(mask: np.ndarray, min_area: int = 80, max_frac: float = 0.15,
                  k_context: int = 6, valid: np.ndarray | None = None) -> Blobs:
    """Connected components of both canopy (white) and gaps (black).

    Descriptor = own shape (log area, Hu moments, solidity) + a rotation-invariant
    context: sorted distances to the k nearest blobs and their log areas. The
    context is what separates otherwise identical-looking repeated blobs.
    """
    max_area = max_frac * mask.size
    cents, shapes, pol = [], [], []
    for polarity, m in ((1, mask), (0, 255 - mask)):
        n, labels, stats, cc = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (min_area <= area <= max_area):
                continue
            cx, cy = cc[i]
            if valid is not None and not valid[int(cy), int(cx)]:
                continue
            x, y, w, h = stats[i, :4]
            if x == 0 or y == 0 or x + w >= mask.shape[1] or y + h >= mask.shape[0]:
                continue  # clipped by the image border: shape is unreliable
            cents.append(cc[i])
            shapes.append(_shape_features(stats, labels, i))
            pol.append(polarity)
    if not cents:
        return Blobs(np.zeros((0, 2), np.float32), np.zeros((0, 6 + 2 * k_context), np.float32),
                     np.zeros(0, np.int8))
    cents = np.asarray(cents, np.float32)
    shapes = np.asarray(shapes, np.float32)
    pol = np.asarray(pol, np.int8)

    # Neighbourhood context, vectorised: pairwise distances in one shot.
    d = np.linalg.norm(cents[:, None] - cents[None], axis=2)
    np.fill_diagonal(d, np.inf)
    k = min(k_context, len(cents) - 1)
    ctx = np.zeros((len(cents), 2 * k_context), np.float32)
    if k > 0:
        nn = np.argsort(d, axis=1)[:, :k]
        nd = np.take_along_axis(d, nn, axis=1)
        scale = np.exp(shapes[:, :1] / 2)  # sqrt(area): makes distances scale invariant
        ctx[:, :k] = np.log1p(nd / scale)
        ctx[:, k_context:k_context + k] = shapes[nn, 0]
    desc = np.hstack([shapes, ctx]).astype(np.float32)
    return Blobs(cents, desc, pol)


def match_blobs(a: Blobs, b: Blobs, ratio: float = 0.85, H_prior=None,
                radius: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbour + ratio test, only within the same polarity.

    With H_prior (A->B pixel mapping from drone GPS/gimbal metadata), a blob may
    only match blobs within `radius` px of where the prior predicts it. That turns
    "find this blob anywhere" into "find it nearby", which is what makes
    repetitive canopy matchable.
    """
    empty = np.zeros((0, 2), np.float32)
    if len(a.centroids) < 2 or len(b.centroids) < 2:
        return empty, empty
    # Standardise features jointly so no single feature dominates the distance.
    allf = np.vstack([a.descriptors, b.descriptors])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    da, db = (a.descriptors - mu) / sd, (b.descriptors - mu) / sd
    D = np.linalg.norm(da[:, None] - db[None], axis=2)
    D[a.polarity[:, None] != b.polarity[None]] = np.inf
    if H_prior is not None:
        pred = cv2.perspectiveTransform(a.centroids[None].astype(np.float64), H_prior)[0]
        geo = np.linalg.norm(pred[:, None] - b.centroids[None], axis=2)
        D[geo > radius] = np.inf
    order = np.argsort(D, axis=1)
    best, second = order[:, 0], order[:, 1]
    rows = np.arange(len(D))
    d1, d2 = D[rows, best], D[rows, second]
    ok = np.isfinite(d1) & ((d1 < ratio * d2) | ~np.isfinite(d2))
    ok &= np.argmin(D, axis=0)[best] == rows  # mutual nearest neighbour
    return a.centroids[ok].astype(np.float32), b.centroids[best[ok]].astype(np.float32)


def multiscale_blob_matches(scales_a: dict, scales_b: dict, method: str = "otsu",
                            valid_a=None, valid_b=None, H_prior=None,
                            radius: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Run blob matching at every blur scale and pool the correspondences."""
    pa, pb = [], []
    for s in scales_a:
        ma, mb = binarize(scales_a[s], method), binarize(scales_b[s], method)
        min_area = int(s * s)  # blobs smaller than the blur kernel are noise
        ba = extract_blobs(ma, min_area=min_area, valid=valid_a)
        bb = extract_blobs(mb, min_area=min_area, valid=valid_b)
        x, y = match_blobs(ba, bb, H_prior=H_prior, radius=radius)
        pa.append(x)
        pb.append(y)
    return np.vstack(pa), np.vstack(pb)


def mser_matches(index_a, index_b, sigma: float = 2.0, mask_a=None, mask_b=None,
                 delta: int = 5, ratio: float = 0.8):
    """MSER = the binary idea done rigorously: threshold at EVERY level and keep only
    regions whose shape stays stable across many thresholds (both canopy and gaps).
    Regions are then described with SIFT on the smoothed vegetation index.
    """
    from .vegetation import smooth
    sa, sb = smooth(index_a, sigma), smooth(index_b, sigma)
    mser = cv2.MSER_create(delta=delta, min_area=60, max_area=int(0.05 * sa.size))
    sift = cv2.SIFT_create()
    feats = []
    for img, m in ((sa, mask_a), (sb, mask_b)):
        kps = mser.detect(img, m)
        kps, desc = sift.compute(img, kps)
        feats.append((kps, desc))
    (ka, da), (kb, db) = feats
    if da is None or db is None or len(ka) < 2 or len(kb) < 2:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    good = [m for m, n in (p for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
                           if len(p) == 2) if m.distance < ratio * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pa, pb
