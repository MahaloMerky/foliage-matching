"""Vegetation index and anti-aliased scale pyramid."""
import cv2
import numpy as np


def excess_green(bgr: np.ndarray) -> np.ndarray:
    """Excess Green index 2g - r - b on chromaticity-normalised RGB, scaled to uint8.

    More stable across photos than brightness because it is normalised by
    total intensity (partly cancels exposure and shading changes).
    """
    img = bgr.astype(np.float32)
    s = img.sum(axis=2) + 1e-6
    b, g, r = img[..., 0] / s, img[..., 1] / s, img[..., 2] / s
    exg = 2 * g - r - b  # range roughly [-1, 2]
    return cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI for multispectral drones (e.g. Mavic 3M), scaled to uint8."""
    nir, red = nir.astype(np.float32), red.astype(np.float32)
    v = (nir - red) / (nir + red + 1e-6)
    return ((v + 1) * 127.5).clip(0, 255).astype(np.uint8)


def smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; runs through OpenCL (GPU) via UMat when enabled."""
    if sigma <= 0:
        return img
    if cv2.ocl.useOpenCL():
        return cv2.GaussianBlur(cv2.UMat(img), (0, 0), sigma).get()
    return cv2.GaussianBlur(img, (0, 0), sigma)


def scale_space(index_img: np.ndarray, sigmas=(2, 4, 8, 16)) -> dict[float, np.ndarray]:
    """Blur at several scales at full resolution.

    Blurring before any downsampling is the anti-aliasing step: it removes
    leaf-scale texture that would otherwise alias into moire patterns that
    differ between photos.
    """
    return {s: smooth(index_img, s) for s in sigmas}
