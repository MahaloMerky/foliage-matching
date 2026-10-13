"""Multi-photo consensus: warp neighbours onto a reference, majority-vote, measure agreement."""
import cv2
import numpy as np


def warp_stack(masks: list[np.ndarray], homs: list[np.ndarray], shape: tuple[int, int]):
    """Warp binary masks into the reference frame.

    homs[i] maps pixels of masks[i] -> reference pixels. Returns (stack, valid),
    both (N, H, W) bool, where valid marks pixels each warped image covers.
    """
    h, w = shape
    stack = np.zeros((len(masks), h, w), bool)
    valid = np.zeros_like(stack)
    ones = np.full(masks[0].shape[:2], 255, np.uint8)
    for i, (m, H) in enumerate(zip(masks, homs)):
        stack[i] = cv2.warpPerspective(m, H, (w, h), flags=cv2.INTER_NEAREST) > 127
        valid[i] = cv2.warpPerspective(ones, H, (w, h), flags=cv2.INTER_NEAREST) > 127
    return stack, valid


def consensus(stack: np.ndarray, valid: np.ndarray, min_views: int = 2):
    """Per-pixel majority vote ("most common" value) and agreement fraction.

    agreement = share of covering views that agree with the consensus value.
    1.0 = every photo saw the same thing (rigid); ~0.5 = coin flip (wind/parallax).
    """
    votes = (stack & valid).sum(0)
    count = valid.sum(0)
    cons = votes * 2 > count
    agree = np.where(cons, votes, count - votes) / np.maximum(count, 1)
    agree[count < min_views] = 0.0
    return cons.astype(np.uint8) * 255, agree.astype(np.float32)


def stable_mask(agreement: np.ndarray, thresh: float = 0.8, open_px: int = 5) -> np.ndarray:
    """Binary mask of regions where photos agree; small speckles removed."""
    m = (agreement >= thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k)


def agreement_heatmap(agreement: np.ndarray) -> np.ndarray:
    """Colour visualisation: dark = unstable, bright = stable."""
    return cv2.applyColorMap((agreement * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
