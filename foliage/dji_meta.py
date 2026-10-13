"""Read DJI XMP/EXIF pose metadata and build a rough ground-plane homography between photos.

Assumes nadir (straight-down) photos. Good enough as a starting alignment for
consensus stacking and for gating matches; not a substitute for bundle adjustment.
"""
import math
import re
import warnings

import numpy as np
from PIL import Image

_ATTR = re.compile(rb'drone-dji:(\w+)="([^"]*)"')
_ELEM = re.compile(rb"<drone-dji:(\w+)>([^<]*)</drone-dji:\1>")
_EARTH_R = 6378137.0


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dms(v, ref):
    d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -d if ref in ("S", "W") else d


def read_dji_meta(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    xmp = {k.decode(): v.decode() for k, v in _ATTR.findall(raw) + _ELEM.findall(raw)}
    img = Image.open(path)
    exif = img.getexif()
    gps = exif.get_ifd(0x8825)
    sub = exif.get_ifd(0x8769)
    meta = {
        "width": img.width,
        "height": img.height,
        "lat": _num(xmp.get("GpsLatitude")),
        "lon": _num(xmp.get("GpsLongitude") or xmp.get("GpsLongtitude")),  # DJI typo exists
        "rel_alt": _num(xmp.get("RelativeAltitude")),
        "yaw": _num(xmp.get("GimbalYawDegree") or xmp.get("FlightYawDegree")),
        "pitch": _num(xmp.get("GimbalPitchDegree")),
        "f35": _num(sub.get(0xA405)),  # FocalLengthIn35mmFilm
        "rtk": xmp.get("RtkFlag"),
    }
    if meta["lat"] is None and gps:
        meta["lat"] = _dms(gps[2], gps.get(1))
        meta["lon"] = _dms(gps[4], gps.get(3))
    return meta


def pixel_to_world(meta: dict, origin: tuple[float, float], canopy_height: float = 0.0):
    """3x3 matrix: image pixel -> local (east, north) metres on a plane at canopy_height."""
    if meta["pitch"] is not None and abs(meta["pitch"] + 90) > 10:
        warnings.warn(f"gimbal pitch {meta['pitch']} is not nadir; prior will be poor")
    w, h = meta["width"], meta["height"]
    f35 = meta["f35"] or 24.0
    f_px = f35 / 36.0 * w
    gsd = max(meta["rel_alt"] - canopy_height, 1.0) / f_px  # metres per pixel
    lat0, lon0 = origin
    east = math.radians(meta["lon"] - lon0) * _EARTH_R * math.cos(math.radians(lat0))
    north = math.radians(meta["lat"] - lat0) * _EARTH_R
    psi = math.radians(meta["yaw"] or 0.0)
    r = (math.cos(psi), -math.sin(psi))   # image-right direction in (E, N)
    u = (math.sin(psi), math.cos(psi))    # image-up (heading) direction
    L = np.array([[1, 0, -w / 2], [0, -1, h / 2], [0, 0, 1]])
    S = np.diag([gsd, gsd, 1.0])
    R = np.array([[r[0], u[0], 0], [r[1], u[1], 0], [0, 0, 1]])
    T = np.array([[1, 0, east], [0, 1, north], [0, 0, 1]])
    return T @ R @ S @ L


def prior_homography(meta_a: dict, meta_b: dict, canopy_height: float = 0.0) -> np.ndarray:
    """Pixel in A -> predicted pixel in B, via the shared ground/canopy plane."""
    origin = (meta_a["lat"], meta_a["lon"])
    Pa = pixel_to_world(meta_a, origin, canopy_height)
    Pb = pixel_to_world(meta_b, origin, canopy_height)
    return np.linalg.inv(Pb) @ Pa


def scale_homography(H: np.ndarray, s: float) -> np.ndarray:
    """Adapt a full-resolution homography to images resized by factor s."""
    S = np.diag([s, s, 1.0])
    return S @ H @ np.linalg.inv(S)
