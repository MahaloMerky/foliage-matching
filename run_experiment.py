#!/usr/bin/env python3
"""Compare matching methods on overlapping foliage photos.

  python run_experiment.py                       # synthetic canopy (no photos needed)
  python run_experiment.py --wind 3              # windier synthetic day
  python run_experiment.py --images DIR          # real DJI photos, sorted by filename
  python run_experiment.py --methods sift,blobs,consensus_sift,lightglue

Writes results.csv, match visualisations and agreement maps to --out.
"""
import argparse
import csv
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from foliage import blobs, consensus, device, matchers, verify, vegetation

ALL_METHODS = ["sift", "blobs", "mser", "consensus_sift", "consensus_blobs", "consensus_mser",
               "lightglue", "loftr", "consensus_lightglue", "consensus_loftr"]


def load_real(folder, max_side, canopy_height):
    from foliage import dji_meta
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    if len(paths) < 2:
        raise SystemExit(f"need at least 2 images in {folder}")
    imgs, metas, scales = [], [], []
    for p in paths:
        im = cv2.imread(p)
        s = min(1.0, max_side / max(im.shape[:2]))
        imgs.append(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im)
        scales.append(s)
        try:
            metas.append(dji_meta.read_dji_meta(p))
        except Exception as e:  # missing/odd metadata: fall back to no prior
            print(f"  no usable metadata in {os.path.basename(p)}: {e}")
            metas.append(None)

    def prior(i, j):
        if metas[i] is None or metas[j] is None or None in (metas[i]["lat"], metas[j]["lat"],
                                                            metas[i]["rel_alt"]):
            return None
        H = dji_meta.prior_homography(metas[i], metas[j], canopy_height)
        return dji_meta.scale_homography(H, scales[i])

    names = [os.path.basename(p) for p in paths]
    return imgs, prior, None, names


def load_synthetic(args):
    from foliage.synthetic import CanopyScene
    print("Rendering synthetic canopy...")
    scene = CanopyScene(n_crowns=args.crowns, seed=args.seed)
    views = scene.flight_strip(n_views=args.views, overlap=args.overlap, wind_px=args.wind,
                               gps_sigma=args.gps_sigma, seed=args.seed)
    plane = 18.0  # assumed canopy-top height for the metadata prior
    prior = lambda i, j: scene.prior_homography(views[i], views[j], plane)
    gt = lambda i, j: scene.gt_fn(views[i], views[j])
    return [v.image for v in views], prior, gt, [f"synthetic_{i}" for i in range(len(views))]


def agreement_maps(masks, prior, radius=2):
    """For each view, vote with its neighbours warped in via the metadata prior."""
    out = []
    for i, m in enumerate(masks):
        nb = [j for j in range(max(0, i - radius), min(len(masks), i + radius + 1))]
        homs = [np.eye(3) if j == i else prior(j, i) for j in nb]
        if any(H is None for H in homs):
            out.append(np.ones(m.shape, np.float32))  # no prior: treat all as stable
            continue
        stack, valid = consensus.warp_stack([masks[j] for j in nb], homs, m.shape)
        out.append(consensus.consensus(stack, valid)[1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", help="folder of overlapping photos (omit for synthetic)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--methods", default="sift,blobs,consensus_sift,consensus_blobs,mser")
    ap.add_argument("--max-side", type=int, default=2000, help="resize real photos for speed")
    ap.add_argument("--canopy-height", type=float, default=0.0,
                    help="metres above takeoff of the plane used for the pose prior")
    ap.add_argument("--sigmas", default="3,6,12", help="blur scales for binary blobs")
    ap.add_argument("--consensus-sigma", type=float, default=6)
    ap.add_argument("--agree", type=float, default=0.8, help="agreement threshold for stable mask")
    ap.add_argument("--prior-gate", type=float, default=0,
                    help="drop matches > N px from the pose prediction (0 = off)")
    ap.add_argument("--search-radius", type=float, default=60,
                    help="blob matching: max px from the pose-predicted location")
    ap.add_argument("--device", default="auto", help="cuda / cpu / mps / auto")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    # synthetic options
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--overlap", type=float, default=0.75)
    ap.add_argument("--wind", type=float, default=1.5, help="leaf motion std (px)")
    ap.add_argument("--gps-sigma", type=float, default=3.0)
    ap.add_argument("--crowns", type=int, default=260, help="tree count (density)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(device.describe())
    os.makedirs(args.out, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",")]
    for m in methods:
        if m not in ALL_METHODS:
            raise SystemExit(f"unknown method {m}; choose from {ALL_METHODS}")

    imgs, prior, gt, names = (load_real(args.images, args.max_side, args.canopy_height)
                              if args.images else load_synthetic(args))
    for i, im in enumerate(imgs[:3]):
        cv2.imwrite(os.path.join(args.out, f"view_{i}.jpg"), im)

    t0 = time.perf_counter()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    exg = [vegetation.excess_green(im) for im in imgs]
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    pyr = [vegetation.scale_space(e, sigmas) for e in exg]

    stable = [None] * len(imgs)
    if any(m.startswith("consensus") for m in methods):
        cmasks = [blobs.binarize(vegetation.smooth(e, args.consensus_sigma)) for e in exg]
        agrees = agreement_maps(cmasks, prior)
        stable = [consensus.stable_mask(a, args.agree) for a in agrees]
        for i in range(min(3, len(imgs))):
            cv2.imwrite(os.path.join(args.out, f"agreement_{i}.png"),
                        consensus.agreement_heatmap(agrees[i]))
            cv2.imwrite(os.path.join(args.out, f"binary_{i}.png"), cmasks[i])
            print(f"  view {i}: {100 * (stable[i] > 0).mean():.0f}% of pixels stable")
    print(f"Preprocessing: {time.perf_counter() - t0:.1f}s")

    learned = {}
    for k in ("lightglue", "loftr"):
        if k in methods or f"consensus_{k}" in methods:
            try:
                learned[k] = matchers.LearnedMatcher(k, args.device)
            except Exception as e:
                print(f"  skipping {k}: {e} (pip install -r requirements-gpu.txt)")
    methods = [m for m in methods
               if m.replace("consensus_", "") not in ("lightglue", "loftr")
               or m.replace("consensus_", "") in learned]

    def run(method, i, j):
        t = time.perf_counter()
        if method == "sift":
            pa, pb = matchers.sift_matches(grays[i], grays[j])
        elif method == "consensus_sift":
            pa, pb = matchers.sift_matches(grays[i], grays[j], stable[i], stable[j])
        elif method == "blobs":
            pa, pb = blobs.multiscale_blob_matches(pyr[i], pyr[j], H_prior=prior(i, j),
                                                   radius=args.search_radius)
        elif method == "mser":
            pa, pb = blobs.mser_matches(exg[i], exg[j])
        elif method == "consensus_mser":
            pa, pb = blobs.mser_matches(exg[i], exg[j], mask_a=stable[i], mask_b=stable[j])
        elif method == "consensus_blobs":
            pa, pb = blobs.multiscale_blob_matches(pyr[i], pyr[j], valid_a=stable[i] > 0,
                                                   valid_b=stable[j] > 0, H_prior=prior(i, j),
                                                   radius=args.search_radius)
        elif method.startswith("consensus_"):  # learned matcher, then keep stable regions only
            pa, pb = learned[method.split("_", 1)[1]].match(imgs[i], imgs[j])
            pa, pb = matchers.filter_by_mask(pa, pb, stable[i], stable[j])
        else:
            pa, pb = learned[method].match(imgs[i], imgs[j])
        if args.prior_gate > 0:
            pa, pb = matchers.gate_by_prior(pa, pb, prior(i, j), args.prior_gate)
        inl = verify.ransac_inliers(pa, pb)
        row = {"pair": f"{names[i]}->{names[j]}", "method": method, "matches": len(pa),
               "inliers": int(inl.sum()), "inlier_ratio": round(inl.mean(), 3) if len(pa) else 0,
               "coverage": round(verify.coverage(pa[inl], imgs[i].shape), 3),
               "seconds": round(time.perf_counter() - t, 2)}
        if gt is not None:
            ok = verify.gt_correct(pa[inl], pb[inl], gt(i, j))
            row["correct_inliers"] = int(ok.sum())
            row["precision"] = round(ok.mean(), 3) if len(ok) else 0
        if i == 0:
            cv2.imwrite(os.path.join(args.out, f"matches_{method}.jpg"),
                        verify.draw_matches(imgs[i], imgs[j], pa, pb, inl))
        return row

    pairs = [(i, i + 1) for i in range(len(imgs) - 1)]
    jobs = [(m, i, j) for m in methods for i, j in pairs]
    # CPU methods run in parallel threads (OpenCV releases the GIL);
    # learned matchers run sequentially to share one GPU.
    is_learned = lambda m: m.replace("consensus_", "") in learned
    cpu_jobs = [jb for jb in jobs if not is_learned(jb[0])]
    gpu_jobs = [jb for jb in jobs if is_learned(jb[0])]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(lambda jb: run(*jb), cpu_jobs))
    rows += [run(*jb) for jb in gpu_jobs]

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\nMean over {len(pairs)} pairs" + (" (synthetic ground truth available)" if gt else ""))
    cols = [k for k in keys if k not in ("pair", "method")]
    print(f"{'method':<17}" + "".join(f"{c:>16}" for c in cols))
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        print(f"{m:<17}" + "".join(f"{np.mean([x[c] for x in r]):>16.3f}" for c in cols))
    print(f"\nOutputs in {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
