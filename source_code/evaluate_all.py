#!/usr/bin/env python3
"""
Unified evaluator for KITTI and EuRoC datasets (Sim(3) alignment for EuRoC)

Generates:
    - kitti_metrics.csv
    - euroc_metrics.csv
"""

import os
import csv
import itertools
import numpy as np

# --- Paths ---
KITTI_GT_DIR = "/media/shared/KITTI/poses/"
KITTI_RESULTS = "/home/ws-rtx/Documents/Projects/pyslam/results_kitti"

EUROC_GT_CSV = lambda seq: f"/media/shared/euroc/{seq}/mav0/state_groundtruth_estimate0/data.csv"
EUROC_TS = lambda seq: f"/media/shared/euroc/{seq}/mav0/cam0/data.csv"
EUROC_RESULTS = "/home/ws-rtx/Documents/Projects/pyslam/results_euroc"

KITTI_CSV = "kitti_metrics.csv"
EUROC_CSV = "euroc_metrics.csv"

KITTI_SEQUENCES = [f"{i:02d}" for i in range(11) if i != 3]
EUROC_SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult"
]

# --- Sim(3) alignment ---
def umeyama(src, dst, estimate_scale=True):
    μs, μd = src.mean(0), dst.mean(0)
    src_c, dst_c = src - μs, dst - μd
    U, S, Vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((src_c**2).sum() / len(src)) if estimate_scale else 1.0
    t = μd - s * R @ μs
    aligned = (s * (R @ src.T)).T + t
    return aligned, s, R, t

def all_axis_maps():
    for p in itertools.permutations([0, 1, 2]):
        for sgn in itertools.product([-1, 1], repeat=3):
            M = np.zeros((3, 3))
            for r, (ax, s) in enumerate(zip(p, sgn)):
                M[r, ax] = s
            yield M

def best_sim3_alignment(vo, gt):
    best_err, best_aligned = np.inf, None
    for M in all_axis_maps():
        vo_m = (M @ vo.T).T
        aligned, *_ = umeyama(vo_m, gt, True)
        err = np.sqrt(np.mean(np.linalg.norm(aligned - gt, axis=1) ** 2))
        if err < best_err:
            best_err, best_aligned = err, aligned
    return best_aligned

# --- Evaluation metrics ---
def adjust_lengths(gt, vo):
    if gt is None or vo is None:
        return None, None
    n = min(len(gt), len(vo))
    return gt[:n], vo[:n]

def compute_rpe_xyz(gt_xyz, est_xyz, delta=1):
    if len(gt_xyz) <= delta:
        return np.nan
    diff = (gt_xyz[delta:] - gt_xyz[:-delta]) - (est_xyz[delta:] - est_xyz[:-delta])
    return np.mean(np.linalg.norm(diff, axis=1))

def compute_metrics(gt_xyz, vo_xyz, align=False):
    gt_xyz, vo_xyz = adjust_lengths(gt_xyz, vo_xyz)
    if gt_xyz is None or vo_xyz is None or len(gt_xyz) == 0:
        return None
    if align:
        vo_xyz = best_sim3_alignment(vo_xyz, gt_xyz)

    d = np.linalg.norm(gt_xyz - vo_xyz, axis=1)
    ate = np.sqrt(np.mean(d ** 2))
    mae = np.mean(d)
    mse = np.mean(d ** 2)

    gt_norm = np.linalg.norm(gt_xyz, axis=1)
    valid = gt_norm > 1e-6
    mre = np.mean(d[valid] / gt_norm[valid]) if np.any(valid) else np.nan

    rpe = compute_rpe_xyz(gt_xyz, vo_xyz)
    fde = np.linalg.norm(gt_xyz[-1] - vo_xyz[-1])
    return ate, mae, mse, mre, rpe, fde

# --- File loading ---
def load_kitti_gt(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        lines = [l.split() for l in f if len(l.split()) == 12]
    return np.array(lines, dtype=np.float32)[:, [3, 7, 11]] if lines else None

def load_xyz(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        lines = [l.split() for l in f if len(l.split()) == 3]
    return np.array(lines, dtype=np.float32) if lines else None

def load_gt_from_csv(seq):
    path = EUROC_GT_CSV(seq)
    if not os.path.exists(path): return None
    data = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    ts = float(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    data.append([ts, x, y, z])
                except ValueError:
                    continue
    arr = np.array(data)
    if arr[:, 0].max() > 1e12: arr[:, 0] *= 1e-9
    return arr

def load_ts(path):
    if not os.path.exists(path): return None
    ts = np.loadtxt(path, delimiter=',', skiprows=1, usecols=0)
    if ts.max() > 1e12: ts *= 1e-9
    return ts

def match_gt_to_ts(gt_csv, img_ts, tol=0.03):
    gt_ts, gt_xyz = gt_csv[:, 0], gt_csv[:, 1:4]
    idx = np.searchsorted(gt_ts, img_ts).clip(1, len(gt_ts) - 1)
    left = idx - 1
    nearest = np.where(np.abs(img_ts - gt_ts[left]) < np.abs(img_ts - gt_ts[idx]), left, idx)
    dt = np.abs(img_ts - gt_ts[nearest])
    keep = dt <= tol
    return gt_xyz[nearest[keep]], keep

# --- Evaluators ---
def evaluate_kitti(gt_dir, base_dir):
    rows = []
    for tracker in sorted(os.listdir(base_dir)):
        tracker_dir = os.path.join(base_dir, tracker)
        if not os.path.isdir(tracker_dir): continue
        tracker_metrics = []
        for seq in KITTI_SEQUENCES:
            gt = load_kitti_gt(os.path.join(gt_dir, f"{seq}.txt"))
            vo = load_xyz(os.path.join(tracker_dir, f"{seq}.txt"))
            metrics = compute_metrics(gt, vo, align=False)
            if metrics:
                tracker_metrics.append([tracker, seq] + list(metrics))
                rows.append(tracker_metrics[-1])
        if tracker_metrics:
            avg = np.mean(np.array(tracker_metrics)[:, 2:].astype(np.float32), axis=0)
            rows.append([tracker, "AVG"] + list(avg))
    return rows

def evaluate_euroc(ts_path_fn, base_dir):
    rows = []
    for tracker in sorted(os.listdir(base_dir)):
        tracker_dir = os.path.join(base_dir, tracker)
        if not os.path.isdir(tracker_dir): continue
        tracker_metrics = []
        for seq in EUROC_SEQUENCES:
            gt_csv = load_gt_from_csv(seq)
            ts = load_ts(ts_path_fn(seq))
            vo = load_xyz(os.path.join(tracker_dir, f"{seq}.txt"))
            if any(x is None for x in (gt_csv, ts, vo)):
                continue
            if len(vo) != len(ts):
                n = min(len(vo), len(ts))
                vo, ts = vo[:n], ts[:n]
            gt_xyz, keep = match_gt_to_ts(gt_csv, ts)
            vo_xyz = vo[keep]
            if len(vo_xyz) < 10: continue
            metrics = compute_metrics(gt_xyz, vo_xyz, align=True)
            if metrics:
                tracker_metrics.append([tracker, seq] + list(metrics))
                rows.append(tracker_metrics[-1])
        if tracker_metrics:
            avg = np.mean(np.array(tracker_metrics)[:, 2:].astype(np.float32), axis=0)
            rows.append([tracker, "AVG"] + list(avg))
    return rows

# --- CSV Writer ---
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tracker", "sequence", "ATE", "MAE", "MSE", "MRE", "RPE", "FDE"])
        writer.writerows(rows)

# --- Main ---
if __name__ == "__main__":
    kitti_rows = evaluate_kitti(KITTI_GT_DIR, KITTI_RESULTS)
    euroc_rows = evaluate_euroc(EUROC_TS, EUROC_RESULTS)
    write_csv(KITTI_CSV, kitti_rows)
    write_csv(EUROC_CSV, euroc_rows)
    print("Results written to:")
    print(f"  • {KITTI_CSV}")
    print(f"  • {EUROC_CSV}")
