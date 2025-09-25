#!/usr/bin/env python3
"""Plot KITTI-style trajectories from ground truth and VO results.

Example
-------
python source_code/trajectory_plot.py \
    --gt dataset/poses/00.txt --sc results/VO/KITTI/LIGHTGLUESIFT/00.txt
"""
from __future__ import annotations

import argparse
import itertools
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_trajectory(path: str) -> np.ndarray:
    """Load trajectory as an ``(N, 3)`` array.

    Supports KITTI pose format (12 floats per line) and plain XYZ (3 floats).
    """
    points: List[Tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            entries = stripped.split()
            values = [float(entry) for entry in entries]
            if len(values) >= 12:
                points.append((values[3], values[7], values[11]))
            elif len(values) == 3:
                points.append(tuple(values))
            else:
                raise ValueError(
                    f"Unsupported pose format in {path} at line {line_number}: "
                    f"expected 3 or 12 floats, got {len(values)}"
                )
    if not points:
        raise ValueError(f"No trajectory data found in {path}")
    return np.asarray(points)


def _format_gt_label(gt_path: str) -> str:
    stem = os.path.splitext(os.path.basename(gt_path))[0]
    return stem or os.path.basename(os.path.dirname(gt_path)) or gt_path


def _format_sc_label(sc_path: str) -> str:
    parent = os.path.basename(os.path.dirname(sc_path))
    return parent or os.path.splitext(os.path.basename(sc_path))[0] or sc_path


def _normalise_lengths(gt: np.ndarray, sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if gt.shape[0] == sc.shape[0]:
        return gt, sc
    min_len = min(gt.shape[0], sc.shape[0])
    if min_len == 0:
        raise ValueError("Cannot align empty trajectories")
    if gt.shape[0] != min_len:
        gt = gt[:min_len]
    if sc.shape[0] != min_len:
        sc = sc[:min_len]
    return gt, sc


def plot_trajectories(pairs: Sequence[Tuple[str, str]], output: str | None = None, show: bool = True) -> None:
    if not pairs:
        raise ValueError("At least one pair of --gt/--sc paths is required")

    fig, ax = plt.subplots(figsize=(8, 6))
    colour_cycle = itertools.cycle(plt.get_cmap("tab20").colors)
    gt_cache: Dict[str, np.ndarray] = {}
    sc_cache: Dict[str, np.ndarray] = {}
    plotted_gt: Dict[str, bool] = {}

    for gt_path, sc_path in pairs:
        gt_traj = gt_cache.setdefault(gt_path, _load_trajectory(gt_path))
        sc_traj = sc_cache.setdefault(sc_path, _load_trajectory(sc_path))
        gt_traj, sc_traj = _normalise_lengths(gt_traj, sc_traj)

        gt_label = _format_gt_label(gt_path)
        sc_label = _format_sc_label(sc_path)

        if gt_path not in plotted_gt:
            gt_colour = next(colour_cycle)
            ax.plot(
                gt_traj[:, 0],
                gt_traj[:, 2],
                label=f"Ground Truth {gt_label}",
                color=gt_colour,
                linewidth=2.0,
            )
            plotted_gt[gt_path] = True

        sc_colour = next(colour_cycle)
        ax.plot(
            sc_traj[:, 0],
            sc_traj[:, 2],
            label=sc_label,
            color=sc_colour,
            linestyle="--",
            linewidth=1.5,
        )

    ax.set_title("Trajectory Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.axis("equal")
    ax.legend()

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectories from ground truth and VO results.")
    parser.add_argument("--gt", nargs="+", dest="gt_paths", help="Ground truth pose files (KITTI format).")
    parser.add_argument("--sc", nargs="+", dest="sc_paths", help="Estimated trajectory files.")
    parser.add_argument("--output", "-o", help="Path to save the generated plot.")
    parser.add_argument("--no-show", action="store_true", help="Skip opening the interactive figure window.")
    args = parser.parse_args(argv)

    gt_count = len(args.gt_paths)
    sc_count = len(args.sc_paths)
    if not (gt_count == sc_count or gt_count == 1 or sc_count == 1):
        raise SystemExit("--gt and --sc must be provided the same number of times or one of them must be a single file")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    if len(args.gt_paths) == len(args.sc_paths):
        pairs = list(zip(args.gt_paths, args.sc_paths))
    elif len(args.gt_paths) == 1:
        pairs = [(args.gt_paths[0], sc_path) for sc_path in args.sc_paths]
    else:
        pairs = [(gt_path, args.sc_paths[0]) for gt_path in args.gt_paths]
    show = not args.no_show
    plot_trajectories(pairs, output=args.output, show=show)


if __name__ == "__main__":
    main()
