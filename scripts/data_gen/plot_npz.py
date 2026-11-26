"""Headless visual check of RiEMann npz files: three orthographic views + label axes -> PNG.

    python scripts/data_gen/plot_npz.py data/mug/place/demo.npz --traj 0 --frames 0 19 --out place_demo.png
    python scripts/data_gen/plot_npz.py data/mug/place/newpose.npz --out place_newpose.png

Label axes are drawn at seg_center: x red, y green, z blue (5 cm). For place files the EE target
(label A, ee_target_*) is drawn with dashed axes when present.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VIEWS = [(0, 1, "top (x-y)"), (0, 2, "side (x-z)"), (1, 2, "front (y-z)")]


def draw(ax, xyz, rgb, poses, i, j, title, zoom=None):
    m = np.ones(len(xyz), bool)
    if zoom is not None:
        m = np.linalg.norm(xyz - zoom[0], axis=1) < zoom[1]
    ax.scatter(xyz[m, i], xyz[m, j], c=np.clip(rgb[m], 0, 1), s=1.5 if zoom is None else 6, linewidths=0)
    for c, axes, style in poses:
        R = axes.reshape(3, 3).T
        for k, col in enumerate("rgb"):
            e = c + 0.05 * R[:, k]
            ax.plot([c[i], e[i]], [c[j], e[j]], col, ls=style, lw=2)
        ax.plot(c[i], c[j], "k+", ms=10)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="*", default=[0, 19])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = np.load(a.path, allow_pickle=True)
    rows = []
    if d["xyz"].ndim == 4:
        for f in a.frames:
            poses = [(d["seg_center"][a.traj, f], d["axes"][a.traj, f], "-")]
            if "ee_target_pos" in d:
                poses.append((d["ee_target_pos"][a.traj, f], d["ee_target_axes"][a.traj, f], "--"))
            rows.append((d["xyz"][a.traj, f], d["rgb"][a.traj, f], poses, f"traj {a.traj} frame {f} (EE frame)"))
    else:
        poses = [(d["seg_center"], d["axes"], "-")] if "seg_center" in d else []
        if "ee_target_pos" in d:
            poses.append((d["ee_target_pos"], d["ee_target_axes"], "--"))
        rows.append((d["xyz"], d["rgb"], poses, "world frame"))
    fig, axs = plt.subplots(len(rows), 4, figsize=(18, 4.4 * len(rows)), squeeze=False)
    for r, (xyz, rgb, poses, name) in enumerate(rows):
        for c, (i, j, t) in enumerate(VIEWS):
            draw(axs[r, c], xyz, rgb, poses, i, j, f"{name}: {t}")
        if poses:
            draw(axs[r, 3], xyz, rgb, poses, 0, 2, f"{name}: zoom 12 cm around label (x-z)", zoom=(poses[0][0], 0.12))
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
