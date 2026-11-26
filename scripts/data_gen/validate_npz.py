"""Sanity checks for RiEMann point-cloud npz files (demo.npz and evaluation scenes).

Only depends on numpy, so it runs both in Isaac Sim's bundled Python and in the pixi env.

Usage (from the repo root):
    python scripts/data_gen/validate_npz.py data/mug/place/demo.npz
    python scripts/data_gen/validate_npz.py data/mug/place/newpose.npz
    python scripts/data_gen/validate_npz.py data/mug/place/*.npz

Exit code is 1 if any FAIL is reported, 0 otherwise (WARNs do not fail).
"""
import sys
import numpy as np

TABLE_RGB = np.array([0.1176, 0.4392, 0.4078])  # must match utils/data_utils.py:downsample_table
TABLE_TOL = 0.1                                  # same tolerance as downsample_table
FEATURE_POINT_RADIUS = 0.02                      # mani.feature_point_radius
EXPECTED_POINTS = 8192
MIN_FRAMES = 20                                  # SE3Demo keeps frames [0:20]


class Report:
    def __init__(self, name):
        self.name, self.fails, self.warns = name, 0, 0

    def ok(self, msg):
        print(f"  [ OK ] {msg}")

    def warn(self, msg):
        self.warns += 1
        print(f"  [WARN] {msg}")

    def fail(self, msg):
        self.fails += 1
        print(f"  [FAIL] {msg}")


def table_mask(rgb):
    return np.linalg.norm(rgb - TABLE_RGB, axis=-1) < TABLE_TOL


def axes_to_R(axes):
    # Training code: R = axes.reshape(3, 3).T  (axes = [x_axis, y_axis, z_axis] concatenated)
    return axes.reshape(3, 3).T


def check_rotation(axes, rep, tag):
    R = axes_to_R(axes)
    orth = np.abs(R.T @ R - np.eye(3)).max()
    det = np.linalg.det(R)
    if orth > 1e-3 or abs(det - 1) > 1e-3:
        rep.fail(f"{tag}: axes is not a proper rotation (|RᵀR-I|max={orth:.2e}, det={det:.4f})")
        return False
    return True


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return R, t, np.abs(A @ R.T + t - B).max()


def check_cloud(xyz, rgb, rep, tag):
    n = xyz.shape[0]
    if n != EXPECTED_POINTS:
        rep.warn(f"{tag}: {n} points (expected {EXPECTED_POINTS})")
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        rep.fail(f"{tag}: NaN/Inf in xyz or rgb")
    if rgb.min() < 0 or rgb.max() > 1:
        rep.fail(f"{tag}: rgb outside [0,1] (min={rgb.min():.3f}, max={rgb.max():.3f}); RiEMann expects 0-1 floats")
    m = table_mask(rgb)
    if m.sum() == 0:
        rep.fail(f"{tag}: no table-colored points; downsample_table will keep only object points")
    if (~m).sum() > 2500:
        rep.warn(f"{tag}: {(~m).sum()} non-table points > 2500; downsample_table will drop ALL table points")
    return m


def check_demo(d, rep):
    for k in ["xyz", "rgb", "seg_center", "axes"]:
        if k not in d:
            rep.fail(f"missing key '{k}'")
            return
    xyz, rgb, c, ax = d["xyz"], d["rgb"], d["seg_center"], d["axes"]
    if xyz.ndim != 4 or rgb.shape != xyz.shape or c.shape != xyz.shape[:2] + (3,) or ax.shape != xyz.shape[:2] + (9,):
        rep.fail(f"bad shapes xyz{xyz.shape} rgb{rgb.shape} seg_center{c.shape} axes{ax.shape}; "
                 f"expected [traj,T,N,3], [traj,T,3], [traj,T,9]")
        return
    n_traj, n_frames, n_pts, _ = xyz.shape
    rep.ok(f"shapes: {n_traj} traj x {n_frames} frames x {n_pts} points")
    if n_frames < MIN_FRAMES:
        rep.fail(f"only {n_frames} frames; SE3Demo slices [0:20]")
    elif n_frames > MIN_FRAMES:
        rep.warn(f"{n_frames} frames; SE3Demo will silently use only the first {MIN_FRAMES}")
    if n_traj < 5:
        rep.warn(f"only {n_traj} trajectories (paper uses 10)")

    F = min(n_frames, MIN_FRAMES)
    for tr in range(n_traj):
        tag = f"traj{tr}"
        bad_rot = sum(not check_rotation(ax[tr, f], rep, f"{tag} f{f}") for f in range(F))
        # degenerate label (seen in data/mug/pick traj 7): identity pose at origin in every frame
        if np.abs(c[tr, :F]).max() < 1e-3 and np.abs(ax[tr, :F] - np.eye(3).reshape(9)).max() < 1e-3:
            rep.fail(f"{tag}: degenerate label (seg_center=0, axes=identity in all frames)")
            continue
        near = []
        for f in range(F):
            check_cloud(xyz[tr, f], rgb[tr, f], rep, f"{tag} f{f}") if f == 0 else None
            near.append(np.linalg.norm(xyz[tr, f] - c[tr, f], axis=1).min())
        near = np.array(near)
        if near.max() > FEATURE_POINT_RADIUS:
            rep.warn(f"{tag}: label is {near.max()*100:.1f} cm from nearest point in some frame "
                     f"(> feature_point_radius {FEATURE_POINT_RADIUS*100:.0f} cm); orientation pooling will fall back")
        dist = np.linalg.norm(c[tr, :F], axis=1)
        trend = "decreasing" if dist[-1] < dist[0] else "NOT decreasing"
        msg = f"{tag}: |t| {dist[0]:.3f} -> {dist[-1]:.3f} m ({trend}), nearest-point max {near.max()*100:.1f} cm"

        # EE-frame consistency: table points (static in world) must move rigidly, and the label
        # must move with the same transform. Requires the same table points in the same order per frame.
        m0 = table_mask(rgb[tr, 0])
        same_order = all((table_mask(rgb[tr, f]) == m0).all() for f in range(F))
        if same_order and m0.sum() >= 3:
            worst_fit, worst_lab = 0.0, 0.0
            R0 = axes_to_R(ax[tr, 0])
            for f in range(1, F):
                R, t, err = kabsch(xyz[tr, 0][m0], xyz[tr, f][m0])
                worst_fit = max(worst_fit, err)
                lab = max(np.abs(R @ c[tr, 0] + t - c[tr, f]).max(), np.abs(R @ R0 - axes_to_R(ax[tr, f])).max())
                worst_lab = max(worst_lab, lab)
            if worst_fit > 1e-4:
                rep.fail(f"{tag}: table points not rigid across frames (err {worst_fit:.2e}); EE transform is wrong")
            elif worst_lab > 1e-3:
                rep.fail(f"{tag}: label does not follow the scene transform (err {worst_lab:.2e}); "
                         f"label must be T_ee_k^-1 * T_label_world")
            else:
                msg += ", EE-frame consistency OK"
        else:
            msg += ", EE-frame consistency SKIPPED (table points differ between frames)"
        if bad_rot == 0:
            rep.ok(msg)


def check_eval(d, rep):
    for k in ["xyz", "rgb"]:
        if k not in d:
            rep.fail(f"missing key '{k}'")
            return
    xyz, rgb = d["xyz"], d["rgb"]
    if xyz.ndim != 2 or xyz.shape != rgb.shape:
        rep.fail(f"bad shapes xyz{xyz.shape} rgb{rgb.shape}; expected [N,3] (infer.py reads a single scene)")
        return
    m = check_cloud(xyz, rgb, rep, "scene")
    tz = xyz[m][:, 2] if m.any() else np.array([np.nan])
    rep.ok(f"{xyz.shape[0]} points, table={m.sum()} (z mean {np.nanmean(tz):.3f}), "
           f"bbox min={np.round(xyz.min(0), 3)} max={np.round(xyz.max(0), 3)}")
    if "seg_center" in d and "axes" in d:
        if check_rotation(np.asarray(d["axes"]).reshape(9), rep, "gt"):
            near = np.linalg.norm(xyz - np.asarray(d["seg_center"]).reshape(3), axis=1).min()
            rep.ok(f"ground truth present, nearest point to label {near*100:.1f} cm")
    else:
        rep.warn("no seg_center/axes ground truth; geodesic error cannot be computed")


def main(paths):
    total_fail = 0
    for p in paths:
        if "/._" in p or p.startswith("._"):
            continue  # macOS AppleDouble files
        print(f"== {p}")
        rep = Report(p)
        d = np.load(p, allow_pickle=True)
        if isinstance(d, np.ndarray):
            d = d.item()
        if "xyz" in d and np.asarray(d["xyz"]).ndim == 4:
            check_demo(d, rep)
        else:
            check_eval(d, rep)
        print(f"   -> {rep.fails} FAIL, {rep.warns} WARN")
        total_fail += rep.fails
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
