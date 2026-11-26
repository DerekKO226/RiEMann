"""Evaluate a trained RiEMann model on the Isaac Sim evaluation seeds (plan 11.3).

    python scripts/testing/eval_place.py -exp_name mug -pick_or_place place
    python scripts/testing/eval_place.py -exp_name mug_sim -pick_or_place pick -settings training newpose

Reads data/{exp}/{task}/eval/{setting}/seed_*.npz (world frame, ground truth seg_center/axes), predicts
with the trained SegNet + ManiNet and reports per setting:
  pos_err (m), rot_err (deg), double geodesic sqrt(pos^2 + rot_rad^2) as in the paper (Table 2).
For place files with grasp_in_ee (G), the EE target T_peg_pred @ inv(G) is also compared with the
ground-truth EE target (label A). Predictions are written to
experiments/{exp}/{task}/eval_predictions.npz for the Isaac closed-loop test
(scripts/data_gen/closed_loop_place.py).
"""
import sys
sys.path.append(".")
sys.path.append("scripts/testing")
import argparse
import glob
import os

import numpy as np

from infer import load_policies, predict

SETTINGS = ["training", "newpose", "instance", "distracting", "all"]


def rot_err(R1, R2):
    return float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def main(args):
    cfgs = load_policies(args.exp_name, args.pick_or_place)
    root = os.path.join("data", args.exp_name, args.pick_or_place, "eval")
    rows, preds = [], {}
    for setting in args.settings:
        files = sorted(glob.glob(os.path.join(root, setting, "seed_*.npz")))
        if not files:
            print(f"[{setting}] no files under {root}/{setting}")
            continue
        errs = []
        for f in files:
            d = np.load(f)
            pos, R = predict(*cfgs, d["xyz"], d["rgb"], draw_pcd=False, save_ori_feature=False)
            R_gt = d["axes"].reshape(3, 3).T
            e = dict(pos=float(np.linalg.norm(pos - d["seg_center"])), rot=rot_err(R, R_gt))
            e["geo"] = float(np.sqrt(e["pos"] ** 2 + e["rot"] ** 2))
            if "grasp_in_ee" in d.files:
                T_ee = to_T(R, pos) @ np.linalg.inv(d["grasp_in_ee"])
                R_ee_gt = d["ee_target_axes"].reshape(3, 3).T
                e["ee_pos"] = float(np.linalg.norm(T_ee[:3, 3] - d["ee_target_pos"]))
                e["ee_rot"] = rot_err(T_ee[:3, :3], R_ee_gt)
            errs.append(e)
            preds[f"{setting}/{os.path.basename(f)}"] = to_T(R, pos)
        mean = {k: float(np.mean([e[k] for e in errs])) for k in errs[0]}
        rows.append((setting, len(errs), mean))
        msg = (f"[{setting}] n={len(errs)}  pos {mean['pos']*100:.2f} cm  rot {np.rad2deg(mean['rot']):.1f} deg  "
               f"double-geodesic {mean['geo']:.4f}")
        if "ee_pos" in mean:
            msg += f"  | EE target pos {mean['ee_pos']*100:.2f} cm rot {np.rad2deg(mean['ee_rot']):.1f} deg"
        print(msg)
    out = os.path.join("experiments", args.exp_name, args.pick_or_place, "eval_predictions.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, keys=np.array(list(preds)), poses=np.array(list(preds.values())))
    print("predictions saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_name", type=str, default="mug")
    parser.add_argument("-pick_or_place", type=str, choices=["pick", "place"], default="place")
    parser.add_argument("-settings", nargs="*", default=SETTINGS)
    main(parser.parse_args())
