"""Closed-loop place test in Isaac Sim (plan C13 / 11.3).

    python scripts/testing/eval_place.py -exp_name mug -pick_or_place place        # pixi env: predictions
    isaac-python scripts/data_gen/closed_loop_place.py --exp mug                    # Isaac: execute them
    isaac-python scripts/data_gen/closed_loop_place.py --exp mug --gt               # sanity check with ground truth

For every evaluation seed the predicted placement frame T_peg_pred is executed as the plan does at run
time: EE target = T_peg_pred @ inv(G). The mug (rigidly held, G from the seed) is moved along the
predicted peg axis from 0.10 m out to the predicted pose; if any mug surface sample penetrates the rack
on the way the trial fails ("collision"). Otherwise the dynamic mug is released there and must still be
hooked on the target peg after 3 s (same criterion as data generation).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="mug")
    ap.add_argument("--gt", action="store_true", help="execute the ground-truth label instead of predictions")
    args = ap.parse_args()
    data_root = os.path.join("data", args.exp, "place", "eval")
    if args.gt:
        import glob
        keys = sorted(os.path.relpath(f, data_root) for f in glob.glob(os.path.join(data_root, "*", "seed_*.npz")))
        poses = [None] * len(keys)
    else:
        p = np.load(os.path.join("experiments", args.exp, "place", "eval_predictions.npz"))
        keys, poses = list(p["keys"]), list(p["poses"])

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "extra_args": ["--portable", "--portable-root", os.path.expanduser("~/.cache/isaac-sim-5.1")]})
    code = 0
    try:
        from isaac_scene import MugRackScene
        sc = MugRackScene(app, with_distractors=False)
        results = {}
        for key, T_pred in zip(keys, poses):
            d = np.load(os.path.join(data_root, key))
            setting = key.split("/")[0]
            mug_key = str(d["mug_key"])
            m = G.MUGS[mug_key]
            T_rack = d["T_rack"]
            if T_pred is None:
                T_pred = G.make_T(d["axes"].reshape(3, 3).T, d["seg_center"])
            M_H = G.mug_T_MH(m)
            sc.park_all()
            sc.set_rack(T_rack)
            rack = G.transform_prims(G.rack_collision_prims(), T_rack)
            samples = G.mug_surface_samples(m, 1500)
            # insertion along the predicted peg axis, mug pose = T_peg_pred(shifted) @ inv(M_H)
            pen = np.inf
            for s in np.linspace(0.10, 0, 41):                       # pre-place -> target (gen_mug_sim.PRE_PLACE)
                TM = G.trans(s * T_pred[:3, 2]) @ T_pred @ G.inv_T(M_H)
                pen = min(pen, G.sdf_prims(samples @ TM[:3, :3].T + TM[:3, 3], rack).min())
            a, b = G.peg_segment(G.RACK["pegs"][G.RACK["target"]])
            seg_w = (T_rack[:3, :3] @ a + T_rack[:3, 3], T_rack[:3, :3] @ b + T_rack[:3, 3])
            T_peg_gt = T_rack @ G.rack_T_peg()
            if pen < -0.001:
                r = dict(success=False, reason="collision", penetration=float(-pen))
            else:
                r = sc.hang_test(mug_key, T_pred @ G.inv_T(M_H), T_peg_gt, seg_w)
                r["reason"] = "ok" if r["success"] else "fell"
            r["pred_pos_err"] = float(np.linalg.norm(T_pred[:3, 3] - T_peg_gt[:3, 3]))
            results.setdefault(setting, []).append(r)
            print(f"{key}: {r['reason']}  (pos err {r['pred_pos_err']*100:.1f} cm)")
        summary = {s: dict(n=len(v), success_rate=float(np.mean([x["success"] for x in v])),
                           reasons={k: sum(x["reason"] == k for x in v) for k in ("ok", "collision", "fell")})
                   for s, v in results.items()}
        for s, v in summary.items():
            print(f"[{s}] success {v['success_rate']*100:.0f}% of {v['n']}  {v['reasons']}")
        out = os.path.join("experiments", args.exp, "place", "closed_loop_gt.json" if args.gt else "closed_loop.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(summary=summary, trials=results), f, indent=1)
        print("saved", out)
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    app.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
