"""Generate RiEMann Plane-on-Shelf *place* data in Isaac Sim 5.1 (same pipeline as gen_mug_sim.py).

    isaac-python scripts/data_gen/gen_plane_sim.py --out data/plane/place

Task: the plane (geometry.PLANES) is grasped on the fuselage behind the wing, lifted, carried and
lowered onto a box shelf whose two support rails carry the wings (geometry.SHELF). A back panel on
the shelf's -y side makes the placement frame unique (no 180 deg yaw ambiguity).

Outputs (under --out), identical layout to the mug place data:
    demo.npz                      [traj,20,8192,3] in the EE frame of each transport step + labels
    {setting}.npz, eval/{setting}/seed_XX.npz   world-frame scenes with ground truth
    raw/, gen_log.json

Labels
  seg_center/axes      = label B = T_place (box top midway between the rails; x = nose direction, z = up)
  ee_target_pos/axes   = label A = T_place @ inv(G)  (EE pose at release)
  grasp_in_ee          = G = inv(M_G) @ T_PS  (placement frame in the EE frame while held)
Settings: training (T), newpose (NP: shelf/plane anywhere, any yaw), instance (NI: plane_b),
distracting (DO: 2-4 YCB objects), all (NP + NI + DO).
Success: the dynamic plane released at the EE target must rest on the rails within 1.2 cm / 8 deg
of T_place after 2 s (isaac_scene.PlaneShelfScene.rest_test).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import geometry as G  # noqa: E402
from gen_mug_sim import (SETTINGS, N_FRAMES, LIFT, PRE_PLACE, RETREAT, DENSE, LABEL_SEEN,  # noqa: E402
                         Reject, solve_ik, solve_path, bezier_path, check_robot, check_held_mug,
                         episode_arrays, write_demo)

PLANE_CLEAR = 0.04        # plane footprint <-> shelf / distractors on the table


# --------------------------------------------------------------- layouts
def sample_layout(setting, rng, distractor_names, distractor_bbox):
    key = "plane_b" if setting in ("instance", "all") else "plane_a"
    pl = G.PLANES[key]
    newpose = setting in ("newpose", "all")
    shelf0 = G.shelf_prims()
    plane_pts0 = G.prims_surface_samples(G.plane_prims(pl), 600)
    for _ in range(2000):
        if newpose:
            shelf_xy = [rng.uniform(0.50, 0.72), rng.uniform(-0.22, 0.22)]
            shelf_yaw = rng.uniform(-np.pi, np.pi)
            plane_xy = [rng.uniform(0.28, 0.62), rng.uniform(-0.25, 0.25)]
            plane_yaw = rng.uniform(-np.pi, np.pi)
        else:
            shelf_xy = [rng.uniform(0.60, 0.70), rng.uniform(-0.20, 0.05)]
            shelf_yaw = rng.uniform(-np.pi / 2, np.pi / 2)
            plane_xy = [rng.uniform(0.32, 0.45), rng.uniform(-0.05, 0.25)]
            plane_yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        T_shelf = G.make_T(G.rot_z(shelf_yaw), [*shelf_xy, 0])
        T_plane = G.make_T(G.rot_z(plane_yaw), [*plane_xy, 0])
        shelf_w = G.transform_prims(shelf0, T_shelf)
        Pw = plane_pts0 @ T_plane[:3, :3].T + T_plane[:3, 3]
        if G.sdf_prims(Pw, shelf_w).min() < PLANE_CLEAR:
            continue
        if not (G.WS_X[0] <= plane_xy[0] <= G.WS_X[1] and G.WS_Y[0] <= plane_xy[1] <= G.WS_Y[1]):
            continue
        if np.hypot(*plane_xy) < 0.30:                 # keep the tail away from the robot base
            continue
        break
    else:
        raise Reject("layout")

    distractors = []
    if setting in ("distracting", "all"):
        k = int(rng.integers(2, 5))
        names = list(rng.choice(distractor_names, k, replace=False))
        placed = []
        for name in names:
            lo, hi = distractor_bbox[name]
            r_d = float(np.linalg.norm(np.maximum(np.abs(lo[:2]), np.abs(hi[:2]))))
            for _ in range(300):
                xy = np.array([rng.uniform(0.25, 0.75), rng.uniform(-0.27, 0.27)])
                if np.linalg.norm(Pw[:, :2] - xy, axis=1).min() < r_d + 0.03:
                    continue
                if G.sdf_prims(np.array([[*xy, 0.02]]), shelf_w)[0] < r_d + 0.03:
                    continue
                if any(np.linalg.norm(xy - p) < r_d + rp + 0.02 for p, rp in placed):
                    continue
                T = G.make_T(G.rot_z(rng.uniform(-np.pi, np.pi)), [*xy, -lo[2]])
                distractors.append(dict(name=name, T=T, lo=lo, hi=hi))
                placed.append((xy, r_d))
                break
    return dict(setting=setting, plane_key=key, T_shelf=T_shelf, T_plane=T_plane, distractors=distractors)


def obstacles(layout):
    prims = G.transform_prims(G.shelf_prims(), layout["T_shelf"])
    for d in layout["distractors"]:
        c = (d["lo"] + d["hi"]) / 2
        prims.append(("box", d["T"] @ G.trans(c), (d["hi"] - d["lo"]) / 2))
    return prims


def place_scene(sc, layout):
    sc.park_all()
    sc.set_shelf(layout["T_shelf"])
    for d in layout["distractors"]:
        sc.set_distractor(d["name"], d["T"])


# -------------------------------------------------------------- episodes
def plan_place(sc, layout, M_G, prims, anchor):
    """grasp -> lift -> transport (20 frames) -> lower onto the rails -> retreat, as gen_mug_sim.plan_place."""
    pl = G.PLANES[layout["plane_key"]]
    T_grasp = layout["T_plane"] @ M_G
    PS = G.plane_T_PS(pl)
    Gr = G.inv_T(M_G) @ PS                                      # placement frame in the EE frame
    T_place = layout["T_shelf"] @ G.shelf_T_place()
    T_target = T_place @ G.inv_T(Gr)
    up = T_place[:3, 2]
    T_pre = G.trans(PRE_PLACE * up) @ T_target
    T_lift = G.trans([0, 0, LIFT]) @ T_grasp
    finger = pl["fus_r"] + 0.0005
    segs = dict(
        lift=[G.interp_T(T_grasp, T_lift, s) for s in np.linspace(0, 1, 11)],
        transport=bezier_path(T_lift, T_pre, up, DENSE * (N_FRAMES - 1) + 1),
        insert=[G.interp_T(T_pre, T_target, s) for s in np.linspace(0, 1, 21)],
        retreat=[T_target @ G.trans([0, 0, -RETREAT * s]) for s in np.linspace(0, 1, 9)],
    )
    poses, idx = [], {}
    for name, seg in segs.items():
        a = len(poses) - (1 if poses else 0)
        poses += seg if not poses else seg[1:]
        idx[name] = (a, len(poses))
    k = 0 if anchor == "grasp" else idx["insert"][1] - 1
    qk = solve_ik(sc, poses[k], anchor)
    after = solve_path(sc, poses[k + 1:], qk, f"track_{anchor}") if k + 1 < len(poses) else np.zeros((0, 7))
    before = solve_path(sc, poses[:k][::-1], qk, f"track_{anchor}")[::-1] if k > 0 else np.zeros((0, 7))
    q_all = np.vstack([before, qk[None], after])
    q = {n: q_all[a:b] for n, (a, b) in idx.items()}

    check_robot(sc, q["lift"][1:], prims, finger, "lift")
    check_robot(sc, q["transport"], prims, finger, "transport")
    check_robot(sc, q["insert"], prims, finger, "insert", margin=0.005)
    check_robot(sc, q["retreat"][1:], prims, 0.04, "retreat", margin=0.005)
    samples = G.prims_surface_samples(G.plane_prims(pl), 1200)
    check_held_mug(segs["transport"], M_G, samples, prims, "transport", 0.005)
    check_held_mug(segs["insert"], M_G, samples, prims, "insert", 0.0015)
    return dict(T_place=T_place, T_target=T_target, M_G=M_G, PS=PS, G=Gr, finger=finger, q_grasp=q["lift"][0],
                Ts=segs["transport"][::DENSE], qs=q["transport"][::DENSE], anchor=anchor)


def run_place(sc, layout, rng):
    key = layout["plane_key"]
    pl = G.PLANES[key]
    prims = obstacles(layout)
    dx = rng.uniform(-0.008, 0.008)                         # grasp jitter along the fuselage
    err = None
    for flip in (False, True):
        for anchor in ("grasp", "target"):
            try:
                M_G = G.trans([dx, 0, 0]) @ G.plane_T_PG(pl, flip=flip)
                plan = plan_place(sc, layout, M_G, prims, anchor)
                break
            except Reject as e:
                err = e
        else:
            continue
        break
    else:
        raise err
    rest = sc.rest_test(key, plan["T_target"] @ G.inv_T(plan["M_G"]), plan["T_place"])
    if not rest["success"]:
        raise Reject("rest")
    place_scene(sc, layout)
    sc.set_plane_vis(key, plan["Ts"][0] @ G.inv_T(plan["M_G"]))
    sc.set_robot(plan["qs"][0], plan["finger"])
    return dict(T_ee=np.array(plan["Ts"]), q=plan["qs"], T_label=plan["T_place"], T_ee_target=plan["T_target"],
                G=plan["G"], M_G=plan["M_G"], PS=plan["PS"], held_label="plane", rest=rest, q_grasp=plan["q_grasp"],
                grasp=dict(dx=dx, flip=bool(flip), anchor=plan["anchor"]))


# ---------------------------------------------------------------- output
def check_capture(P, S, layout, T_plane_now):
    from isaac_scene import SEM
    shelf = G.transform_prims(G.shelf_prims(), layout["T_shelf"])
    plane = G.transform_prims(G.plane_prims(G.PLANES[layout["plane_key"]]), T_plane_now)
    for name, prims in (("shelf", shelf), ("plane", plane)):
        Q = P[S == SEM[name]]
        if len(Q) < 200:
            raise Reject(f"few_{name}_points")
        d = np.abs(G.sdf_prims(Q[:: max(1, len(Q) // 3000)], prims))
        if np.percentile(d, 95) > 0.004:
            raise RuntimeError(f"captured {name} points do not match the scene (p95 {np.percentile(d, 95):.4f} m); stale render?")


def build_cloud(sc, ep_seed, layout, T_plane_now):
    P, C, S = sc.capture()
    keep = G.crop_mask(P)
    check_capture(P[keep], S[keep], layout, T_plane_now)
    P, C, S = G.voxel_down(P[keep], C[keep], S[keep], 0.002)
    if len(P) < 500:
        raise Reject("few_points")
    xyz, rgb, sem = G.assemble_cloud(P, C, S, ep_seed)
    return xyz, rgb, sem, (P.astype(np.float32), C.astype(np.float32), S.astype(np.int8))


def save_raw(path, layout, ep, raw, seeds, cams):
    np.savez_compressed(
        path, task="place", setting=layout["setting"], plane_key=layout["plane_key"], seed=np.array(seeds),
        T_shelf=layout["T_shelf"], T_plane=layout["T_plane"], T_ee=ep["T_ee"], q=ep["q"], q_grasp=ep["q_grasp"],
        T_label=ep["T_label"], T_ee_target=ep["T_ee_target"], M_G=ep["M_G"], G=ep["G"], PS=ep["PS"],
        P_obj_w=raw[0], rgb_obj=raw[1], sem_obj=raw[2], cam_poses=cams,
        rest=json.dumps(ep["rest"]), grasp=json.dumps(ep["grasp"]),
        distractors=json.dumps([dict(name=x["name"], T=x["T"].tolist()) for x in layout["distractors"]]))


def generate(sc, setting, n, base_seed, setting_idx, out_dir, kind, log):
    from isaac_scene import world_T
    cams = np.array([world_T(c) for c in sc.cams])
    bbox = {k: sc.distractor_bbox(k) for k in sc.distractors}
    eps, stats, attempt = [], {}, 0
    t0 = time.time()
    while len(eps) < n:
        attempt += 1
        if attempt > 60 * n:
            raise RuntimeError(f"{setting}: only {len(eps)}/{n} episodes after {attempt} attempts: {stats}")
        seeds = [base_seed, setting_idx, 0 if kind == "demo" else 1, attempt]
        rng = np.random.default_rng(seeds)
        try:
            layout = sample_layout(setting, rng, list(sc.distractors), bbox)
            place_scene(sc, layout)
            ep = run_place(sc, layout, rng)
            ep_seed = int(np.random.SeedSequence(seeds).generate_state(1)[0])
            T_plane_now = ep["T_ee"][0] @ G.inv_T(ep["M_G"])
            xyz_w, rgb, sem, raw = build_cloud(sc, ep_seed, layout, T_plane_now)
            if np.linalg.norm(xyz_w[sem >= 0] - ep["T_label"][:3, 3], axis=1).min() > LABEL_SEEN:
                raise Reject("label_unseen")
        except Reject as e:
            stats[str(e)] = stats.get(str(e), 0) + 1
            continue
        i = len(eps)
        os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
        save_raw(os.path.join(out_dir, "raw", f"{kind}_{setting}_{i:02d}.npz"), layout, ep, raw, seeds, cams)
        eps.append((layout, ep, xyz_w, rgb, sem, seeds))
        print(f"[plane/{kind}/{setting}] {i + 1}/{n} attempt {attempt} seed {seeds} objects={int((sem >= 0).sum())} "
              f"rest: {ep['rest']['pos_err']*100:.1f} cm {ep['rest']['rot_err_deg']:.1f} deg")
    stats["accepted"], stats["attempts"], stats["seconds"] = len(eps), attempt, round(time.time() - t0, 1)
    log[f"{kind}/{setting}"] = stats
    print(f"[plane/{kind}/{setting}] done: {stats}")
    return eps


def write_eval(ep_tuple, path):
    layout, ep, xyz_w, rgb, sem, seeds = ep_tuple
    c, a = G.pose_to_label(ep["T_label"])
    ct, at = G.pose_to_label(ep["T_ee_target"])
    np.savez(path, xyz=xyz_w, rgb=rgb, seg_center=c, axes=a, ee_pose=ep["T_ee"][0], seed=np.array(seeds),
             plane_key=layout["plane_key"], T_shelf=layout["T_shelf"], T_plane=layout["T_plane"],
             ee_target_pos=ct, ee_target_axes=at, grasp_in_ee=ep["G"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demos", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--settings", nargs="*", default=SETTINGS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-demo", action="store_true")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "extra_args": ["--portable", "--portable-root", os.path.expanduser("~/.cache/isaac-sim-5.1")]})
    code = 0
    try:
        from isaac_scene import PlaneShelfScene
        sc = PlaneShelfScene(app)
        os.makedirs(args.out, exist_ok=True)
        log = dict(task="place", object="plane", seed=args.seed, argv=sys.argv)
        if not args.no_demo:
            eps = generate(sc, "training", args.demos, args.seed, 0, args.out, "demo", log)
            write_demo("place", eps, os.path.join(args.out, "demo.npz"))
        for s in args.settings:
            eps = generate(sc, s, args.eval_seeds, args.seed, SETTINGS.index(s), args.out, "eval", log)
            os.makedirs(os.path.join(args.out, "eval", s), exist_ok=True)
            for i, e in enumerate(eps):
                write_eval(e, os.path.join(args.out, "eval", s, f"seed_{i:02d}.npz"))
            write_eval(eps[0], os.path.join(args.out, f"{s}.npz"))
            print("wrote", os.path.join(args.out, f"{s}.npz"), "+", len(eps), "eval seeds")
        with open(os.path.join(args.out, "gen_log.json"), "w") as f:
            json.dump(log, f, indent=1)
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    app.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
