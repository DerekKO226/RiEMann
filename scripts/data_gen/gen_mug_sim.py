"""Generate RiEMann mug/rack point-cloud data in Isaac Sim 5.1 (plan: place_p_cloud_plan.md).

Run with Isaac Sim's Python from the repo root, e.g.

    isaac-python scripts/data_gen/gen_mug_sim.py --task pick  --out data/mug_sim/pick
    isaac-python scripts/data_gen/gen_mug_sim.py --task place --out data/mug/place

Outputs (under --out):
    demo.npz                      [traj,20,8192,3] in the EE frame of each step + labels (training setting)
    {setting}.npz                 one world-frame scene per setting (= eval/{setting}/seed_00.npz), infer.py compatible
    eval/{setting}/seed_XX.npz    world-frame scenes with ground truth
    raw/                          per-episode world clouds, poses, joint paths, seeds, hang-test result
    gen_log.json                  attempts / rejection reasons per setting

Labels
  pick : seg_center/axes = TCP grasp pose (rim opposite the handle, approach down).
  place: seg_center/axes = label B = T_peg (placement frame on the highest peg).
         ee_target_pos/axes = label A = T_peg @ inv(G); grasp_in_ee = G = inv(T_ee) @ T_handle.
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

SETTINGS = ["training", "newpose", "instance", "distracting", "all"]
N_FRAMES = 20
LIFT = 0.15            # plan 9.3
PRE_PLACE = 0.10       # plan 9.4: back off along the peg
PRE_GRASP = (0.12, 0.21)   # pick trajectories end this far above the grasp (data/mug/pick: 0.11-0.22 m)
PRE_GRASP_ROT = (2, 20)    # ... with this much residual rotation (deg; data/mug/pick: 2.5-21)
START_ROT = (15, 50)       # first frame rotation offset (deg; data/mug/pick median 34)
RETREAT = 0.08
MAX_JOINT_STEP = 0.25  # rad between consecutive (dense) path samples
DENSE = 3              # IK is solved on a DENSE-times finer path; every DENSE-th sample is a frame
LABEL_SEEN = 0.015     # the label must be within this distance of an observed point (plan 2.5)


class Reject(Exception):
    pass


# --------------------------------------------------------------- layouts
def sample_layout(setting, rng, distractor_names, distractor_bbox):
    mug_key = "mug_b" if setting in ("instance", "all") else "mug_a"
    m = G.MUGS[mug_key]
    newpose = setting in ("newpose", "all")
    rack_prims0 = G.rack_collision_prims()
    r_mug = m["handle_cx"] + m["handle_rh"] + m["handle_rt"]
    for _ in range(1000):
        if newpose:
            rack_xy = [rng.uniform(0.48, 0.70), rng.uniform(-0.20, 0.20)]
            rack_yaw = rng.uniform(-np.pi, np.pi)
            mug_xy = [rng.uniform(0.30, 0.65), rng.uniform(-0.25, 0.25)]
            mug_yaw = rng.uniform(-np.pi, np.pi)
        else:
            rack_xy = [rng.uniform(0.60, 0.70), rng.uniform(-0.12, 0.02)]
            rack_yaw = rng.uniform(-np.pi / 2, np.pi / 2)
            mug_xy = [rng.uniform(0.35, 0.55), rng.uniform(0.00, 0.22)]
            mug_yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        T_rack = G.make_T(G.rot_z(rack_yaw), [*rack_xy, 0])
        T_mug = G.make_T(G.rot_z(mug_yaw), [*mug_xy, 0])
        dist = np.linalg.norm(np.subtract(mug_xy, rack_xy))
        if not 0.19 <= dist <= 0.31:
            continue
        rack_w = G.transform_prims(rack_prims0, T_rack)
        if G.sdf_prims(np.array([[*mug_xy, 0.05]]), rack_w)[0] < r_mug + 0.03:
            continue
        if not (G.WS_X[0] + r_mug <= mug_xy[0] <= G.WS_X[1] - r_mug and G.WS_Y[0] + r_mug <= mug_xy[1] <= G.WS_Y[1] - r_mug):
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
            for _ in range(200):
                xy = np.array([rng.uniform(0.25, 0.75), rng.uniform(-0.27, 0.27)])
                if np.linalg.norm(xy - mug_xy) < r_mug + r_d + 0.03:
                    continue
                if G.sdf_prims(np.array([[*xy, 0.02]]), rack_w)[0] < r_d + 0.03:
                    continue
                if any(np.linalg.norm(xy - p) < r_d + rp + 0.02 for p, rp in placed):
                    continue
                yaw = rng.uniform(-np.pi, np.pi)
                T = G.make_T(G.rot_z(yaw), [*xy, -lo[2]])
                distractors.append(dict(name=name, T=T, lo=lo, hi=hi))
                placed.append((xy, r_d))
                break
    return dict(setting=setting, mug_key=mug_key, T_rack=T_rack, T_mug=T_mug, distractors=distractors)


def obstacles(layout, include_mug=False):
    prims = G.transform_prims(G.rack_collision_prims(), layout["T_rack"])
    for d in layout["distractors"]:
        c = (d["lo"] + d["hi"]) / 2
        prims.append(("box", d["T"] @ G.trans(c), (d["hi"] - d["lo"]) / 2))
    if include_mug:
        m = G.MUGS[layout["mug_key"]]
        T = layout["T_mug"]
        prims.append(("capsule", T[:3, 3], T[:3, :3] @ [0, 0, m["height"]] + T[:3, 3], m["radius"]))
        prims.append(("capsule", T[:3, :3] @ [m["handle_cx"] + m["handle_rh"], 0, m["handle_z"] - 0.03] + T[:3, 3],
                      T[:3, :3] @ [m["handle_cx"] + m["handle_rh"], 0, m["handle_z"] + 0.03] + T[:3, 3], m["handle_rt"] + 0.01))
    return prims


# ------------------------------------------------------- kinematic checks
def solve_ik(sc, T, what):
    """Global IK (Lula, several seeds) for the first pose of a motion."""
    from isaac_scene import HOME_Q
    q, ok = sc.ik_tcp(T, warm=HOME_Q)
    if not ok:
        raise Reject(f"ik:{what}")
    return q


def solve_path(sc, Ts, q_start, what, max_step=MAX_JOINT_STEP):
    """Continuous joint path along dense TCP poses, starting from q_start (DLS tracking)."""
    qs, ok = sc.track_tcp(Ts, q_start)
    if not ok:
        raise Reject(f"ik:{what}")
    steps = np.abs(np.diff(np.vstack([q_start, qs]), axis=0)).max()
    if steps > max_step:
        raise Reject(f"jump:{what}")
    return qs


def bezier_path(T0, T1, approach_dir, n, lead=0.12):
    """Quadratic Bezier in translation whose last tangent is -approach_dir (arrive along it),
    geodesic rotation; both on a min-jerk time scale."""
    p0, p1 = T0[:3, 3], T1[:3, 3]
    pc = p1 + lead * approach_dir
    out = []
    for s in G.min_jerk(n):
        T = G.interp_T(T0, T1, s)
        T[:3, 3] = (1 - s) ** 2 * p0 + 2 * s * (1 - s) * pc + s ** 2 * p1
        out.append(T)
    return out


def check_robot(sc, qs, prims, finger, what, margin=0.01, skip_links=()):
    links = [l for l in sc.spheres if l not in skip_links] + (["fingers"] if "fingers" not in skip_links else [])
    for q in qs:
        C, R = sc.robot_spheres(q, finger, links=links)
        if prims and (G.sdf_prims(C, prims) - R).min() < margin:
            raise Reject(f"collide:{what}")
        C2, R2 = sc.robot_spheres(q, finger, links=[l for l in links if l not in ("panda_link0", "panda_link1")])
        if (C2[:, 2] - R2).min() < 0.005:
            raise Reject(f"table:{what}")


def check_held_mug(Ts_ee, M_G, samples, prims, what, margin):
    inv_MG = G.inv_T(M_G)
    for T in Ts_ee:
        TM = T @ inv_MG
        P = samples @ TM[:3, :3].T + TM[:3, 3]
        if P[:, 2].min() < 0.003:
            raise Reject(f"mug_table:{what}")
        if prims and G.sdf_prims(P, prims).min() < margin:
            raise Reject(f"mug_collide:{what}")


# -------------------------------------------------------------- episodes
def place_scene(sc, layout):
    sc.park_all()
    sc.set_rack(layout["T_rack"])
    for d in layout["distractors"]:
        sc.set_distractor(d["name"], d["T"])


def run_pick(sc, layout, rng):
    m = G.MUGS[layout["mug_key"]]
    T_mug = layout["T_mug"]
    prims = obstacles(layout, include_mug=True)
    for flip in (False, True):
        M_G = G.mug_T_MG(m, flip=flip)
        T_grasp = T_mug @ M_G
        try:
            q_grasp = solve_ik(sc, T_grasp, "grasp")
            break
        except Reject:
            if flip:
                raise
    T_end = G.trans([*rng.normal(0, 0.015, 2), rng.uniform(*PRE_GRASP)]) @ T_grasp
    T_end[:3, :3] = G.rot_axis(rng.normal(size=3), np.deg2rad(rng.uniform(*PRE_GRASP_ROT))) @ T_end[:3, :3]
    off = np.array([-rng.uniform(0.0, 0.6), rng.uniform(-0.5, 0.5), rng.uniform(0.5, 1.0)])
    off *= rng.uniform(0.15, 0.20) / np.linalg.norm(off)
    R0 = G.rot_axis(rng.normal(size=3), np.deg2rad(rng.uniform(*START_ROT))) @ T_end[:3, :3]
    T_start = G.make_T(R0, T_end[:3, 3] + off)
    # solved backwards from the grasp so the approach ends in the configuration that reaches the mug
    q_up = solve_path(sc, [G.interp_T(T_grasp, T_end, s) for s in np.linspace(0, 1, 11)], q_grasp, "pregrasp")
    Td = [G.interp_T(T_start, T_end, s) for s in G.min_jerk(DENSE * (N_FRAMES - 1) + 1)]
    qd = solve_path(sc, Td[::-1], q_up[-1], "approach")[::-1]
    Ts, qs = Td[::DENSE], qd[::DENSE]
    check_robot(sc, qd, prims, 0.04, "approach")
    check_robot(sc, q_up[1:], prims[:-2], 0.04, "pregrasp", skip_links=("panda_hand", "fingers"))
    check_robot(sc, [q_grasp], prims[:-2], 0.04, "grasp", skip_links=("panda_hand", "fingers", "panda_link7"))
    sc.set_mug_vis(layout["mug_key"], T_mug)
    sc.set_robot(qs[0], 0.04)
    return dict(T_ee=np.array(Ts), q=qs, T_label=T_grasp, M_G=M_G, held=None, q_grasp=q_grasp)


def run_place(sc, layout, rng):
    key = layout["mug_key"]
    m = G.MUGS[key]
    prims = obstacles(layout)
    rim, depth = rng.uniform(*np.deg2rad([-15, 15])), G.GRASP_DEPTH + rng.uniform(-0.004, 0.004)
    err = None
    # both symmetric grasps; each solved forward from the grasp or backward from the release pose
    for flip in (False, True):
        for anchor in ("grasp", "target"):
            try:
                plan = plan_place(sc, layout, G.mug_T_MG(m, rim_angle=rim, depth=depth, flip=flip), prims, anchor)
                break
            except Reject as e:
                err = e
        else:
            continue
        break
    else:
        raise err
    T_rack, T_peg, M_G = layout["T_rack"], plan["T_peg"], plan["M_G"]
    T_target, Ts, qs, finger = plan["T_target"], plan["Ts"], plan["qs"], plan["finger"]
    peg = G.RACK["pegs"][G.RACK["target"]]
    a, b = G.peg_segment(peg)
    seg_w = (T_rack[:3, :3] @ a + T_rack[:3, 3], T_rack[:3, :3] @ b + T_rack[:3, 3])
    hang = sc.hang_test(key, T_target @ G.inv_T(M_G), T_peg, seg_w)
    if not hang["success"]:
        raise Reject("hang")
    # restore the capture state: robot at transport start, mug in the gripper
    place_scene(sc, layout)
    sc.set_mug_vis(key, Ts[0] @ G.inv_T(M_G))
    sc.set_robot(qs[0], finger)
    return dict(T_ee=np.array(Ts), q=qs, T_label=T_peg, T_ee_target=T_target, G=plan["G"], M_G=M_G, M_H=plan["M_H"],
                held=True, hang=hang, q_grasp=plan["q_grasp"], grasp=dict(rim_angle=rim, depth=depth, flip=bool(flip),
                                                                          anchor=plan["anchor"]))


def plan_place(sc, layout, M_G, prims, anchor):
    """Poses and a continuous joint path for grasp -> lift -> transport (20 frames) -> insert -> retreat."""
    m = G.MUGS[layout["mug_key"]]
    T_grasp = layout["T_mug"] @ M_G
    M_H = G.mug_T_MH(m)
    Gr = G.inv_T(M_G) @ M_H                                     # handle frame in the EE frame
    T_peg = layout["T_rack"] @ G.rack_T_peg()
    T_target = T_peg @ G.inv_T(Gr)
    T_pre = G.trans(PRE_PLACE * T_peg[:3, 2]) @ T_target
    T_lift = G.trans([0, 0, LIFT]) @ T_grasp
    finger = m["wall"] / 2 + 0.0005
    segs = dict(
        lift=[G.interp_T(T_grasp, T_lift, s) for s in np.linspace(0, 1, 11)],
        transport=bezier_path(T_lift, T_pre, T_peg[:3, 2], DENSE * (N_FRAMES - 1) + 1),
        insert=[G.interp_T(T_pre, T_target, s) for s in np.linspace(0, 1, 21)],
        retreat=[T_target @ G.trans([0, 0, -RETREAT * s]) for s in np.linspace(0, 1, 9)],
    )
    poses, idx = [], {}
    for name, seg in segs.items():               # consecutive segments share their end poses
        a = len(poses) - (1 if poses else 0)
        poses += seg if not poses else seg[1:]
        idx[name] = (a, len(poses))
    k = 0 if anchor == "grasp" else idx["insert"][1] - 1
    qk = solve_ik(sc, poses[k], f"{anchor}")
    after = solve_path(sc, poses[k + 1:], qk, f"track_{anchor}") if k + 1 < len(poses) else np.zeros((0, 7))
    before = solve_path(sc, poses[:k][::-1], qk, f"track_{anchor}")[::-1] if k > 0 else np.zeros((0, 7))
    q_all = np.vstack([before, qk[None], after])
    q = {n: q_all[a:b] for n, (a, b) in idx.items()}

    check_robot(sc, q["lift"][1:], prims, finger, "lift")
    check_robot(sc, q["transport"], prims, finger, "transport")
    check_robot(sc, q["insert"], prims, finger, "insert", margin=0.005)
    check_robot(sc, q["retreat"][1:], prims, 0.04, "retreat", margin=0.005)
    samples = G.mug_surface_samples(m, 800)
    check_held_mug(segs["transport"], M_G, samples, prims, "transport", 0.005)
    check_held_mug(segs["insert"], M_G, samples, prims, "insert", 0.0015)
    return dict(T_peg=T_peg, T_target=T_target, M_G=M_G, M_H=M_H, G=Gr, finger=finger, q_grasp=q["lift"][0],
                Ts=segs["transport"][::DENSE], qs=q["transport"][::DENSE], anchor=anchor)


# ---------------------------------------------------------------- output
def check_capture(P, S, layout, T_mug_now):
    """Captured rack/mug points must lie on the known geometry (catches stale renders and frame bugs)."""
    from isaac_scene import SEM
    rack = G.transform_prims(G.rack_collision_prims(), layout["T_rack"])
    mug = G.transform_prims(G.mug_collision_prims(G.MUGS[layout["mug_key"]]), T_mug_now)
    for name, prims in (("rack", rack), ("mug", mug)):
        Q = P[S == SEM[name]]
        if len(Q) < 200:
            raise Reject(f"few_{name}_points")
        d = np.abs(G.sdf_prims(Q[:: max(1, len(Q) // 3000)], prims))
        if np.percentile(d, 95) > 0.004:
            raise RuntimeError(f"captured {name} points do not match the scene (p95 {np.percentile(d, 95):.4f} m); stale render?")


def build_cloud(sc, ep_seed, layout, T_mug_now):
    P, C, S = sc.capture()
    keep = G.crop_mask(P)
    check_capture(P[keep], S[keep], layout, T_mug_now)
    P, C, S = G.voxel_down(P[keep], C[keep], S[keep], 0.002)
    if len(P) < 500:
        raise Reject("few_points")
    xyz, rgb, sem = G.assemble_cloud(P, C, S, ep_seed)
    return xyz, rgb, sem, (P.astype(np.float32), C.astype(np.float32), S.astype(np.int8))


def episode_arrays(task, ep, xyz_w, sem):
    from isaac_scene import SEM
    held = (sem == SEM[ep.get("held_label", "mug")]) if task == "place" else np.zeros(len(sem), bool)
    xyz = np.empty((N_FRAMES,) + xyz_w.shape)
    c, a = np.empty((N_FRAMES, 3)), np.empty((N_FRAMES, 9))
    ca, aa = np.empty((N_FRAMES, 3)), np.empty((N_FRAMES, 9))
    for k, T in enumerate(ep["T_ee"]):
        xyz[k] = G.to_ee(xyz_w, T)
        xyz[k][held] = G.to_ee(xyz_w[held], ep["T_ee"][0])
        c[k], a[k] = G.label_in_ee(ep["T_label"], T)
        if task == "place":
            ca[k], aa[k] = G.label_in_ee(ep["T_ee_target"], T)
    out = dict(xyz=xyz, seg_center=c, axes=a)
    if task == "place":
        out.update(ee_target_pos=ca, ee_target_axes=aa)
    return out


def save_raw(path, task, layout, ep, raw, seeds, cams):
    d = dict(task=task, setting=layout["setting"], mug_key=layout["mug_key"], seed=np.array(seeds),
             T_rack=layout["T_rack"], T_mug=layout["T_mug"], T_ee=ep["T_ee"], q=ep["q"], q_grasp=ep["q_grasp"],
             T_label=ep["T_label"], M_G=ep["M_G"], P_obj_w=raw[0], rgb_obj=raw[1], sem_obj=raw[2],
             cam_poses=cams, distractors=json.dumps([dict(name=x["name"], T=x["T"].tolist()) for x in layout["distractors"]]))
    if task == "place":
        d.update(T_ee_target=ep["T_ee_target"], G=ep["G"], M_H=ep["M_H"], hang=json.dumps(ep["hang"]),
                 grasp=json.dumps(ep["grasp"]))
    np.savez_compressed(path, **d)


def generate(sc, task, setting, n, base_seed, setting_idx, out_dir, kind, log):
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
            ep = run_pick(sc, layout, rng) if task == "pick" else run_place(sc, layout, rng)
            ep_seed = int(np.random.SeedSequence(seeds).generate_state(1)[0])
            T_mug_now = ep["T_ee"][0] @ G.inv_T(ep["M_G"]) if task == "place" else layout["T_mug"]
            xyz_w, rgb, sem, raw = build_cloud(sc, ep_seed, layout, T_mug_now)
            if np.linalg.norm(xyz_w[sem >= 0] - ep["T_label"][:3, 3], axis=1).min() > LABEL_SEEN:
                raise Reject("label_unseen")
        except Reject as e:
            stats[str(e)] = stats.get(str(e), 0) + 1
            continue
        i = len(eps)
        os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
        save_raw(os.path.join(out_dir, "raw", f"{kind}_{setting}_{i:02d}.npz"), task, layout, ep, raw, seeds, cams)
        eps.append((layout, ep, xyz_w, rgb, sem, seeds))
        extra = f" hang: slid {ep['hang']['hole_to_target']*100:.1f} cm" if task == "place" else ""
        print(f"[{task}/{kind}/{setting}] {i + 1}/{n} attempt {attempt} seed {seeds} objects={int((sem >= 0).sum())}{extra}")
    stats["accepted"], stats["attempts"], stats["seconds"] = len(eps), attempt, round(time.time() - t0, 1)
    log[f"{kind}/{setting}"] = stats
    print(f"[{task}/{kind}/{setting}] done: {stats}")
    return eps


def write_demo(task, eps, path):
    arrs = [episode_arrays(task, ep, xyz_w, sem) for (_, ep, xyz_w, _, sem, _) in eps]
    d = {k: np.stack([a[k] for a in arrs]) for k in arrs[0]}
    d["rgb"] = np.stack([np.broadcast_to(rgb, (N_FRAMES,) + rgb.shape) for (_, _, _, rgb, _, _) in eps]).astype(np.float64)
    if task == "place":
        d["grasp_in_ee"] = np.stack([ep["G"] for (_, ep, _, _, _, _) in eps])
    d["seeds"] = np.array([s for (*_, s) in eps])
    np.savez(path, **d)
    print("wrote", path, {k: v.shape for k, v in d.items()})


def write_eval(task, ep_tuple, path):
    layout, ep, xyz_w, rgb, sem, seeds = ep_tuple
    c, a = G.pose_to_label(ep["T_label"])
    d = dict(xyz=xyz_w, rgb=rgb, seg_center=c, axes=a, ee_pose=ep["T_ee"][0], seed=np.array(seeds),
             mug_key=layout["mug_key"], T_rack=layout["T_rack"], T_mug=layout["T_mug"])
    if task == "place":
        ct, at = G.pose_to_label(ep["T_ee_target"])
        d.update(ee_target_pos=ct, ee_target_axes=at, grasp_in_ee=ep["G"])
    np.savez(path, **d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["pick", "place"], required=True)
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
        from isaac_scene import MugRackScene
        sc = MugRackScene(app)
        os.makedirs(args.out, exist_ok=True)
        log = dict(task=args.task, seed=args.seed, argv=sys.argv)
        if not args.no_demo:
            eps = generate(sc, args.task, "training", args.demos, args.seed, 0, args.out, "demo", log)
            write_demo(args.task, eps, os.path.join(args.out, "demo.npz"))
        for s in args.settings:
            eps = generate(sc, args.task, s, args.eval_seeds, args.seed, SETTINGS.index(s), args.out, "eval", log)
            os.makedirs(os.path.join(args.out, "eval", s), exist_ok=True)
            for i, e in enumerate(eps):
                write_eval(args.task, e, os.path.join(args.out, "eval", s, f"seed_{i:02d}.npz"))
            write_eval(args.task, eps[0], os.path.join(args.out, f"{s}.npz"))
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
