"""Numpy-only geometry shared by the Isaac Sim mug/rack data generator.

Nothing here imports Isaac Sim, so it can be unit-checked in any Python with numpy.

Frames (all 4x4 homogeneous, column-vector convention, metres):
  world  = robot base frame, table surface at z = 0.
  M      = mug frame: origin at the bottom centre, +z = mug axis (up), handle along +x.
  H      = handle frame in M: origin at the centre of the handle hole, z = hole axis (-y_M),
           x = radial (+x_M), y = mug axis (+z_M).  When hung, T_H == T_peg.
  Rk     = rack frame: origin at the bottom centre of the base, +z up.
  peg    = placement frame on the target peg: origin on the peg axis, s metres from the post
           axis; z = peg direction (root -> tip), x = up, perpendicular to the peg in the
           vertical plane, y = z x x (horizontal).
  TCP    = Franka TCP: panda_hand translated +TCP_OFFSET along its z (approach) axis;
           y = finger closing axis.
"""
import numpy as np

TABLE_RGB = np.array([0.1176, 0.4392, 0.4078])   # utils/data_utils.py:downsample_table
N_TABLE, N_OBJ = 6144, 2048
WS_X, WS_Y = (0.2, 0.8), (-0.3, 0.3)
CROP_Z = (-0.03, 0.6)                             # plan C11
TCP_OFFSET = 0.1034                               # plan C10: panda_hand -> TCP along +z


# --------------------------------------------------------------------------- SE(3)
def make_T(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_axis(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def rot_z(a):
    return rot_axis([0, 0, 1], a)


def trans(t):
    return make_T(t=np.asarray(t, float))


def log_R(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-8:
        return np.zeros(3)
    if np.pi - a < 1e-6:                          # 180 deg: axis from the symmetric part
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        return V[:, -1] * a
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def exp_R(w):
    a = np.linalg.norm(w)
    return np.eye(3) if a < 1e-12 else rot_axis(w / a, a)


def rot_angle(R):
    return np.linalg.norm(log_R(R))


def interp_T(T0, T1, s):
    """Straight-line translation + geodesic (slerp) rotation."""
    R = T0[:3, :3] @ exp_R(s * log_R(T0[:3, :3].T @ T1[:3, :3]))
    return make_T(R, (1 - s) * T0[:3, 3] + s * T1[:3, 3])


def min_jerk(n):
    tau = np.linspace(0, 1, n)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def quat_wxyz(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return q / np.linalg.norm(q)


def R_from_quat_wxyz(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------- RiEMann label conventions
def to_ee(P_w, T_ee):
    """World points -> EE frame:  R^T (p - t)."""
    return (P_w - T_ee[:3, 3]) @ T_ee[:3, :3]


def pose_to_label(T):
    """4x4 -> (seg_center[3], axes[9]); axes = R.T.reshape(9) so that R == axes.reshape(3,3).T."""
    return T[:3, 3].copy(), T[:3, :3].T.reshape(9).copy()


def label_in_ee(T_label_w, T_ee):
    return pose_to_label(inv_T(T_ee) @ T_label_w)


# ------------------------------------------------------------------------ mugs
MUGS = {
    # matches data/mug/pick: height 0.108 m, body ~8.5 cm, pink
    "mug_a": dict(height=0.108, radius=0.0425, wall=0.005, bottom=0.008,
                  handle_cx=0.0345, handle_rh=0.035, handle_rt=0.006, handle_z=0.058,
                  color=(0.80, 0.63, 0.59), mass=0.30),
    # second instance for the NI / ALL settings: taller, thinner, yellow
    "mug_b": dict(height=0.150, radius=0.036, wall=0.0045, bottom=0.008,
                  handle_cx=0.028, handle_rh=0.042, handle_rt=0.0065, handle_z=0.085,
                  color=(0.86, 0.76, 0.32), mass=0.35),
}
GRASP_DEPTH = 0.024          # TCP below the rim top (data/mug/pick: 2.4 cm)


def handle_end_angle(m):
    """Arc half-angle where the handle centreline meets the outer wall."""
    return np.arccos((m["radius"] - m["handle_cx"]) / m["handle_rh"])


def handle_hole(m):
    """(centre x in M, half width along x) of the free hole between wall and handle tube."""
    inner = m["handle_cx"] + m["handle_rh"] - m["handle_rt"]
    return 0.5 * (inner + m["radius"]), 0.5 * (inner - m["radius"])


def mug_T_MH(m):
    x_h, _ = handle_hole(m)
    R = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]])   # x_H=x_M, y_H=z_M, z_H=-y_M
    return make_T(R, [x_h, 0, m["handle_z"]])


def mug_T_MG(m, rim_angle=0.0, depth=GRASP_DEPTH, flip=False):
    """TCP grasp pose in M: on the rim opposite the handle (+rim_angle), approach straight down,
    fingers closing radially. flip rotates 180 deg about the approach axis (symmetric gripper)."""
    phi = np.pi + rim_angle
    e = np.array([np.cos(phi), np.sin(phi), 0.0])
    z = np.array([0, 0, -1.0])
    y = -e if not flip else e
    x = np.cross(y, z)
    p = (m["radius"] - m["wall"] / 2) * e + np.array([0, 0, m["height"] - depth])
    return make_T(np.column_stack([x, y, z]), p)


def mug_mesh(m, n=48, n_arc=24, n_tube=12):
    """Visual triangle mesh of the mug in M. Returns (vertices[V,3], faces[F,3])."""
    R, Ri, H, B = m["radius"], m["radius"] - m["wall"], m["height"], m["bottom"]
    V, F = [], []

    def add(verts):
        V.extend(verts)
        return len(V) - len(verts)

    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda r, z: [(r * np.cos(t), r * np.sin(t), z) for t in th]
    o0, o1 = add(ring(R, 0)), add(ring(R, H))
    i0, i1 = add(ring(Ri, B)), add(ring(Ri, H))
    for k in range(n):
        k2 = (k + 1) % n
        F += [(o0 + k, o0 + k2, o1 + k2), (o0 + k, o1 + k2, o1 + k)]      # outer wall
        F += [(i0 + k, i1 + k2, i0 + k2), (i0 + k, i1 + k, i1 + k2)]      # inner wall
        F += [(o1 + k, o1 + k2, i1 + k2), (o1 + k, i1 + k2, i1 + k)]      # rim
    cb, ci = add([(0, 0, 0)]), add([(0, 0, B)])
    for k in range(n):
        k2 = (k + 1) % n
        F += [(cb, o0 + k2, o0 + k), (ci, i0 + k, i0 + k2)]               # bottoms

    te = handle_end_angle(m)
    cx, rh, rt, hz = m["handle_cx"], m["handle_rh"], m["handle_rt"], m["handle_z"]
    base = len(V)
    for a in np.linspace(-te, te, n_arc):
        c = np.array([cx + rh * np.cos(a), 0, hz + rh * np.sin(a)])
        n1, n2 = np.array([np.cos(a), 0, np.sin(a)]), np.array([0, 1.0, 0])
        for p in np.linspace(0, 2 * np.pi, n_tube, endpoint=False):
            V.append(tuple(c + rt * (np.cos(p) * n1 + np.sin(p) * n2)))
    for i in range(n_arc - 1):
        for j in range(n_tube):
            a, b = base + i * n_tube + j, base + i * n_tube + (j + 1) % n_tube
            c, d = a + n_tube, b + n_tube
            F += [(a, b, d), (a, d, c)]
    return np.array(V), np.array(F, dtype=np.int32)


def mug_collision_prims(m, n_wall=16, n_handle=7):
    """Compound collider in M that keeps the handle hole open.
    Returns list of ('box', T_center, half_extents) / ('capsule', p0, p1, r) / ('cyl', T_center, r, h)."""
    R, w, H, B = m["radius"], m["wall"], m["height"], m["bottom"]
    prims = [("cyl", make_T(t=[0, 0, B / 2]), R, B)]
    chord = 2 * R * np.sin(np.pi / n_wall) * 1.08
    for k in range(n_wall):
        a = 2 * np.pi * k / n_wall
        Tk = make_T(rot_z(a), [(R - w / 2) * np.cos(a), (R - w / 2) * np.sin(a), (B + H) / 2])
        prims.append(("box", Tk, np.array([w / 2, chord / 2, (H - B) / 2])))
    te = handle_end_angle(m)
    cx, rh, rt, hz = m["handle_cx"], m["handle_rh"], m["handle_rt"], m["handle_z"]
    pts = [np.array([cx + rh * np.cos(a), 0, hz + rh * np.sin(a)]) for a in np.linspace(-te, te, n_handle + 1)]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        prims.append(("capsule", p0, p1, rt))
    return prims


def mug_surface_samples(m, n=600, seed=0):
    V, _ = mug_mesh(m, n=32, n_arc=16, n_tube=8)
    rng = np.random.default_rng(seed)
    return V[rng.choice(len(V), min(n, len(V)), replace=False)]


# ------------------------------------------------------------------------ rack
RACK = dict(
    base=(0.18, 0.18, 0.015), post_r=0.0125, post_h=0.346, peg_r=0.006, peg_len=0.12,
    # yaw in the rack frame (0 = +x_rack), elevation above horizontal
    pegs=[dict(z=0.25, yaw=np.pi, elev=np.deg2rad(45)),     # target: highest peg
          dict(z=0.16, yaw=0.0, elev=np.deg2rad(45))],
    target=0, s=0.055,             # hung mugs slide ~1 cm further down to the post
    color=(0.62, 0.53, 0.37),       # rendered colour to match (data/mug/pick)
    albedo=(0.31, 0.25, 0.15),      # sRGB albedo that renders as `color` under the scene lights
)


def peg_dir(peg):
    h = np.array([np.cos(peg["yaw"]), np.sin(peg["yaw"]), 0])
    return np.cos(peg["elev"]) * h + np.array([0, 0, np.sin(peg["elev"])]), h


def peg_segment(peg, rack=RACK):
    """Peg capsule segment in Rk (from the post axis to the tip)."""
    d, _ = peg_dir(peg)
    p0 = np.array([0, 0, peg["z"]])
    return p0, p0 + d * (rack["peg_len"] + rack["post_r"])


def rack_T_peg(rack=RACK):
    peg = rack["pegs"][rack["target"]]
    d, h = peg_dir(peg)
    x = -np.sin(peg["elev"]) * h + np.array([0, 0, np.cos(peg["elev"])])
    y = np.cross(d, x)
    return make_T(np.column_stack([x, y, d]), np.array([0, 0, peg["z"]]) + rack["s"] * d)


def rack_collision_prims(rack=RACK):
    bx, by, bz = rack["base"]
    prims = [("box", make_T(t=[0, 0, bz / 2]), np.array([bx / 2, by / 2, bz / 2])),
             ("capsule", np.array([0, 0, bz]), np.array([0, 0, rack["post_h"] - rack["post_r"]]), rack["post_r"])]
    for peg in rack["pegs"]:
        p0, p1 = peg_segment(peg, rack)
        prims.append(("capsule", p0, p1, rack["peg_r"]))
    return prims


# ----------------------------------------------------------- distance queries
def transform_prims(prims, T):
    out = []
    for p in prims:
        if p[0] == "box":
            out.append(("box", T @ p[1], p[2]))
        elif p[0] == "capsule":
            out.append(("capsule", T[:3, :3] @ p[1] + T[:3, 3], T[:3, :3] @ p[2] + T[:3, 3], p[3]))
        elif p[0] == "cyl":                          # conservative: bounding box of the cylinder
            out.append(("box", T @ p[1], np.array([p[2], p[2], p[3] / 2])))
    return out


def sdf_prims(P, prims):
    """Signed distance from points P[N,3] to the union of primitives (negative inside)."""
    P = np.atleast_2d(P)
    d = np.full(len(P), np.inf)
    for p in prims:
        if p[0] == "box":
            q = np.abs((P - p[1][:3, 3]) @ p[1][:3, :3]) - p[2]
            dd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        else:
            a, b, r = p[1], p[2], p[3]
            ab = b - a
            t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            dd = np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - r
        d = np.minimum(d, dd)
    return d


# ------------------------------------------------------------ point clouds
def voxel_down(P, C, S, voxel):
    if len(P) == 0:
        return P, C, S
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[idx], C[idx], S[idx]


def fps(P, k, seed=0):
    """Farthest point sampling; returns indices. Pads with random repeats if len(P) < k."""
    n = len(P)
    rng = np.random.default_rng(seed)
    if n <= k:
        extra = rng.choice(n, k - n, replace=True) if n else np.zeros(0, int)
        return np.concatenate([np.arange(n), extra]).astype(int)
    idx = np.empty(k, dtype=int)
    idx[0] = rng.integers(n)
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(P - P[idx[i]], axis=1))
    return idx


def synthetic_table(seed):
    rng = np.random.default_rng(seed)
    xyz = np.stack([rng.uniform(*WS_X, N_TABLE), rng.uniform(*WS_Y, N_TABLE),
                    rng.uniform(-0.01, 0.01, N_TABLE)], 1)
    return xyz, np.tile(TABLE_RGB, (N_TABLE, 1))


def crop_mask(P):
    return ((P[:, 0] >= WS_X[0]) & (P[:, 0] <= WS_X[1]) & (P[:, 1] >= WS_Y[0]) & (P[:, 1] <= WS_Y[1])
            & (P[:, 2] >= CROP_Z[0]) & (P[:, 2] <= CROP_Z[1]))


def assemble_cloud(P_obj, C_obj, S_obj, seed):
    """[table(6144); objects(2048)] in world. Returns xyz, rgb, sem (sem=-1 for table)."""
    i = fps(P_obj, N_OBJ, seed)
    tx, tc = synthetic_table(seed)
    xyz = np.concatenate([tx, P_obj[i]]).astype(np.float64)
    rgb = np.clip(np.concatenate([tc, C_obj[i]]), 0, 1).astype(np.float64)
    sem = np.concatenate([np.full(N_TABLE, -1), S_obj[i]])
    return xyz, rgb, sem


# ------------------------------------------------------------- plane on shelf
# Frames:
#   P  = plane frame: origin under the wing centre on the fuselage bottom (table contact),
#        +x = nose, +y = left wing, +z = up.  Fuselage axis at z = fus_r.
#   Sh = shelf frame: origin at the bottom centre of the base box, +z up; the two support rails run
#        along +x_Sh, the back panel is on the -y_Sh side (breaks the 180 deg yaw symmetry).
#   place = placement frame (label B): origin on the base-box top midway between the rails (an
#        observed surface, so ManiNet can pool features there), axes = Sh axes. A correctly placed
#        plane has its nose along +x_place, wings resting on the rails.
PLANES = {
    # matches data/plane/pick (blue, ~0.34 m long, ~0.32 m span, ~8 cm tall)
    "plane_a": dict(length=0.34, nose_x=0.14, fus_r=0.022, span=0.32, chord=0.07, wing_t=0.008, wing_x=0.0,
                    stab_span=0.12, stab_chord=0.045, fin_h=0.040, fin_chord=0.05,
                    color=(0.176, 0.427, 0.647), albedo=(0.01, 0.194, 0.359), mass=0.15),   # renders (0.24, 0.43, 0.66)
    # NI / ALL instance (data/plane/pick instance.npz: green, narrower span)
    "plane_b": dict(length=0.35, nose_x=0.15, fus_r=0.020, span=0.26, chord=0.065, wing_t=0.008, wing_x=0.0,
                    stab_span=0.11, stab_chord=0.042, fin_h=0.045, fin_chord=0.05,
                    color=(0.322, 0.651, 0.471), albedo=(0.124, 0.379, 0.234), mass=0.14),  # renders (0.33, 0.67, 0.47)
}
PLANE_GRASP_BEHIND = 0.03     # TCP this far behind the wing trailing edge, on the fuselage

SHELF = dict(
    base=(0.20, 0.40, 0.10),          # box (x, y, z)
    rail=(0.06, 0.02, 0.035),         # each support rail (length x, width y, height z)
    rail_gap=0.07,                    # between the rails' inner faces (fuselage drops in between)
    panel=(0.015, 0.15),              # back panel (thickness, height above the box top), at -y
    release_gap=0.003,                # wings released this far above the rails
    color=(0.62, 0.53, 0.37), albedo=(0.31, 0.25, 0.15),   # same wood as RACK
)


def _plane_parts(p):
    r, t = p["fus_r"], p["wing_t"]
    tail_x = p["nose_x"] - p["length"]
    stab_x = tail_x + p["stab_chord"] / 2 + 0.008
    fin_lo, fin_hi = r, 2 * r + p["fin_h"]
    return dict(
        fuselage=("capsule", np.array([tail_x + r, 0, r]), np.array([p["nose_x"] - r, 0, r]), r),
        wing=("box", make_T(t=[p["wing_x"], 0, r]), np.array([p["chord"] / 2, p["span"] / 2, t / 2])),
        stab=("box", make_T(t=[stab_x, 0, r]), np.array([p["stab_chord"] / 2, p["stab_span"] / 2, t / 2])),
        fin=("box", make_T(t=[stab_x, 0, (fin_lo + fin_hi) / 2]), np.array([p["fin_chord"] / 2, t / 2, (fin_hi - fin_lo) / 2])),
    )


def plane_prims(p):
    """Visual = collision primitives of the plane in P."""
    return list(_plane_parts(p).values())


def plane_T_PG(p, flip=False):
    """TCP grasp pose in P: on the fuselage behind the wing, approach straight down, fingers closing
    across the fuselage (y). flip = 180 deg about the approach axis."""
    x = p["wing_x"] - p["chord"] / 2 - PLANE_GRASP_BEHIND
    z = np.array([0, 0, -1.0])
    y = np.array([0, 1.0, 0]) * (-1 if flip else 1)
    return make_T(np.column_stack([np.cross(y, z), y, z]), [x, 0, p["fus_r"]])


def plane_T_PS(p, shelf=None):
    """The placement frame expressed in P at the release pose (wings release_gap above the rails)."""
    shelf = shelf or SHELF
    z = p["fus_r"] - p["wing_t"] / 2 - shelf["rail"][2] - shelf["release_gap"]
    return make_T(t=[p["wing_x"], 0, z])


def shelf_prims(shelf=None):
    shelf = shelf or SHELF
    bx, by, bz = shelf["base"]
    rl, rw, rh = shelf["rail"]
    pt, ph = shelf["panel"]
    yc = shelf["rail_gap"] / 2 + rw / 2
    return [("box", make_T(t=[0, 0, bz / 2]), np.array([bx / 2, by / 2, bz / 2])),
            ("box", make_T(t=[0, yc, bz + rh / 2]), np.array([rl / 2, rw / 2, rh / 2])),
            ("box", make_T(t=[0, -yc, bz + rh / 2]), np.array([rl / 2, rw / 2, rh / 2])),
            ("box", make_T(t=[0, -(by / 2 - pt / 2), bz + ph / 2]), np.array([bx / 2, pt / 2, ph / 2]))]


def shelf_T_place(shelf=None):
    shelf = shelf or SHELF
    return make_T(t=[0, 0, shelf["base"][2]])


def prims_surface_samples(prims, n=800, seed=0):
    """Area-weighted random points on the surfaces of box / capsule primitives."""
    rng = np.random.default_rng(seed)
    areas = []
    for p in prims:
        if p[0] == "box":
            a, b, c = 2 * np.asarray(p[2])
            areas.append(2 * (a * b + b * c + a * c))
        else:
            L, r = np.linalg.norm(p[2] - p[1]), p[3]
            areas.append(2 * np.pi * r * L + 4 * np.pi * r * r)
    cnt = rng.multinomial(n, np.array(areas) / sum(areas))
    out = []
    for p, k in zip(prims, cnt):
        if k == 0:
            continue
        if p[0] == "box":
            h = np.asarray(p[2])
            q = rng.uniform(-1, 1, (k, 3)) * h
            ax = rng.choice(3, k, p=[h[1] * h[2], h[0] * h[2], h[0] * h[1]] / (h[1] * h[2] + h[0] * h[2] + h[0] * h[1]))
            q[np.arange(k), ax] = np.sign(rng.uniform(-1, 1, k)) * h[ax]
            out.append(q @ p[1][:3, :3].T + p[1][:3, 3])
        else:
            a, b, r = p[1], p[2], p[3]
            u = (b - a) / np.linalg.norm(b - a)
            d = rng.normal(size=(k, 3))
            d -= (d @ u)[:, None] * u                   # radial: cylinder part (caps are small, skipped)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            out.append(a + rng.uniform(0, 1, (k, 1)) * (b - a) + r * d)
    P = np.concatenate(out)
    keep = sdf_prims(P, prims) > -1e-6                 # drop samples buried inside other parts
    return P[keep]
