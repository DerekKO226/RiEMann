"""Isaac Sim 5.1 side of the mug/rack generator: stage, cameras, capture, Franka kinematics, hang test.

Import this module only after a SimulationApp exists (see gen_mug_sim.py). Geometry and label
conventions live in geometry.py (numpy only).

Design notes
- The arm is kinematic for data generation: Lula IK gives joint positions that are written to the
  articulation for rendering (occlusion), and Lula collision spheres are checked against the scene
  primitives. Robot collisions are filtered out of PhysX.
- The mug seen by the cameras ("vis" mug) is a plain, physics-free Xform posed through USD.
  A separate invisible dynamic copy ("phys" mug, compound collider that keeps the handle hole)
  is teleported to the release pose for the hang test, then parked again.
"""
import os
import numpy as np
import yaml

import omni.replicator.core as rep
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema, Vt

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name
from isaacsim.storage.native import get_assets_root_path

import geometry as G

SEM = {"mug": 1, "rack": 2, "distractor": 3, "table": 4, "plane": 5, "shelf": 6}
FRANKA_USD = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
YCB = ["003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle",
       "008_pudding_box", "010_potted_meat_can", "036_wood_block", "061_foam_brick"]
PARK = np.array([-3.0, 0.0, 0.0])
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
HOME_Q = np.array([0.0, -0.6, 0.0, -2.4, 0.0, 1.9, 0.785])
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.02
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) - 0.02


# ----------------------------------------------------------------- USD helpers
def _gf_matrix(T):
    return Gf.Matrix4d(*np.asarray(T, float).T.reshape(-1).tolist())   # USD is row-vector


def _usd_to_T(m):
    return np.array(m, dtype=float).reshape(4, 4).T


def set_xform(prim, T):
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    op = ops[0] if len(ops) == 1 and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform else None
    if op is None:
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(_gf_matrix(T))


def world_T(prim):
    return _usd_to_T(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def make_material(stage, path, rgb, roughness=0.6):
    """rgb is an sRGB albedo; UsdPreviewSurface.diffuseColor is linear."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*srgb_to_linear(rgb).tolist()))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat, rgb):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def add_prim_geom(stage, path, p, collide, visible=True):
    """Author one geometry.py primitive (local frame) as a USD gprim."""
    kind = p[0]
    if kind == "box":
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(g)
        xf.AddTransformOp().Set(_gf_matrix(p[1]))
        xf.AddScaleOp().Set(Gf.Vec3f(*(2 * np.asarray(p[2])).tolist()))
    elif kind == "capsule":
        a, b, r = p[1], p[2], p[3]
        d = b - a
        L = np.linalg.norm(d)
        z = d / L
        x = np.cross([0, 1, 0], z) if abs(z[1]) < 0.9 else np.cross([1, 0, 0], z)
        x /= np.linalg.norm(x)
        g = UsdGeom.Capsule.Define(stage, path)
        g.CreateAxisAttr("Z"); g.CreateRadiusAttr(r); g.CreateHeightAttr(L)
        UsdGeom.Xformable(g).AddTransformOp().Set(_gf_matrix(G.make_T(np.column_stack([x, np.cross(z, x), z]), (a + b) / 2)))
    elif kind == "cyl":
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateAxisAttr("Z"); g.CreateRadiusAttr(p[2]); g.CreateHeightAttr(p[3])
        UsdGeom.Xformable(g).AddTransformOp().Set(_gf_matrix(p[1]))
    else:
        raise ValueError(kind)
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    if not visible:
        g.CreateVisibilityAttr("invisible")
    return g.GetPrim()


# ------------------------------------------------------------------- the scene
class MugRackScene:
    def __init__(self, app, n_cams=6, res=(640, 480), with_distractors=True, physics_dt=1 / 120):
        self.app, self.res = app, res
        enable_extension("isaacsim.robot_motion.motion_generation")
        app.update()
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)
        self.stage = self.world.stage
        self.assets = get_assets_root_path()
        self._lights()
        self._physics_material()
        self._table()
        self.phys_prims = {}
        self._build_objects()
        self.distractors = self._distractors() if with_distractors else {}
        self._franka()
        self._cameras(n_cams)
        self._collision_groups()
        self.world.reset()
        self.robot.initialize()
        self.robot.disable_gravity()
        self.dof = list(self.robot.dof_names)
        self.arm_idx = [self.dof.index(j) for j in ARM_JOINTS]
        self.finger_idx = [self.dof.index(j) for j in ("panda_finger_joint1", "panda_finger_joint2")]
        for k, rb in self.phys_prims.items():
            rb.initialize()
        self._kinematics()
        self.park_all()

    # ---- building
    def _lights(self):
        dome = UsdLux.DomeLight.Define(self.stage, "/World/lights/dome")
        dome.CreateIntensityAttr(1150.0)            # with RACK["albedo"]: rendered medians match data/mug/pick
        sun = UsdLux.DistantLight.Define(self.stage, "/World/lights/sun")
        sun.CreateIntensityAttr(2300.0); sun.CreateAngleAttr(2.0)
        set_xform(sun.GetPrim(), G.make_T(G.rot_axis([1, 0.3, 0], np.deg2rad(35))))

    def _physics_material(self):
        self.phys_mat = UsdShade.Material.Define(self.stage, "/World/Looks/contact")
        pm = UsdPhysics.MaterialAPI.Apply(self.phys_mat.GetPrim())
        pm.CreateStaticFrictionAttr(0.6); pm.CreateDynamicFrictionAttr(0.5); pm.CreateRestitutionAttr(0.0)

    def _table(self):
        p = add_prim_geom(self.stage, "/World/table", ("box", G.make_T(t=[0.5, 0, -0.025]), np.array([0.7, 0.8, 0.025])), collide=True)
        bind(p, make_material(self.stage, "/World/Looks/table", (0.45, 0.42, 0.38)), (0.45, 0.42, 0.38))
        add_labels(p, ["table"])
        add_prim_geom(self.stage, "/World/park_pad", ("box", G.make_T(t=PARK + [0, 0, -0.025]), np.array([0.5, 0.8, 0.025])),
                      collide=True, visible=False)

    def _build_objects(self):
        self._rack()
        self.mugs_vis = {k: self._mug_vis(k, m) for k, m in G.MUGS.items()}
        self.mugs_phys = {k: self._mug_phys(k, m) for k, m in G.MUGS.items()}

    def _env_paths(self):
        return ["/World/table", "/World/rack", "/World/park_pad"] + [f"/World/{k}_phys" for k in G.MUGS]

    def _rack(self):
        root = UsdGeom.Xform.Define(self.stage, "/World/rack").GetPrim()
        mat = make_material(self.stage, "/World/Looks/rack", G.RACK["albedo"], 0.7)
        bx, by, bz = G.RACK["base"]
        parts = [("box", G.make_T(t=[0, 0, bz / 2]), np.array([bx / 2, by / 2, bz / 2])),
                 ("cyl", G.make_T(t=[0, 0, (bz + G.RACK["post_h"]) / 2]), G.RACK["post_r"], G.RACK["post_h"] - bz)]
        for peg in G.RACK["pegs"]:
            a, b = G.peg_segment(peg)
            parts.append(("capsule", a, b, G.RACK["peg_r"]))
        for i, p in enumerate(parts):
            g = add_prim_geom(self.stage, f"/World/rack/part{i}", p, collide=True)
            bind(g, mat, G.RACK["albedo"])
            UsdShade.MaterialBindingAPI(g).Bind(self.phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        add_labels(root, ["rack"])
        self.rack_prim = root

    def _mug_vis(self, key, m):
        root = UsdGeom.Xform.Define(self.stage, f"/World/{key}_vis").GetPrim()
        V, F = G.mug_mesh(m)
        mesh = UsdGeom.Mesh.Define(self.stage, f"/World/{key}_vis/mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in V.tolist()]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(F)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(F.reshape(-1).tolist()))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        bind(mesh.GetPrim(), make_material(self.stage, f"/World/Looks/{key}", m["color"], 0.4), m["color"])
        add_labels(root, ["mug"])
        return root

    def _mug_phys(self, key, m):
        path = f"/World/{key}_phys"
        root = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(root)
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(m["mass"])
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        prb.CreateEnableCCDAttr(True)
        prb.CreateAngularDampingAttr(1.0)      # stands in for contact damping so a hung mug stops swinging
        for i, p in enumerate(G.mug_collision_prims(m)):
            g = add_prim_geom(self.stage, f"{path}/c{i}", p, collide=True, visible=False)
            PhysxSchema.PhysxCollisionAPI.Apply(g).CreateContactOffsetAttr(0.002)
            PhysxSchema.PhysxCollisionAPI(g).CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(g).Bind(self.phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        self.phys_prims[key] = SingleRigidPrim(path, name=f"{key}_phys")
        return root

    def _distractors(self):
        out = {}
        for i, name in enumerate(YCB):
            path = f"/World/distractors/ycb_{name}"
            root = UsdGeom.Xform.Define(self.stage, path).GetPrim()
            inner = UsdGeom.Xform.Define(self.stage, path + "/asset").GetPrim()
            inner.GetReferences().AddReference(self.assets + f"/Isaac/Props/YCB/Axis_Aligned/{name}.usd")
            add_labels(root, ["distractor"])
            out[name] = root
        return out

    def _franka(self):
        add_reference_to_stage(self.assets + FRANKA_USD, "/World/franka")
        self.robot = SingleArticulation("/World/franka", name="franka")
        self.world.scene.add(self.robot)

    def _collision_groups(self):
        """Robot links never collide with the environment (the arm is kinematic here)."""
        g_robot = UsdPhysics.CollisionGroup.Define(self.stage, "/World/collision_groups/robot")
        g_env = UsdPhysics.CollisionGroup.Define(self.stage, "/World/collision_groups/env")
        Usd.CollectionAPI(g_robot.GetPrim(), "colliders").CreateIncludesRel().AddTarget("/World/franka")
        for p in self._env_paths():
            Usd.CollectionAPI(g_env.GetPrim(), "colliders").CreateIncludesRel().AddTarget(p)
        g_robot.CreateFilteredGroupsRel().AddTarget(g_env.GetPath())

    def _cameras(self, n):
        """n RGB-D cameras on a ring, 1 m from the table centre, elevation 40/48 deg (plan 6.3; 6 as in the paper sim)."""
        self.cams, self.annot = [], []
        center, look = np.array([0.5, 0.0, 0.0]), np.array([0.5, 0.0, 0.12])
        W, H = self.res
        for i in range(n):
            az = np.deg2rad(45 + 360.0 * i / n)
            el = np.deg2rad(40 if i % 2 == 0 else 48)
            pos = center + 1.0 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            f = look - pos; f /= np.linalg.norm(f)
            r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
            u = np.cross(r, f)
            path = f"/World/cameras/cam{i}"
            cam = UsdGeom.Camera.Define(self.stage, path)
            cam.CreateFocalLengthAttr(18.0)
            cam.CreateHorizontalApertureAttr(20.955)
            cam.CreateVerticalApertureAttr(20.955 * H / W)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10.0))
            set_xform(cam.GetPrim(), G.make_T(np.column_stack([r, u, -f]), pos))
            rp = rep.create.render_product(path, (W, H))
            anns = {}
            for name, kw in [("rgb", {}), ("distance_to_image_plane", {}), ("semantic_segmentation", {"colorize": False})]:
                a = rep.AnnotatorRegistry.get_annotator(name, init_params=kw) if kw else rep.AnnotatorRegistry.get_annotator(name)
                a.attach([rp])
                anns[name] = a
            self.cams.append(cam.GetPrim())
            self.annot.append(anns)

    def _kinematics(self):
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self.ik = LulaKinematicsSolver(**cfg)
        assert list(self.ik.get_joint_names()) == ARM_JOINTS, self.ik.get_joint_names()
        with open(cfg["robot_description_path"]) as f:
            desc = yaml.safe_load(f)
        self.spheres = {}
        for entry in desc["collision_spheres"]:
            for link, sph in entry.items():
                self.spheres[link] = [(np.array(s["center"]), s["radius"]) for s in sph]

    # ---- posing
    def park_all(self):
        for k, p in self.mugs_vis.items():
            set_xform(p, G.trans(PARK + [0, 0.3 + 0.2 * list(G.MUGS).index(k), -1.0]))
        for i, (k, rb) in enumerate(self.phys_prims.items()):
            rb.set_world_pose(PARK + [0.2 * i, -0.4, 0.002], [1, 0, 0, 0])
            rb.set_linear_velocity(np.zeros(3)); rb.set_angular_velocity(np.zeros(3))
        for i, p in enumerate(self.distractors.values()):
            set_xform(p, G.trans(PARK + [0.1 * i, 0.6, -1.0]))

    def set_rack(self, T):
        set_xform(self.rack_prim, T)

    def set_mug_vis(self, key, T):
        set_xform(self.mugs_vis[key], T)

    def set_distractor(self, name, T):
        set_xform(self.distractors[name], T)

    def distractor_bbox(self, name):
        """Local axis-aligned bbox (min, max) of a YCB asset in its own (root) frame."""
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        root = self.distractors[name]
        Tw = world_T(root)
        rng = cache.ComputeWorldBound(root).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
        return lo - Tw[:3, 3], hi - Tw[:3, 3]

    def set_robot(self, q7, finger=0.04):
        q = np.zeros(len(self.dof))
        q[self.arm_idx] = q7
        q[self.finger_idx] = finger
        self.robot.set_joint_positions(q)
        self.robot.set_joint_velocities(np.zeros(len(self.dof)))
        self.robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))

    # ---- kinematics
    def fk_tcp(self, q7):
        p, R = self.ik.compute_forward_kinematics("panda_hand", np.asarray(q7, float))
        return G.make_T(R, p) @ G.trans([0, 0, G.TCP_OFFSET])

    def ik_tcp(self, T_tcp, warm=None, pos_tol=1e-3, ori_tol=1e-2):
        T_hand = T_tcp @ G.trans([0, 0, -G.TCP_OFFSET])
        q, ok = self.ik.compute_inverse_kinematics("panda_hand", T_hand[:3, 3], G.quat_wxyz(T_hand[:3, :3]),
                                                   warm_start=warm, position_tolerance=pos_tol, orientation_tolerance=ori_tol)
        return np.asarray(q, float), bool(ok)

    def fk_hand(self, q7):
        p, R = self.ik.compute_forward_kinematics("panda_hand", np.asarray(q7, float))
        return G.make_T(R, p)

    def track_tcp(self, Ts, q0, pos_tol=1e-3, ori_tol=1e-2, iters=80, damping=0.02, max_dq=0.15):
        """Follow a dense TCP path from q0 with damped least squares (numeric Jacobian on Lula FK),
        so consecutive joint configurations stay continuous. Returns (qs[N,7], ok)."""
        q = np.asarray(q0, float).copy()
        mid = (JOINT_LO + JOINT_HI) / 2
        out = []
        for T in Ts:
            Th = T @ G.trans([0, 0, -G.TCP_OFFSET])
            ok = False
            for _ in range(iters):
                Tc = self.fk_hand(q)
                e = np.r_[Th[:3, 3] - Tc[:3, 3], G.log_R(Th[:3, :3] @ Tc[:3, :3].T)]
                if np.linalg.norm(e[:3]) < pos_tol * 0.5 and np.linalg.norm(e[3:]) < ori_tol * 0.5:
                    ok = True
                    break
                J = np.empty((6, 7))
                for j in range(7):
                    dqj = np.zeros(7); dqj[j] = 1e-5
                    Tj = self.fk_hand(q + dqj)
                    J[:3, j] = (Tj[:3, 3] - Tc[:3, 3]) / 1e-5
                    J[3:, j] = G.log_R(Tj[:3, :3] @ Tc[:3, :3].T) / 1e-5
                Jp = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
                dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.05 * (mid - q))
                n = np.abs(dq).max()
                if n > max_dq:
                    dq *= max_dq / n
                q = np.clip(q + dq, JOINT_LO, JOINT_HI)
            if not ok:
                return np.array(out), False
            out.append(q.copy())
        return np.array(out), True

    def robot_spheres(self, q7, finger=0.04, links=None):
        """World centres/radii of the Lula collision spheres (+2 per finger)."""
        C, Rr = [], []
        for link, sph in self.spheres.items():
            if links is not None and link not in links:
                continue
            p, R = self.ik.compute_forward_kinematics(link, np.asarray(q7, float))
            for c, r in sph:
                C.append(R @ c + p); Rr.append(r)
        if links is None or "fingers" in links:
            p, R = self.ik.compute_forward_kinematics("panda_hand", np.asarray(q7, float))
            for s in (1, -1):
                for z, r in ((0.075, 0.012), (0.098, 0.011)):
                    C.append(R @ np.array([0, s * (finger + 0.008), z]) + p); Rr.append(r)
        return np.array(C), np.array(Rr)

    # ---- capture
    def render(self, max_steps=6):
        """Synchronous render. world.step(render=True) delivers annotator data ~4 frames late, so
        physics is stepped first (pushes joint/pose writes to Fabric), then replicator renders until
        two consecutive segmentation images agree."""
        for _ in range(2):
            self.world.step(render=False)
        prev = None
        for _ in range(max_steps):
            rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
            cur = [np.asarray(a["semantic_segmentation"].get_data()["data"]).copy() for a in self.annot]
            if prev is not None and all(np.array_equal(x, y) for x, y in zip(prev, cur)):
                return
            prev = cur
        raise RuntimeError("render did not settle")

    def capture(self, labels=("mug", "rack", "distractor")):
        """Fuse all cameras into one world-frame cloud: (xyz[N,3], rgb[N,3] in 0-1, sem[N] codes)."""
        self.render()
        P, C, S = [], [], []
        W, H = self.res
        for cam, anns in zip(self.cams, self.annot):
            rgb = np.asarray(anns["rgb"].get_data())[..., :3].astype(np.float64) / 255.0
            depth = np.asarray(anns["distance_to_image_plane"].get_data()).astype(np.float64)
            seg = anns["semantic_segmentation"].get_data()
            ids = np.asarray(seg["data"]).astype(np.int64)
            lut = np.zeros(ids.max() + 1 if ids.size else 1, dtype=np.int64)
            for sid, lab in seg["info"]["idToLabels"].items():
                name = lab.get("class", "") if isinstance(lab, dict) else str(lab)
                for nm in name.split(","):
                    if nm.strip() in SEM and int(sid) < len(lut):
                        lut[int(sid)] = SEM[nm.strip()]
            sem = lut[ids]
            keep = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0) & np.isin(sem, [SEM[l] for l in labels])
            v, u = np.nonzero(keep)
            d = depth[v, u]
            gc = UsdGeom.Camera(cam).GetCamera(Usd.TimeCode.Default())
            fx = W * gc.focalLength / gc.horizontalAperture
            fy = H * gc.focalLength / gc.verticalAperture
            x = (u + 0.5 - W / 2) / fx * d
            y = -(v + 0.5 - H / 2) / fy * d
            pc = np.stack([x, y, -d, np.ones_like(d)], 0)
            Pw = (world_T(cam) @ pc)[:3].T
            P.append(Pw); C.append(rgb[v, u]); S.append(sem[v, u])
        return np.concatenate(P), np.concatenate(C), np.concatenate(S)

    # ---- hang test
    def hang_test(self, key, T_mug_release, T_rack_peg_w, peg_seg_w, seconds=3.0):
        """Drop the dynamic mug at its release pose; report whether it stays on the target peg."""
        m = G.MUGS[key]
        rb = self.phys_prims[key]
        for k in self.mugs_vis:
            set_xform(self.mugs_vis[k], G.trans(PARK + [0, 0, -1.0]))
        self.world.step(render=False)
        rb.set_world_pose(T_mug_release[:3, 3], G.quat_wxyz(T_mug_release[:3, :3]))
        rb.set_linear_velocity(np.zeros(3)); rb.set_angular_velocity(np.zeros(3))
        n = int(seconds / self.world.get_physics_dt())
        traj = []
        for i in range(n + int(0.5 / self.world.get_physics_dt())):
            self.world.step(render=False)
            if i == n - 1:
                p, q = rb.get_world_pose()
                T_end = G.make_T(G.R_from_quat_wxyz(q), p)
        p, q = rb.get_world_pose()
        T_late = G.make_T(G.R_from_quat_wxyz(q), p)
        MH = G.mug_T_MH(m)
        hole_late = (T_late @ MH)[:3, 3]
        hole = (T_end @ MH)[:3, 3]
        a, b = peg_seg_w
        ab = b - a
        t = float(np.clip((hole - a) @ ab / (ab @ ab), 0, 1))
        d_axis = float(np.linalg.norm(hole - (a + t * ab)))
        hole_axis = (T_end @ MH)[:3, 2]
        res = dict(
            hole_to_peg_axis=d_axis, hole_along_peg=t,
            hole_axis_alignment=float(abs(hole_axis @ (ab / np.linalg.norm(ab)))),
            hole_to_target=float(np.linalg.norm(hole - T_rack_peg_w[:3, 3])),
            mug_origin_z=float(T_end[2, 3]),
            settle_motion=float(np.linalg.norm(hole_late - hole)),       # hole centre, last 0.5 s
            swing_motion=float(np.linalg.norm(T_late[:3, 3] - T_end[:3, 3])),
        )
        _, half = G.handle_hole(m)
        res["success"] = bool(d_axis < half + m["handle_rt"] + 0.01 and 0.0 < t < 1.0
                              and res["hole_axis_alignment"] > 0.5 and T_end[2, 3] > 0.03
                              and res["settle_motion"] < 0.01)
        rb.set_world_pose(PARK + [0.2 * list(G.MUGS).index(key), -0.4, 0.002], [1, 0, 0, 0])
        rb.set_linear_velocity(np.zeros(3)); rb.set_angular_velocity(np.zeros(3))
        self.world.step(render=False)
        return res


# ------------------------------------------------------------- plane on shelf
class PlaneShelfScene(MugRackScene):
    """Same stage, cameras, robot and capture as MugRackScene; the objects are a box shelf with two
    support rails (geometry.SHELF) and the planes in geometry.PLANES (vis copy + invisible dynamic copy)."""

    def _build_objects(self):
        root = UsdGeom.Xform.Define(self.stage, "/World/shelf").GetPrim()
        mat = make_material(self.stage, "/World/Looks/shelf", G.SHELF["albedo"], 0.7)
        for i, p in enumerate(G.shelf_prims()):
            g = add_prim_geom(self.stage, f"/World/shelf/part{i}", p, collide=True)
            bind(g, mat, G.SHELF["albedo"])
            UsdShade.MaterialBindingAPI(g).Bind(self.phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        add_labels(root, ["shelf"])
        self.shelf_prim = root
        self.planes_vis, self.planes_phys = {}, {}
        for key, pl in G.PLANES.items():
            vroot = UsdGeom.Xform.Define(self.stage, f"/World/{key}_vis").GetPrim()
            pmat = make_material(self.stage, f"/World/Looks/{key}", pl["albedo"] if "albedo" in pl else pl["color"], 0.4)
            for i, p in enumerate(G.plane_prims(pl)):
                g = add_prim_geom(self.stage, f"/World/{key}_vis/part{i}", p, collide=False)
                bind(g, pmat, pl["color"])
            add_labels(vroot, ["plane"])
            self.planes_vis[key] = vroot
            path = f"/World/{key}_phys"
            proot = UsdGeom.Xform.Define(self.stage, path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(proot)
            mass = UsdPhysics.MassAPI.Apply(proot)
            mass.CreateMassAttr(pl["mass"])
            mass.CreateCenterOfMassAttr(Gf.Vec3f(pl["wing_x"], 0.0, pl["fus_r"]))
            prb = PhysxSchema.PhysxRigidBodyAPI.Apply(proot)
            prb.CreateEnableCCDAttr(True)
            prb.CreateAngularDampingAttr(1.0)
            for i, p in enumerate(G.plane_prims(pl)):
                g = add_prim_geom(self.stage, f"{path}/c{i}", p, collide=True, visible=False)
                PhysxSchema.PhysxCollisionAPI.Apply(g).CreateContactOffsetAttr(0.002)
                PhysxSchema.PhysxCollisionAPI(g).CreateRestOffsetAttr(0.0)
                UsdShade.MaterialBindingAPI.Apply(g).Bind(self.phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            self.phys_prims[key] = SingleRigidPrim(path, name=f"{key}_phys")
            self.planes_phys[key] = proot

    def _env_paths(self):
        return ["/World/table", "/World/shelf", "/World/park_pad"] + [f"/World/{k}_phys" for k in G.PLANES]

    def park_all(self):
        for i, p in enumerate(self.planes_vis.values()):
            set_xform(p, G.trans(PARK + [0, 0.3 + 0.4 * i, -1.0]))
        for i, rb in enumerate(self.phys_prims.values()):
            rb.set_world_pose(PARK + [0.4 * i, -0.4, 0.002], [1, 0, 0, 0])
            rb.set_linear_velocity(np.zeros(3)); rb.set_angular_velocity(np.zeros(3))
        for i, p in enumerate(self.distractors.values()):
            set_xform(p, G.trans(PARK + [0.1 * i, 0.6, -1.0]))

    def set_shelf(self, T):
        set_xform(self.shelf_prim, T)

    def set_plane_vis(self, key, T):
        set_xform(self.planes_vis[key], T)

    def capture(self, labels=("plane", "shelf", "distractor")):
        return super().capture(labels)

    def rest_test(self, key, T_plane_release, T_place_w, seconds=2.0):
        """Release the dynamic plane at its release pose; it must come to rest on the rails at the
        placement frame (wings on the rails, fuselage between them)."""
        pl = G.PLANES[key]
        rb = self.phys_prims[key]
        for p in self.planes_vis.values():
            set_xform(p, G.trans(PARK + [0, 0, -1.0]))
        self.world.step(render=False)
        rb.set_world_pose(T_plane_release[:3, 3], G.quat_wxyz(T_plane_release[:3, :3]))
        rb.set_linear_velocity(np.zeros(3)); rb.set_angular_velocity(np.zeros(3))
        dt = self.world.get_physics_dt()
        n = int(seconds / dt)
        for i in range(n + int(0.5 / dt)):
            self.world.step(render=False)
            if i == n - 1:
                p, q = rb.get_world_pose()
                T_end = G.make_T(G.R_from_quat_wxyz(q), p)
        p, q = rb.get_world_pose()
        T_late = G.make_T(G.R_from_quat_wxyz(q), p)
        PS = G.plane_T_PS(pl)
        S_end, S_late = T_end @ PS, T_late @ PS
        res = dict(pos_err=float(np.linalg.norm(S_end[:3, 3] - T_place_w[:3, 3])),
                   rot_err_deg=float(np.rad2deg(G.rot_angle(T_place_w[:3, :3].T @ S_end[:3, :3]))),
                   drop=float(T_place_w[2, 3] - S_end[2, 3]),
                   settle_motion=float(np.linalg.norm(S_late[:3, 3] - S_end[:3, 3])))
        res["success"] = bool(res["pos_err"] < 0.012 and res["rot_err_deg"] < 8.0 and res["settle_motion"] < 0.003)
        rb.set_world_pose(PARK + [0.4 * list(G.PLANES).index(key), -0.4, 0.002], [1, 0, 0, 0])
        rb.set_linear_velocity(np.zeros(3)); rb.set_angular_velocity(np.zeros(3))
        self.world.step(render=False)
        return res
