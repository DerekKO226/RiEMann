# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RiEMann (arXiv 2403.19460): SE(3)-equivariant robot manipulation that predicts a target end-effector pose (position + rotation) directly from a scene point cloud, without segmenting the point cloud first. It generalizes to new object poses from 5–10 demonstrations.

## Environment

The training environment is managed by pixi (`pixi.toml` + `pixi.lock`, linux-64 only). Run `pixi install` once, then `pixi run <task>`.
- The torch stack comes from **conda**, not pip:
  - Python 3.8, from conda-forge.
  - `pytorch` 2.0.1 (cuda11.8 build), `torchvision` 0.15.2 and `pytorch-cuda` 11.8, from the `pytorch` and `nvidia` channels.
  - `pytorch-scatter`, `pytorch-sparse` and `pytorch-cluster` (builds `py38_torch_2.0.*_cu118`), from the `pyg` channel.
  - `dgl` 1.1.2.cu118, from the `dglteam/label/cu118` channel.
- Only the pure-Python packages come from PyPI: `e3nn`, `omegaconf`, `open3d`, `potpourri3d` and `transforms3d`.
- Why conda:
  - pixi rejects plain `linux_x86_64` wheels, which is the only tag that the PyTorch 2.0.1+cu118 and PyG wheels have.
  - `data.dgl.ai` is unreachable from this network (TLS EOF).
- `mkl<2024.1` is pinned because torch < 2.3 breaks with newer MKL. `DGLBACKEND=pytorch` is set through `[activation.env]`.
- The paper tested CUDA 11.7. cu118 is used here for native RTX 4090 (sm_89) support.
- Intentionally left out of pixi: `apex` (the PyPI package with that name is unrelated to NVIDIA apex), plus `wandb`, `pynvml`, `opencv-python` and `dglgo`. Only the unused vendored `networks/se3_transformer/runtime/` scripts import them.
- `requirements.txt` and the README conda/pip steps are kept for non-pixi machines. For example, inference runs separately on a DGX Spark.
- A CUDA GPU is required. Configs hard-code `"device"` (currently `"cuda:0"`; upstream used `"cuda:1"`), so check it on multi-GPU machines.
- Data generation runs in a **separate** Python: Isaac Sim 5.1's (`isaac-python`, Python 3.11, numpy only on our side). Never install the DGL/PyG stack there or Isaac into pixi.

There is no test suite, linter or build step.

## Commands

Run every script from the repo root. Scripts do `sys.path.append(".")`, and all paths are relative to the root.

```bash
pixi run check                    # print torch/dgl versions and visible GPUs
pixi run train-seg mug pick       # args: exp task (default: mug pick)
pixi run train-mani mug pick
pixi run train mug pick           # both steps in order
```

Equivalent raw commands:

```bash
python scripts/training/train_seg.py --exp_name mug --pick_or_place pick    # step 1: SE3SegNet  -> experiments/mug/pick/segnet.pth
python scripts/training/train_mani.py --exp_name mug --pick_or_place pick   # step 2: SE3ManiNet -> experiments/mug/pick/maninet.pth
python scripts/testing/infer.py -exp_name mug -pick_or_place pick -setting newpose
python scripts/testing/eval_place.py -exp_name mug -pick_or_place place   # all eval/{setting}/seed_*.npz, errors vs ground truth
```

- The order matters: `train_mani.py` loads `segnet.pth` from the same experiment directory.
- `infer.py` uses **single-dash** flags (`-exp_name`), while the training scripts use double-dash flags.
- `scripts/testing/vis_*.py` are Open3D viewers with hard-coded placeholder paths (for example `"your_pcd_path"`) in `__main__`. Edit those paths before running them.

Isaac Sim data generation (plan and rationale: `place_p_cloud_plan.md`):

```bash
isaac-python scripts/data_gen/gen_mug_sim.py --task place --out data/mug/place      # demo.npz + 5 settings x 20 seeds + raw/
isaac-python scripts/data_gen/gen_mug_sim.py --task pick  --out data/mug_sim/pick
isaac-python scripts/data_gen/gen_plane_sim.py --out data/plane/place                 # Plane on Shelf (place only)
python scripts/data_gen/validate_npz.py data/mug/place/*.npz data/mug/place/eval/*/*.npz   # numpy only; exit 1 on FAIL
python scripts/data_gen/plot_npz.py data/mug/place/demo.npz --out /tmp/demo.png           # headless visual check
isaac-python scripts/data_gen/closed_loop_place.py --exp mug [--gt]                    # needs eval_place.py predictions
```

- `geometry.py` is numpy-only (frames, mug/rack models, label conventions, FPS); `isaac_scene.py` needs a running SimulationApp. Layouts, grasps, trajectories and point sampling follow from the seed list `[seed, setting, demo/eval, attempt]` stored in every output (rendering and physics are not guaranteed bit-identical).
- Capture must go through `MugRackScene.render()` (physics step, then `rep.orchestrator.step` until two renders agree): `world.step(render=True)` delivers annotator data ~4 frames late. `gen_mug_sim.check_capture` aborts if captured points do not lie on the known geometry.
- `gen_plane_sim.py` reuses the mug pipeline (`PlaneShelfScene` subclasses `MugRackScene`): procedural planes sized/coloured after `data/plane/pick` (`PLANES[...]['albedo']` is the calibrated render albedo), a box shelf with two wing-support rails and a back panel (`SHELF`). Label B = `T_place` on the box top between the rails; success = resting within 1.2 cm / 8 deg (the rails physically hold the plane even 25 deg off, so the thresholds decide).
- Rendered colours are calibrated to `data/mug/pick` through the light intensities and `RACK["albedo"]`; changing lights, materials or camera count changes colours, so re-check the per-class medians.

## Layout and data flow

Everything is keyed by `{exp_name}/{pick_or_place}`:
- Config: `config/{exp}/{task}/config.json`, loaded with OmegaConf. It has a `seg` section and a `mani` section. Configs exist for `mug/pick` (real data), `mug/place`, `mug_sim/pick` and `plane/place` (Isaac Sim data); `data/plane/pick` has data but no config yet.
- Data: `data/{exp}/{task}/*.npz`. Training reads `demo.npz` (`xyz`/`rgb` `[traj, T, N, 3]`, `seg_center` `[traj, T, 3]`, `axes` `[traj, T, 9]` = the 3 rotation-matrix column vectors flattened). `SE3Demo` keeps only the first 20 frames of each trajectory and flattens trajectories × frames into samples. The other files (`newpose`, `instance`, `distracting`, `all`, `training`) are evaluation settings used by `infer.py -setting`.
- Outputs: `experiments/{exp}/{task}/` (checkpoints) and `pcd/seg/`, `pcd/mani/` (debug `.pcd` heatmaps and `ori_feature_*.pt` written when `draw_pcd` / `save_ori_feature` is on). Both directories are gitignored.

Model pipeline (`networks/equinet.py`):
1. `SE3SegNet` (φ, SE(3)-invariant): outputs one scalar saliency heatmap per point and returns a heatmap-weighted reference point.
2. `SE3ManiNet` (ψ): crops the cloud to `distance_threshold` around the reference point. `pos_net` (type-0 heatmap) gives the position. `ori_net` (three type-1 vectors) is pooled within `feature_point_radius` of that position to give the 9-D orientation. `infer.py` orthonormalizes it with `utils.utils.modified_gram_schmidt`.
3. During ManiNet training, `mani.ref_point` selects whether the crop centre is the ground-truth `seg_center` (`"gt"`) or the SegNet output (`"seg_net"`).

Both networks are built from `networks/se3_backbone.py:SE3Backbone`. It voxelizes the input (`voxel_size`), builds a radius graph (`radius_threshold`, `utils/utils.py:build_graph`), and runs the vendored NVIDIA DGL SE(3)-Transformer in `networks/se3_transformer/`. Treat that directory as third-party code.

Scripts that use config strings rely on these conventions:
- `cfg.model` is resolved with `globals()[cfg.model]` after `from networks import *`. A new model must be exported from `networks/__init__.py`.
- `aug_methods` entries are function names in `utils/data_utils.py`, also resolved through `globals()`. Add a new augmentation as a top-level function there that takes and returns the `data` dict.
- `downsample_table` (always applied at inference) removes table points by matching a hard-coded table RGB (`[0.1176, 0.4392, 0.4078]`) and downsamples to 2500 points. New scenes with a different table color need this adjusted.

## Data notes

- `data/plane/pick` (real): the table is real-coloured, so `downsample_table` removes only ~1800 of ~6500 table points; the network input is ~24% plane / ~76% table (mug: ~60% object). Its eval files (`training.npz` ...) also contain a mug rack at (0.69, 0) that never appears in `demo.npz`.
- `data/mug/pick/demo.npz` trajectory 7 has a degenerate label (`seg_center=0`, `axes=I` in every frame); exclude it when training on the real pick data.
- `infer.py` previously unpacked 5 values from `SE3ManiNet.forward`, did not create its output directory and ignored `feature_point_radius`; all three are fixed, and `load_policies` / `predict` are reused by `eval_place.py`.
