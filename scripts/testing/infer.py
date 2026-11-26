import sys
sys.path.append(".")
import os
import torch
from networks import *
from omegaconf import OmegaConf
from utils.data_utils import downsample_table
import argparse
import numpy as np
from utils.utils import modified_gram_schmidt

def load_policies(exp_name, pick_or_place):
    all_cfg = OmegaConf.load(f"config/{exp_name}/{pick_or_place}/config.json")
    cfg_seg = all_cfg.seg
    cfg_mani = all_cfg.mani
    assert cfg_seg.device == cfg_mani.device, "Device mismatch between segmentation and manipulation networks!"

    model_dir = os.path.join("experiments", exp_name, pick_or_place)
    policy_seg = globals()[cfg_seg.model](voxel_size=cfg_seg.voxel_size, radius_threshold=cfg_seg.radius_threshold).float().to(cfg_seg.device)
    policy_seg.load_state_dict(torch.load(os.path.join(model_dir, "segnet.pth"), map_location=cfg_seg.device))
    policy_seg.eval()

    policy_mani = globals()[cfg_mani.model](voxel_size=cfg_mani.voxel_size, radius_threshold=cfg_mani.radius_threshold,
                                            feature_point_radius=cfg_mani.feature_point_radius).float().to(cfg_mani.device)
    policy_mani.load_state_dict(torch.load(os.path.join(model_dir, "maninet.pth"), map_location=cfg_mani.device))
    policy_mani.eval()
    return cfg_seg, cfg_mani, policy_seg, policy_mani


def predict(cfg_seg, cfg_mani, policy_seg, policy_mani, xyz, rgb, pcd_name=None, draw_pcd=True, save_ori_feature=True):
    """xyz, rgb: [N, 3] numpy arrays of one scene. Returns (pred_pos [3], pred_rot [3, 3] with columns = axes)."""
    if draw_pcd or save_ori_feature:
        os.makedirs("pcd/seg", exist_ok=True)
        os.makedirs("pcd/mani", exist_ok=True)
    input_xyz = torch.tensor(xyz).float().to(cfg_seg.device)
    input_rgb = torch.tensor(rgb).float().to(cfg_seg.device)

    # preprossing input to keep the same as training
    data = downsample_table({"xyz": input_xyz, "rgb": input_rgb})
    xyz_t = data["xyz"].unsqueeze(0)
    rgb_t = data["rgb"].unsqueeze(0)

    with torch.no_grad():
        ref_point = policy_seg(
            {"xyz": xyz_t, "rgb": rgb_t},
            draw_pcd=draw_pcd,
            pcd_name=pcd_name,
        )
        output_pos, output_direction = policy_mani(
            {"xyz": xyz_t, "rgb": rgb_t},
            reference_point=ref_point,
            distance_threshold=cfg_mani.distance_threshold,
            save_ori_feature=save_ori_feature,
            draw_pcd=draw_pcd,
            pcd_name=pcd_name,
        )
        out_dir_schmidt = modified_gram_schmidt(output_direction.reshape(-1, 3).T, to_cuda=True)

    return output_pos.detach().cpu().numpy().reshape(3), out_dir_schmidt.detach().cpu().numpy()


def main(args):
    cfg_seg, cfg_mani, policy_seg, policy_mani = load_policies(args.exp_name, args.pick_or_place)

    wd = os.path.join("experiments", args.exp_name, args.pick_or_place, args.setting)
    pcd_path = os.path.join("data", args.exp_name, args.pick_or_place, f"{args.setting}.npz")
    pcd = np.load(pcd_path)

    pred_pos, pred_rot = predict(cfg_seg, cfg_mani, policy_seg, policy_mani, pcd["xyz"], pcd["rgb"], pcd_name=args.setting)

    os.makedirs(wd, exist_ok=True)
    result_path = os.path.join(wd, f"{args.setting}_pred_pose.npz")
    np.savez(result_path,
             pred_pos=pred_pos,
             pred_rot=pred_rot,
            )
    print(f"Result saved to: {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp_name', type=str, default="mug")
    parser.add_argument('-pick_or_place', type=str, choices=["pick", "place"], default="pick")
    parser.add_argument('-setting', type=str, default='newpose')
    args = parser.parse_args()

    main(args)