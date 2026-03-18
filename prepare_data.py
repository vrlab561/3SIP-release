# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os

import numpy as np
import torch

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from tqdm import tqdm
from utils import utils_transform

device = "cuda"

def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def main(args, bm):
    min_seq_len = 8 # for smooth
    for dataroot_subset in ["BioMotionLab_NTroje", "CMU", "MPI_HDM05"]:
        print(dataroot_subset)
        for phase in ["train","test"]:
            print(phase)
            savedir = os.path.join(args.save_dir, dataroot_subset, phase)
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            split_file = os.path.join(
                "prepare_data/data_split", dataroot_subset, phase + "_split.txt"
            )

            with open(split_file, "r") as f:
                filepaths = [line.strip() for line in f]

            idx = 0
            for filepath in tqdm(filepaths):
                data = {}
                bdata = np.load(
                    os.path.join(args.root_dir, filepath), allow_pickle=True
                )

                if "mocap_framerate" in bdata:
                    framerate = bdata["mocap_framerate"]
                else:
                    continue
                idx += 1

                if framerate == 120:
                    stride = 2
                elif framerate == 60:
                    stride = 1
                else:
                    raise AssertionError(
                        "Please check your AMASS data, should only have 2 types of framerate, either 120 or 60!!!"
                    )
                bdata_poses = bdata["poses"][::stride, ...]
                bdata_trans = bdata["trans"][::stride, ...]
                subject_gender = bdata["gender"]

                if bdata_poses.shape[0] <= min_seq_len:
                    continue

                body_parms = {
                    "root_orient": torch.Tensor(
                        bdata_poses[:, :3]
                    ).to(device),  # .to(comp_device), # controls the global root orientation
                    "pose_body": torch.Tensor(
                        bdata_poses[:, 3:66]
                    ).to(device),  # .to(comp_device), # controls the body
                    "trans": torch.Tensor(
                        bdata_trans
                    ).to(device),  # .to(comp_device), # controls the global body position
                }


                body_pose_world = bm(
                    **{
                        k: v
                        for k, v in body_parms.items()
                        if k in ["pose_body", "root_orient", "trans"]
                    }
                )

                body_pose_local = bm(
                    **{
                        "pose_body":body_parms["pose_body"],
                        "root_orient": body_parms["root_orient"],
                        "trans":None
                    }
                )
                position_local_full_gt_world = body_pose_local.Jtr[:, :22, :]

                output_aa = torch.Tensor(bdata_poses[:, :66]).to(device).reshape(-1, 3)
                output_6d = utils_transform.aa2sixd(output_aa).reshape(
                    bdata_poses.shape[0], -1
                )
                rotation_local_full_gt_list = output_6d[1:]

                rotation_local_matrot = aa2matrot(
                    torch.tensor(bdata_poses).to(device).reshape(-1, 3)
                ).reshape(bdata_poses.shape[0], -1, 9)
                rotation_global_matrot = local2global_pose(
                    rotation_local_matrot, bm.kintree_table[0].long()
                )  # rotation of joints relative to the origin

                head_rotation_global_matrot = rotation_global_matrot[:, [15], :, :]

                rotation_global_6d = utils_transform.matrot2sixd(
                    rotation_global_matrot.reshape(-1, 3, 3)
                ).reshape(rotation_global_matrot.shape[0], -1, 6)


                position_global_full_gt_world = body_pose_world.Jtr[
                    :, :22, :
                ]  # position of joints relative to the world origin

                position_head_world = position_global_full_gt_world[
                    :, 15, :
                ]  # world position of head

                head_global_trans = torch.eye(4).repeat(
                    position_head_world.shape[0], 1, 1
                )
                head_global_trans[:, :3, :3] = head_rotation_global_matrot.squeeze()
                head_global_trans[:, :3, 3] = position_global_full_gt_world[:, 15, :]

                head_global_trans_list = head_global_trans[1:]

                sparse_pos_global = position_global_full_gt_world[1:, [15, 20, 21], :] # N-1, 3, 3
                acc = _syn_acc(sparse_pos_global)   # N-1, 3, 3
                rot_sparse = rotation_global_6d[1:, [15, 20, 21], ...]  # N-1, 3, 6
                sparse_input = torch.cat([acc,rot_sparse],dim=2).reshape(-1, 27)    # N-1, 27
                data["rot"] = rotation_local_full_gt_list
                data["sparse"] = sparse_input
                data["trans"] = body_parms["trans"][1:,...]   # N-1, 3
                data["lposition"] = position_local_full_gt_world
                data["gposition"] = position_global_full_gt_world
                data["head_global_trans_list"] = head_global_trans_list
                data["framerate"] = 60
                data["gender"] = subject_gender
                data["filepath"] = filepath

                torch.save(data, os.path.join(savedir, "{}.pt".format(idx)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default=None,
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default=None, help="=dir where you put your AMASS data"
    )
    args = parser.parse_args()

    # Here we follow the AvatarPoser paper and use male model for all sequences
    bm_fname_male = os.path.join(args.support_dir, "smplh/{}/model.npz".format("male"))
    dmpl_fname_male = os.path.join(
        args.support_dir, "dmpls/{}/model.npz".format("male")
    )

    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male,
    ).to(device)

    bm_male = bm_male.eval()
    main(args, bm_male)
