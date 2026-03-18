# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
from human_body_prior.tools import tgm_conversion as tgm
from human_body_prior.tools.rotation_tools import aa2matrot, matrot2aa
from torch.nn import functional as F
import numpy as np
import math

def _glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens):
    glb_full_pose_smpl = torch.eye(3, device=glb_full_pose_xsens.device).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
    indices = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18, 5, 11, 7, 6, 12, 8, 13, 9, 13, 9, 13, 9]
    for idx, i in enumerate(indices):
        glb_full_pose_smpl[:, idx, :] = glb_full_pose_xsens[:, i, :]
    return glb_full_pose_smpl

def _reduced_glb_6d_to_full_glb_mat_xsens(pose_body, root_orient, orientation):  # orientation(3 sensors) b,n,3,6
    # (-1, 23, 3, 3)
    seq_len = pose_body.shape[0]
    pose_body = sixd2matrot(pose_body.reshape(-1, 6)).reshape(-1, 13, 3, 3)
    root_orient = sixd2matrot(root_orient.reshape(-1, 6)).reshape(-1, 3, 3)
    orientation = sixd2matrot(orientation.reshape(-1, 6)).reshape(-1, 3, 3, 3)
    joint_set = [1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 19, 20]  # 13
    sensor_set = [6, 9, 13]  # 3
    ignored = [10, 14, 17, 18, 21, 22]  # 6
    parent = [9, 13, 16, 16, 20, 20]
    # root_rotation = root_orient
    # glb_reduced_pose = art.math.r6d_to_rotation_matrix(pose_body).view(-1, len(joint_set), 3, 3)
    # back to glb coordinate
    """pose_body = root_orient.unsqueeze(1).matmul(pose_body)
    orientation = root_orient.unsqueeze(1).matmul(orientation)"""
    global_full_pose = torch.eye(3, device=pose_body.device).repeat(pose_body.shape[0], 23, 1, 1)
    global_full_pose[:, 0] = root_orient
    global_full_pose[:, joint_set] = pose_body
    global_full_pose[:, sensor_set] = orientation
    global_full_pose[:, ignored] = global_full_pose[:, parent]
    return global_full_pose # (-1, 23, 3, 3)

def bgs(d6s):
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    c = torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1
    b2 = F.normalize(a2 - c, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrot2sixd(pose_matrot):
    """
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    """
    pose_6d = torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)
    return pose_6d


def aa2sixd(pose_aa):
    """
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    """
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d


def sixd2matrot(pose_6d):
    """
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    """
    rot_vec_1 = pose_6d[:, :3]
    rot_vec_2 = pose_6d[:, 3:6]
    rot_vec_3 = torch.cross(rot_vec_1, rot_vec_2)
    pose_matrot = torch.stack([rot_vec_1, rot_vec_2, rot_vec_3], dim=-1)
    return pose_matrot


def sixd2aa(pose_6d, batch=False):
    """
    :param pose_6d: Nx6
    :return: pose_aa: Nx3
    """
    if batch:
        B, J, C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1, 6)
    pose_matrot = sixd2matrot(pose_6d)
    pose_aa = matrot2aa(pose_matrot)
    if batch:
        pose_aa = pose_aa.reshape(B, J, 3)
    return pose_aa


def sixd2quat(pose_6d):
    """
    :param pose_6d: Nx6
    :return: pose_quaternion: Nx4
    """
    pose_mat = sixd2matrot(pose_6d)
    pose_mat_34 = torch.cat(
        (pose_mat, torch.zeros(pose_mat.size(0), pose_mat.size(1), 1)), dim=-1
    )
    pose_quaternion = tgm.rotation_matrix_to_quaternion(pose_mat_34)
    return pose_quaternion


def quat2aa(pose_quat):
    """
    :param pose_quat: Nx4
    :return: pose_aa: Nx3
    """
    return tgm.quaternion_to_angle_axis(pose_quat)


# euler batch*3
# output cuda batch*3*3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
def euler2matrot(euler):
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_x, R_y))

    return R

def eulerAnglesToRotationMatrix_xzy(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_y, np.dot(R_z, R_x))

    return R