import torch
import articulate as art
from articulate.math.angular import radian_to_degree, angle_between
from utils.skeleton import XsensSkeleton
from utils import utils_transform
import numpy as np
import math

class Evaluator:
    def __init__(self, smpl_path, sip_xsens=torch.tensor([8, 12, 15, 19]), sip_smpl=torch.tensor([1, 2, 16, 17]),
                  device='cuda'):
        self.names = ['SIP Error (deg)', 'Global Angle Error (deg)']
        self.body_model = art.ParametricModel(smpl_path, device=device)
        self.xsens_sk = XsensSkeleton()
        self.sip_xsens = sip_xsens
        self.sip_smpl = sip_smpl
        self.lower_smpl = torch.tensor([1, 2, 4, 5])
        self.lower_xsens = torch.tensor([15, 16, 19, 20])

    def eval_xsens(self, pose_p, pose_t):
        r"""
        Args:
            pose_p :xsens prediction rotation matrix that can reshape to [num_frame, 23, 3, 3].
            pose_t :xsens gt rotation matrix that can reshape to [num_frame, 23, 3, 3].  
        Returns:    
            gae: Global Angular Error (deg)
            mgae: SIP Error (deg)
            je: Joint Position Error (cm)
        """
        pose_p = pose_p.clone().view(-1, 23, 3, 3)
        pose_t = pose_t.clone().view(-1, 23, 3, 3)
        #ignored_joint_mask = torch.tensor([0, 10, 14, 17, 18, 21, 22]) # dynaip
        ignored_joint_mask = torch.tensor([10, 14, 18, 22])
        # ignored_joint_mask = torch.tensor([0, 2, 10, 14, 17, 18, 21, 22]) # For PIP and TIP, do not calc L3 error
        
        # replace ignored joint with ground truth global rotation
        pose_p[:, ignored_joint_mask] = pose_t[:, ignored_joint_mask]
        gae = radian_to_degree(angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))
        mgae = radian_to_degree(angle_between(pose_p[:, self.sip_xsens], \
                pose_t[:, self.sip_xsens]).view(pose_p.shape[0], -1))
        lgae = radian_to_degree(angle_between(pose_p[:, self.lower_xsens], \
                                              pose_t[:, self.lower_xsens]).view(pose_p.shape[0], -1))

        # since we did not extract each skeleton offsets, we use forward kinematics
        # and mean skeleton to calculate joint position error
        joint_p = self.xsens_sk.forward_kinematics(pose_p)
        joint_t = self.xsens_sk.forward_kinematics(pose_t)
        je = (joint_p - joint_t).norm(dim=2)
        return torch.stack([mgae.mean(), gae.mean(), je.mean() * 100.0, lgae.mean()])

    def eval_smpl(self, pose_p, pose_t):
        r"""
        Args:
            pose_p :smpl prediction rotation matrix that can reshape to [num_frame, 24, 3, 3].
            pose_t :smpl gt rotation matrix that can reshape to [num_frame, 24, 3, 3].
            we get smpl prediction by assigning corresponding xsens joints to smpl joints.
        Returns:
            gae: Global Angular Error (deg)
            mgae: SIP Error (deg)
            je: Joint Position Error (cm)
            ve: Vertex Position Error (cm)
        """
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        #ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23]) # dynaip
        ignored_joint_mask = torch.tensor([10, 11, 20, 21, 22, 23])
        pose_p[:, ignored_joint_mask] = pose_t[:, ignored_joint_mask]
        """pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)"""
        gae = radian_to_degree(
            angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))

        mgae = radian_to_degree(
            angle_between(pose_p[:, self.sip_smpl], \
                          pose_t[:, self.sip_smpl]).view(pose_p.shape[0], -1))

        lgae = radian_to_degree(
            angle_between(pose_p[:, self.lower_smpl], \
                          pose_t[:, self.lower_smpl]).view(pose_p.shape[0], -1))

        pose_p_local = self.body_model.inverse_kinematics_R(pose_p).view(pose_p.shape[0], 24, 3, 3)
        pose_t_local = self.body_model.inverse_kinematics_R(pose_t).view(pose_t.shape[0], 24, 3, 3)

        _, joint_p, vertex_p = self.body_model.forward_kinematics(pose=pose_p_local, calc_mesh=True)
        _, joint_t, vertex_t = self.body_model.forward_kinematics(pose=pose_t_local, calc_mesh=True)

        je = (joint_p[:, :22, :] - joint_t[:, :22, :]).norm(dim=2)
        ve = (vertex_p - vertex_t).norm(dim=2)
        return torch.stack([mgae.mean(), gae.mean(), je.mean() * 100.0, lgae.mean()])#ve.mean() * 100.0])

    def eval_smpl_(self, pose_p, pose_t):
        r"""
        Args:
            pose_p :smpl prediction rotation matrix that can reshape to [num_frame, 24, 3, 3].
            pose_t :smpl gt rotation matrix that can reshape to [num_frame, 24, 3, 3].
            we get smpl prediction by assigning corresponding xsens joints to smpl joints.   
        Returns:    
            gae: Global Angular Error (deg)
            mgae: SIP Error (deg)
            je: Joint Position Error (cm)
            ve: Vertex Position Error (cm)
        """
        #pose_p = pose_p.clone().view(-1, 24, 3, 3)
        #pose_t = pose_t.clone().view(-1, 24, 3, 3)
        #ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
        #pose_p[:, ignored_joint_mask] = pose_t[:, ignored_joint_mask]
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        gae_global = radian_to_degree(
            angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))
        
        mgae_global = radian_to_degree(
            angle_between(pose_p[:, self.sip_smpl], \
                pose_t[:, self.sip_smpl]).view(pose_p.shape[0], -1))
        
        pose_p_local = self.body_model.inverse_kinematics_R(pose_p).view(pose_p.shape[0], 24, 3, 3)
        pose_t_local = self.body_model.inverse_kinematics_R(pose_t).view(pose_t.shape[0], 24, 3, 3)

        pose_p_local_ = pose_p_local[:, 1:22, ...].contiguous()
        pose_t_local_ = pose_t_local[:, 1:22, ...].contiguous()
        pose_p_local_ = (
            utils_transform.matrot2aa(pose_p_local_.reshape(-1, 3, 3).detach()).reshape(pose_p.shape[0],-1,3).float()
        )
        pose_t_local_ = (
            utils_transform.matrot2aa(pose_t_local_.reshape(-1, 3, 3).detach()).reshape(pose_p.shape[0],-1,3).float()
        )
        diff = pose_p_local_ - pose_t_local_
        diff[diff > np.pi] = diff[diff > np.pi] - 2 * np.pi
        diff[diff < -np.pi] = diff[diff < -np.pi] + 2 * np.pi
        gae = torch.absolute(diff) * 180.0 / math.pi
        """gae = radian_to_degree(
            angle_between(pose_p_local_, pose_t_local_).view(pose_p.shape[0], -1))"""

        mgae = radian_to_degree(
            angle_between(pose_p_local[:, self.sip_smpl], \
                          pose_t_local[:, self.sip_smpl]).view(pose_p.shape[0], -1))

        _, joint_p, vertex_p = self.body_model.forward_kinematics(pose=pose_p_local, calc_mesh=True)
        _, joint_t, vertex_t = self.body_model.forward_kinematics(pose=pose_t_local, calc_mesh=True)
        
        je = (joint_p[:,:22,:] - joint_t[:,:22,:]).norm(dim=2)
        ve = (vertex_p - vertex_t).norm(dim=2)
        return torch.stack([mgae.mean(), gae.mean(), je.mean() * 100.0, ve.mean() * 100.0])