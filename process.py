import os
import sys

sys.path.append(os.path.abspath('./'))

import glob

import articulate as art
import numpy as np
import torch
import utils.config as cfg
from tqdm import tqdm
from utils.data import fill_dip_nan, normalize_imu

from utils.skeleton import *

raw_dir = cfg.raw_dir
extract_dir = cfg.extract_dir
work_dir = cfg.work_dir
split_dir = cfg.split_dir
smpl_m = cfg.smpl_m
dip_dir = os.path.join(cfg.raw_dir, 'dip')
out_dir = os.path.join(cfg.raw_dir, 'dip_trans')
xsens_model = XsensSkeleton()

def process_xsens():
    r"""
    imu_mask: [Head, LeftElbow, RightElbow]
    """     

    imu_mask = torch.tensor([2, 5, 9]) # head right left
    imu_num = 3

    print('\n')
    infos = os.listdir(os.path.join(split_dir, 'xsens'))
    for info_name in infos:
        dataset, phase = info_name.split('.')[0].split('_')
        with open(os.path.join(split_dir, 'xsens', info_name), 'r') as file:
            l = file.read().splitlines()
        print('processing {}_{}...'.format(dataset, phase))
        for motion_name in tqdm(l):
            temp_data = torch.load(os.path.join(extract_dir, dataset, motion_name))
       
            out_data = {'joint': {'orientation': [], 'velocity': []},
                        'imu': {'imu': []},
                        }

            # normalize and to r6d
            glb_pose = art.math.quaternion_to_rotation_matrix(temp_data['joint']['orientation'].contiguous()).view(-1, 23, 3, 3)
            out_data['joint']['orientation'] = glb_pose # glb gt
                
            acc = temp_data['imu']['free acceleration'][:, imu_mask].view(-1, imu_num, 3)
            ori = art.math.quaternion_to_rotation_matrix(temp_data['imu']['calibrated orientation'][:, imu_mask].contiguous()).view(-1, imu_num, 9)
            out_data['imu']['imu'] = torch.cat([ori,acc],dim=-1)#normalize_imu(acc, ori)
            
            # calculate velocity and normalize w.r.t. root orientation
            gt_position = temp_data['joint']['position'] # N 23 3

            trans = gt_position[:,:1,:] - gt_position[:1,:1,:]
            out_data['joint']['trans'] = trans

            gt_position = xsens_model.forward_kinematics(glb_pose)+trans
            lposition = gt_position-gt_position[:,:1,:]
            out_data['joint']['lposition'] = lposition
            velocity = (gt_position[1:] - gt_position[:-1]) * 60.0
            velocity = torch.cat((velocity[:1], velocity), dim=0)

            out_data['joint']['velocity'] = velocity
            
            out_dir = os.path.join(work_dir, phase, dataset, motion_name)
            os.makedirs(os.path.join(work_dir, phase, dataset), exist_ok=True)
            torch.save(out_data, out_dir)         

def generate_dip_trans():
    import pickle
    split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08', 's_09', 's_10']
    for subject_name in tqdm(split):
        for motion_name in os.listdir(os.path.join(dip_dir, subject_name)):
            out_data = {
                'imu_acc': [],
                'imu_ori': [],
                'pose': [],
                'trans': []
                    }
            
            path = os.path.join(dip_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc']).float()
            ori = torch.from_numpy(data['imu_ori']).float()        
            pose_aa = torch.from_numpy(data['gt']).float()
            
            pose = art.math.axis_angle_to_rotation_matrix(pose_aa).view(-1, 24, 3, 3)
            body_model = art.ParametricModel(smpl_m, device='cpu')

            lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
            lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]
                
            j, _ = body_model.get_zero_pose_joint_and_vertex()
            b = art.math.joint_position_to_bone_vector(j[lower_body].unsqueeze(0),
                                                        lower_body_parent).squeeze(0)
            bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
            b = bone_orientation * bone_length
            b[:3] = 0
            floor_y = j[10:12, 1].min().item()

            j = body_model.forward_kinematics(pose=pose, calc_mesh=False)[1]

            trans = torch.zeros(j.shape[0], 3)

            # force lowest foot to the floor
            for i in range(j.shape[0]):
                current_foot_y = j[i, [10, 11], 1].min().item()
                if current_foot_y > floor_y:
                    trans[i, 1] = floor_y - current_foot_y
            
            out_data['imu_acc'] = acc
            out_data['imu_ori'] = ori
            out_data['pose'] = pose_aa
            out_data['trans'] = trans.float()
            
            os.makedirs(os.path.join(raw_dir, 'dip_trans'), exist_ok=True)
            torch.save(out_data, os.path.join(raw_dir, 'dip_trans', subject_name + '_' + motion_name.replace('.pkl', '.pt')))    


def process_dipimu():
    r"""
    imu_mask: [Head, LeftElbow, RightElbow]
    """
    
    print('processing dip-imu...')
    imu_mask = torch.tensor([0, 7, 8])
    imu_num = 3
    infos = os.listdir(os.path.join(split_dir, 'dip'))
    for info_name in infos:
        _, phase = info_name.split('.')[0].split('_')
        with open(os.path.join(split_dir, 'dip', info_name), 'r') as file:
            l = file.read().splitlines()
        for motion_name in tqdm(l):
            path = os.path.join(raw_dir, 'dip_trans', motion_name)
            data = torch.load(path)
            
            acc = data['imu_acc'][:, imu_mask]
            ori = data['imu_ori'][:, imu_mask]
            pose = data['pose']
            trans = data['trans']

            out_data = {'joint': {'orientation': [], 'velocity': [], 'position': []},
                        'imu': {'imu': []},
                        }                       
            
            # fill nan with nearest neighbors
            if True in torch.isnan(acc):
                acc = fill_dip_nan(acc)
            if True in torch.isnan(ori):
                ori = fill_dip_nan(ori.view(-1, imu_num, 9))
                
            body_model = art.ParametricModel(smpl_m, device='cpu')  
            p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)            
            glb_pose, gt_position  = body_model.forward_kinematics(pose=p, tran=trans, calc_mesh=False)        

            glb_pose_norm = glb_pose
            
            acc, ori, glb_pose_norm, gt_position, trans = acc[6:-6], ori[6:-6], glb_pose_norm[6:-6], gt_position[6:-6], trans[6:-6]
            p = p[6:-6]

            out_data['joint']['lposition'] = gt_position-gt_position[:,:1,:]
            out_data['joint']['trans'] = trans
          
            velocity = (gt_position[1:] - gt_position[:-1]) * 60
            velocity = torch.cat((velocity[:1], velocity), dim=0)
            
            out_data['joint']['velocity'] = velocity # N, 24, 3, 3    
            out_data['joint']['orientation'] = glb_pose_norm # N 90
            out_data['imu']['imu'] = torch.cat([ori,acc],dim=-1)

            out_dir = os.path.join(work_dir, phase, 'dip', motion_name)
            os.makedirs(os.path.join(work_dir, phase, 'dip'), exist_ok=True)
            torch.save(out_data, out_dir)     

if __name__ == '__main__':
    process_xsens()
    print('\n')
    print('generating dip-imu pseudo vertical translation...')
    generate_dip_trans()
    process_dipimu()