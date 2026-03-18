import glob
import os
import random

import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from utils import utils_transform

from utils.skeleton import *

class XsensDataset(Dataset):
    def __init__(self, dataset_dir, datasets=['unipd', 'cip', 'andy', 'emokine', 'virginia']
                 , seq_len=196, device='cuda:0'):
        super(XsensDataset, self).__init__()
        self.train_dir = dataset_dir  # os.path.join(cfg.work_dir, 'train')
        self.datasets = datasets
        self.seq_len = seq_len
        self.data = {'imu': [], 'pose': [], 'velocity': [], 'lposition': [], 'trans': []}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prepare_data()
        print("{} samples in total.".format(len(self.data['imu'])))

    def __len__(self):
        return len(self.data['imu'])

    def prepare_data(self):
        r"""
        joint_mask: ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'LeftLowerLeg', 'RightLowerLeg', 'L3',
         'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftUpperArm',
         'RightUpperArm', 'LeftForeArm', 'RightForeArm']
        """
        #joint_trans = [0,19,15,1,20,16,3,21,17,4,22,18,5,11,7,6,12,8,13,9,14,10]

        for dataset in self.datasets:
            temp_data = {'imu': [], 'velocity': [], 'pose': [], 'lposition': [], 'trans': []}
            for motion_name in tqdm(glob.glob(os.path.join(self.train_dir, "train", dataset,  '*.pt'))):
                data = torch.load(motion_name, weights_only=False, map_location="cpu")
                imu = data['imu']['imu'].view(-1, 3, 12).float()
                imu_ori = imu[...,:9]
                imu_acc = imu[...,9:]
                seq_len = imu.shape[0]
                imu_ori = utils_transform.matrot2sixd(imu_ori.reshape(-1,3,3)).reshape(seq_len,-1,6)
                #imu = torch.cat([imu_ori,imu_acc],dim=-1)   # s,3,6+3
                imu = torch.cat([imu_acc, imu_ori], dim=-1)  # s,3,3+6
                if dataset == "dip":
                    imu = imu[:,[0,2,1],:]
                tmp = torch.split(imu.view(seq_len, 27), self.seq_len)
                if tmp[-1].shape[0] != self.seq_len:
                    temp_data['imu'].extend(tmp[:-1])
                else:
                    temp_data['imu'].extend(tmp)

                if dataset == "dip":
                    vel_mask = torch.tensor([0, 15, 21, 20]) # pelvis head rightwrist leftwrist
                else:
                    vel_mask = torch.tensor([0, 6, 10, 14])  # body_vel + sparse_vel
                velocity = data['joint']['velocity'][:, vel_mask].float()
                # velocity = data['joint']['velocity'].float()
                tmp = torch.split(velocity.view(velocity.shape[0], -1, 3), self.seq_len)
                if tmp[-1].shape[0] != self.seq_len:
                    temp_data['velocity'].extend(tmp[:-1])
                else:
                    temp_data['velocity'].extend(tmp)
                # Discard pelvis, head, wrists and ankles, which have direct imu-readings
                # pose missing, infer

                # hips chest chest2 chest3 chest4 neck rightcollar rightshoulder leftcollar leftshoulder
                # righthip rightknee lefthip leftknee
                if dataset == "dip":
                    joint_mask = [0, 3, 3, 6, 9, 12, 14, 17, 13, 16, 2, 5, 1, 4]
                else:
                    joint_mask = [0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 19, 20]
                pose = data['joint']['orientation'].float()
                pose = pose[:, joint_mask]
                pose = utils_transform.matrot2sixd(pose.reshape(-1,3,3)).reshape(pose.shape[0],-1)
                tmp = torch.split(pose, self.seq_len)
                if tmp[-1].shape[0] != self.seq_len:
                    temp_data['pose'].extend(tmp[:-1])
                else:
                    temp_data['pose'].extend(tmp)

                if dataset == "dip":
                    tmp = torch.split(data['joint']['trans'][:, :].float(), self.seq_len)
                else:
                    tmp = torch.split(data['joint']['trans'][:, 0, :].float(), self.seq_len)
                if tmp[-1].shape[0] != self.seq_len:
                    temp_data['trans'].extend(tmp[:-1])
                else:
                    temp_data['trans'].extend(tmp)

                if dataset == "dip":    # head rightwrist leftwrist
                    tmp = torch.split(data['joint']['lposition'][:, [15, 21, 20], :].reshape(-1, 9), self.seq_len)
                else:
                    tmp = torch.split(data['joint']['lposition'][:, [6, 10, 14], :].reshape(-1, 9), self.seq_len)
                if tmp[-1].shape[0] != self.seq_len:
                    temp_data['lposition'].extend(tmp[:-1])
                else:
                    temp_data['lposition'].extend(tmp)

            self.data['imu'].extend(temp_data['imu'])
            self.data['velocity'].extend(temp_data['velocity'])
            self.data['pose'].extend(temp_data['pose'])
            self.data['trans'].extend(temp_data['trans'])
            self.data['lposition'].extend(temp_data['lposition'])

    # rot, sparse, trans, gposition, lposition, vel
    def __getitem__(self, index):
        imu = self.data['imu'][index]
        vel = self.data['velocity'][index]
        trans = self.data['trans'][index]
        pose = self.data['pose'][index]
        lposition = self.data['lposition'][index]
        """imu = self.data['imu'][index].to(self.device)
        vel = self.data['velocity'][index].to(self.device)
        trans = self.data['trans'][index].to(self.device)
        pose = self.data['pose'][index].to(self.device)
        lposition = self.data['lposition'][index].to(self.device)"""
        return pose, imu, trans, lposition, vel

class XsensTestDataset(Dataset):
    def __init__(self, dataset_dir, datasets=['dip', 'unipd', 'cip', 'andy', 'emokine', 'virginia'], seq_len=196, device='cuda:0',):
        super(XsensTestDataset, self).__init__()
        self.train_dir = dataset_dir  # os.path.join(cfg.work_dir, 'train')
        self.datasets = datasets
        self.seq_len = seq_len
        self.data = {'imu': [], 'pose': [], 'velocity': [], 'lposition': [], 'trans': [], 'full smpl glb':[]}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.prepare_data()
        print("{} samples in total.".format(len(self.data['imu'])))

    def __len__(self):
        return len(self.data['imu'])

    def prepare_data(self):
        r"""
        joint_mask: ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'LeftLowerLeg', 'RightLowerLeg', 'L3',
         'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftUpperArm',
         'RightUpperArm', 'LeftForeArm', 'RightForeArm']
        """
        #joint_trans = [0,19,15,1,20,16,3,21,17,4,22,18,5,11,7,6,12,8,13,9,14,10]

        for dataset in self.datasets:
            temp_data = {'imu': [], 'velocity': [], 'pose': [], 'lposition': [], 'trans': [], 'full smpl glb':[]}
            #for motion_name in tqdm(glob.glob(os.path.join(self.train_dir, "test", dataset,  '*.pt'))):
            for motion_name in glob.glob(os.path.join(self.train_dir, "test", dataset, '*.pt')):
                data = torch.load(motion_name, weights_only=False, map_location="cpu")
                imu = data['imu']['imu'].view(-1, 3, 12).float()
                imu_ori = imu[...,:9]
                imu_acc = imu[...,9:]
                seq_len = imu.shape[0]
                imu_ori = utils_transform.matrot2sixd(imu_ori.reshape(-1,3,3)).reshape(seq_len,-1,6)
                imu = torch.cat([imu_acc,imu_ori],dim=-1)   # s,3,6+3
                if dataset == "dip":
                    imu = imu[:,[0,2,1],:]
                temp_data['imu'].append(imu.view(seq_len, 27))

                if dataset == "dip":
                    vel_mask = torch.tensor([0, 15, 21, 20]) # pelvis head rightwrist leftwrist
                else:
                    vel_mask = torch.tensor([0, 6, 10, 14])  # body_vel + sparse_vel
                velocity = data['joint']['velocity'][:, vel_mask].float()
                # velocity = data['joint']['velocity'].float()
                velocity = velocity.view(velocity.shape[0], -1, 3)
                temp_data['velocity'].append(velocity)
                # Discard pelvis, head, wrists and ankles, which have direct imu-readings
                # pose missing, infer

                # hips chest chest2 chest3 chest4 neck rightcollar rightshoulder leftcollar leftshoulder
                # righthip rightknee lefthip leftknee
                pose = data['joint']['orientation'].float()
                if dataset == "dip":
                    joint_mask = [0, 3, 3, 6, 9, 12, 14, 17, 13, 16, 2, 5, 1, 4]
                    temp_data['full smpl glb'].append(pose)
                else:
                    joint_mask = [0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 19, 20]
                    temp_data['full smpl glb'].append(torch.tensor([0]))
                pose = pose[:, joint_mask]
                pose = utils_transform.matrot2sixd(pose.reshape(-1,3,3)).reshape(pose.shape[0],-1)
                temp_data['pose'].append(pose)

                if dataset == "dip":
                    tmp = data['joint']['trans'][:, :].float()
                else:
                    tmp = data['joint']['trans'][:, 0, :].float()
                temp_data['trans'].append(tmp)

                if dataset == "dip":    # head rightwrist leftwrist
                    tmp = data['joint']['lposition'][:, [15, 21, 20], :].reshape(-1, 9)
                else:
                    tmp = data['joint']['lposition'][:, [6, 10, 14], :].reshape(-1, 9)
                temp_data['lposition'].append(tmp)

            self.data['imu'].extend(temp_data['imu'])
            self.data['velocity'].extend(temp_data['velocity'])
            self.data['pose'].extend(temp_data['pose'])
            self.data['trans'].extend(temp_data['trans'])
            self.data['lposition'].extend(temp_data['lposition'])
            self.data['full smpl glb'].extend(temp_data['full smpl glb'])

    # rot, sparse, trans, gposition, lposition, vel
    def __getitem__(self, index):
        imu = self.data['imu'][index]
        vel = self.data['velocity'][index]
        trans = self.data['trans'][index]
        pose = self.data['pose'][index]
        lposition = self.data['lposition'][index]
        smpl_glb = self.data['full smpl glb'][index]
        return pose.unsqueeze(0), imu.unsqueeze(0), trans.unsqueeze(0), lposition.unsqueeze(0), vel.unsqueeze(0), smpl_glb.unsqueeze(0)

class TrainDatasetTrans(Dataset):
    def __init__(
        self,
        rot,sparse,trans,gposition,lposition,
        input_motion_length=196,
        train_dataset_repeat_times=1
    ):
        self.rot = rot
        self.sparse = sparse
        self.trans = trans
        self.gposition = gposition
        self.lposition = lposition
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.input_motion_length = input_motion_length

    def __len__(self):
        return len(self.trans) * self.train_dataset_repeat_times

    def __getitem__(self, idx):

        rot = self.rot[idx % len(self.trans)]
        sparse = self.sparse[idx % len(self.trans)]
        trans = self.trans[idx % len(self.trans)]
        gposition = self.gposition[idx % len(self.trans)]
        lposition = self.lposition[idx % len(self.trans)]
        seqlen = trans.shape[0]

        if seqlen <= self.input_motion_length:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]

        sparse = sparse[idx : idx + self.input_motion_length]
        rot = rot[idx: idx + self.input_motion_length]
        trans = trans[idx: idx + self.input_motion_length]
        gposition = gposition[idx: idx + self.input_motion_length]
        lposition = lposition[idx: idx + self.input_motion_length]

        return rot.unsqueeze(0).float(),sparse.unsqueeze(0).float(),trans.unsqueeze(0).float(),\
            gposition.unsqueeze(0).float(),lposition.unsqueeze(0).float()

class TestDatasetTrans(Dataset):
    def __init__(
        self,
        all_info,
        filename_list
    ):
        self.filename_list = filename_list

        self.rot = []
        self.sparse = []
        self.trans = []
        self.gposition = []
        self.lposition = []
        self.head_motion = []
        for i in all_info:
            self.rot.append(i["rot"])
            self.sparse.append(i["sparse"])
            self.trans.append(i["trans"])
            self.gposition.append(i["gposition"])
            self.lposition.append(i["lposition"])
            self.head_motion.append(i["head_global_trans_list"])

    def __len__(self):
        return len(self.rot)

    def __getitem__(self, idx):
        rot = self.rot[idx]
        sparse = self.sparse[idx]
        trans = self.trans[idx]
        gposition = self.gposition[idx]
        lposition = self.lposition[idx]
        head_motion = self.head_motion[idx]

        return (
            rot.unsqueeze(0),
            sparse.unsqueeze(0).float(),
            trans.unsqueeze(0),
            gposition.unsqueeze(0),
            lposition.unsqueeze(0),
            head_motion
        )

class AMASS_hmd(Dataset):
    def __init__(
            self,
            rot, sparse, trans, gposition, lposition,
            input_motion_length=196,
            train_dataset_repeat_times=1
    ):
        """self.rot = rot
        self.sparse = sparse
        self.trans = trans
        self.gposition = gposition
        self.lposition = lposition"""
        self.train_dataset_repeat_times = train_dataset_repeat_times
        #self.input_motion_length = input_motion_length
        self.new_sparses = []
        self.new_rots = []
        self.new_trans = []
        self.new_gpositions = []
        self.new_lpositions = []
        idx = -1
        for acc in tqdm(sparse):
            idx += 1
            seq_len = acc.shape[0]
            if seq_len < input_motion_length:  # Arbitrary choice
                continue
            seq_idx = random.randint(0,seq_len % input_motion_length)
            while seq_idx + input_motion_length < seq_len:
                self.new_sparses.append(sparse[idx][seq_idx:seq_idx + input_motion_length].cpu())
                self.new_rots.append(rot[idx][seq_idx:seq_idx + input_motion_length].cpu())
                self.new_trans.append(trans[idx][seq_idx:seq_idx + input_motion_length].cpu())
                self.new_gpositions.append(gposition[idx][seq_idx:seq_idx + input_motion_length].cpu())
                self.new_lpositions.append(lposition[idx][seq_idx:seq_idx + input_motion_length].cpu())
                seq_idx += input_motion_length

    def __len__(self):
        return len(self.new_sparses) * self.train_dataset_repeat_times

    def __getitem__(self, idx):

        sparse = self.new_sparses[idx % len(self.new_sparses)]
        rot = self.new_rots[idx % len(self.new_sparses)].reshape(-1,132)
        trans = self.new_trans[idx % len(self.new_sparses)]
        gposition = self.new_gpositions[idx % len(self.new_sparses)]
        lposition = self.new_lpositions[idx % len(self.new_sparses)]

        #return rot.unsqueeze(0).float(),sparse.unsqueeze(0).float(),\
        #    trans.unsqueeze(0).float(),gposition.unsqueeze(0).float(),lposition.unsqueeze(0).float()
        return rot.float(),sparse.float(),trans.float(),gposition.float(),lposition.float()

class AMASS(Dataset):
    def __init__(
        self,
        path,
        input_motion_length=196,
        #psedo_random_seed=7,
        train_dataset_repeat_times=1
    ):
        self.train_dataset_repeat_times = train_dataset_repeat_times
        print("loading training dataset:")
        #data = torch.load(path)
        sparse = torch.load(path+'sparse.pt',weights_only=False)
        rot = torch.load(path+'pose.pt',weights_only=False)
        trans = torch.load(path+'trans.pt',weights_only=False)
        gposition = torch.load(path+'gposition.pt',weights_only=False)
        lposition = torch.load(path+'lposition.pt',weights_only=False)
        self.new_sparses = []
        self.new_rots = []
        self.new_trans = []
        self.new_gpositions = []
        self.new_lpositions = []
        idx = -1
        for acc in tqdm(sparse):
            idx += 1
            seq_len = acc.shape[0]
            if seq_len < input_motion_length:  # Arbitrary choice
                continue
            seq_idx = random.randint(0,seq_len % input_motion_length)
            while seq_idx + input_motion_length < seq_len:
                self.new_sparses.append(sparse[idx][seq_idx:seq_idx + input_motion_length])
                self.new_rots.append(rot[idx][seq_idx:seq_idx + input_motion_length])
                self.new_trans.append(trans[idx][seq_idx:seq_idx + input_motion_length])
                self.new_gpositions.append(gposition[idx][seq_idx:seq_idx + input_motion_length])
                self.new_lpositions.append(lposition[idx][seq_idx:seq_idx + input_motion_length])
                seq_idx += input_motion_length

    def __len__(self):
        return len(self.new_sparses) * self.train_dataset_repeat_times

    def __getitem__(self, idx):

        sparse = self.new_sparses[idx % len(self.new_sparses)]
        rot = self.new_rots[idx % len(self.new_sparses)].reshape(-1,132)
        trans = self.new_trans[idx % len(self.new_sparses)]
        gposition = self.new_gpositions[idx % len(self.new_sparses)]
        lposition = self.new_lpositions[idx % len(self.new_sparses)]

        #return rot.unsqueeze(0).float(),sparse.unsqueeze(0).float(),\
        #    trans.unsqueeze(0).float(),gposition.unsqueeze(0).float(),lposition.unsqueeze(0).float()
        return rot.float(),sparse.float(),trans.float(),gposition.float(),lposition.float()

class AMASS_6imu(Dataset):
    def __init__(
        self,
        path,
        input_motion_length=196,
        #psedo_random_seed=7,
        train_dataset_repeat_times=1
    ):
        self.train_dataset_repeat_times = train_dataset_repeat_times
        print("loading training dataset:")
        #data = torch.load(path)
        sparse = torch.load(path+'sparse.pt',weights_only=False)
        rot = torch.load(path+'pose.pt',weights_only=False)
        trans = torch.load(path+'trans.pt',weights_only=False)
        contact = torch.load(path+'contact.pt',weights_only=False)
        lposition = torch.load(path+'lposition.pt',weights_only=False)
        self.new_sparses = []
        self.new_rots = []
        self.new_trans = []
        self.new_contacts = []
        self.new_lpositions = []
        idx = -1
        for acc in tqdm(sparse):
            idx += 1
            seq_len = acc.shape[0]
            if seq_len < input_motion_length:  # Arbitrary choice
                continue
            seq_idx = random.randint(0,seq_len % input_motion_length)
            while seq_idx + input_motion_length < seq_len:
                self.new_sparses.append(sparse[idx][seq_idx:seq_idx + input_motion_length])
                self.new_rots.append(rot[idx][seq_idx:seq_idx + input_motion_length])
                self.new_trans.append(trans[idx][seq_idx:seq_idx + input_motion_length])
                self.new_contacts.append(contact[idx][seq_idx:seq_idx + input_motion_length])
                self.new_lpositions.append(lposition[idx][seq_idx:seq_idx + input_motion_length])
                seq_idx += input_motion_length

    def __len__(self):
        return len(self.new_sparses) * self.train_dataset_repeat_times

    def __getitem__(self, idx):

        sparse = self.new_sparses[idx % len(self.new_sparses)]
        rot = self.new_rots[idx % len(self.new_sparses)].reshape(-1,132)
        trans = self.new_trans[idx % len(self.new_sparses)]
        contact = self.new_contacts[idx % len(self.new_sparses)]
        lposition = self.new_lpositions[idx % len(self.new_sparses)]

        #return rot.unsqueeze(0).float(),sparse.unsqueeze(0).float(),\
        #    trans.unsqueeze(0).float(),gposition.unsqueeze(0).float(),lposition.unsqueeze(0).float()
        return rot.float(),sparse.float(),trans.float(),contact.float(),lposition.float()

class TotalCapture(Dataset):
    def __init__(
        self,
        path
    ):
        print("loading TotalCapture training dataset:")
        data = torch.load(path,weights_only=False)
        self.sparse = data['sparse']
        self.rot = data['pose']
        self.trans = data['trans']
        self.gposition = data['gposition']
        self.lposition = data['lposition']

    def __len__(self):
        return len(self.sparse)

    def __getitem__(self, idx):

        sparse = self.sparse[idx]
        rot = self.rot[idx].reshape(-1,132)
        trans = self.trans[idx]
        gposition = self.gposition[idx]
        lposition = self.lposition[idx]

        #return rot.unsqueeze(0).float(),sparse.unsqueeze(0).float(),\
        #    trans.unsqueeze(0).float(),gposition.unsqueeze(0).float(),lposition.unsqueeze(0).float()
        return rot.float().unsqueeze(0),sparse.float().unsqueeze(0),trans.float().unsqueeze(0),\
            gposition.float().unsqueeze(0),lposition.float().unsqueeze(0)

class DIPIMU(Dataset):
    def __init__(
        self,
        path
    ):
        print("loading DIPIMU training dataset:")
        data = torch.load(path,weights_only=False)
        self.sparse = data['sparse']
        self.rot = data['pose']
        self.trans = data['trans']
        self.gposition = data['gposition']
        self.lposition = data['lposition']

    def __len__(self):
        return len(self.sparse)

    def __getitem__(self, idx):

        sparse = self.sparse[idx]
        rot = self.rot[idx].reshape(-1,132)
        trans = self.trans[idx]
        gposition = self.gposition[idx]
        lposition = self.lposition[idx]

        #return rot.unsqueeze(0).float(),sparse.unsqueeze(0).float(),\
        #    trans.unsqueeze(0).float(),gposition.unsqueeze(0).float(),lposition.unsqueeze(0).float()
        return rot.float().unsqueeze(0),sparse.float().unsqueeze(0),trans.float().unsqueeze(0),\
            gposition.float().unsqueeze(0),lposition.float().unsqueeze(0)

def get_motion_trans(motion_list):
    # rotation_local_full_gt_list : 6d rotation parameters
    # hmd_position_global_full_gt_list : 3 joints(head, hands) 6d rotation/6d rotation velocity/global translation/global translation velocity
    rot = [i["rot"] for i in motion_list]
    sparse = [i["sparse"] for i in motion_list]
    trans = [i["trans"] for i in motion_list]
    gposition = [i["gposition"] for i in motion_list]
    lposition = [i["lposition"] for i in motion_list]
    return rot,sparse,trans,gposition,lposition

def get_path(dataset_path, split):
    data_list_path = []
    parent_data_path = glob.glob(dataset_path + "/*")
    for d in parent_data_path:
        if os.path.isdir(d):
            files = glob.glob(d + "/" + split + "/*pt")
            data_list_path.extend(files)
    return data_list_path

def load_data_trans(dataset_path, split, **kwargs):

    if split == "test":
        motion_list = get_path(dataset_path, split)
        filename_list = [
            "-".join([i.split("/")[-3], i.split("/")[-1]]).split(".")[0]
            for i in motion_list
        ]
        motion_list = [torch.load(i,weights_only=False) for i in tqdm(motion_list)]
        return filename_list, motion_list

    assert split == "train"
    assert (
        "input_motion_length" in kwargs
    ), "Please specify the input_motion_length to load training dataset"

    motion_list = get_path(dataset_path, split)
    input_motion_length = kwargs["input_motion_length"]
    motion_list = [torch.load(i, weights_only=False) for i in tqdm(motion_list)]

    rot,sparse,trans,gposition,lposition = get_motion_trans(motion_list)

    new_rot = []
    new_sparse = []
    new_trans = []
    new_gposition = []
    new_lposition = []
    for idx, motion in enumerate(rot):
        if motion.shape[0] < input_motion_length:  # Arbitrary choice
            continue
        new_rot.append(rot[idx])
        new_sparse.append(sparse[idx])
        new_trans.append(trans[idx])
        new_gposition.append(gposition[idx])
        new_lposition.append(lposition[idx])

    return new_rot,new_sparse, new_trans, new_gposition, new_lposition

def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=32,
):

    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader