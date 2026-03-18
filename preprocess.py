import articulate as art
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
import glob
from utils import utils_transform

amass_data = ["BioMotionLab_NTroje", "CMU", "MPI_HDM05"]
smpl_file = "./models/SMPL_male.pkl"
raw_amass_dir = "./source_data/AMASS/"
amass_dir = "./datasets/AMASSTC/"
raw_totalcapture_dip_dir = "./source_data/TotalCapture/DIP_recalculate/"
raw_totalcapture_official_dir = "./source_data/TotalCapture/official/"
totalcapture_dir = "./datasets/TotalCapture/"
body_model = art.ParametricModel(smpl_file)

def process_amass(smooth_n=4):

    def _syn_acc(v):
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

    vi_mask = torch.tensor([411, 1961, 5424])
    ji_mask = torch.tensor([15, 18, 19])
    body_model = art.ParametricModel(smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], [],
    out_imu, out_lposition, out_head = [], [], []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape=None, tran=tran[b:b + l], calc_mesh=True)
        pose_ = art.math.rotation_matrix_to_r6d(p).reshape(-1,24,6)[:,:22]
        out_pose.append(pose_.clone())  # N, 22, 6
        out_tran.append(tran[b:b + l].clone())  # N, 3
        global_joints_positions = joint[:, :22].contiguous().clone()
        local_joints_positions = global_joints_positions - tran[b:b + l].view(-1,1,3)
        out_joint.append(global_joints_positions)  # N, 22, 3
        out_lposition.append(local_joints_positions) # N, 22, 3
        vacc = _syn_acc(vert[:, vi_mask])   # N, 3, 3
        vrot = grot[:, ji_mask] # N, 3, 3, 3
        vrot = art.math.rotation_matrix_to_r6d(vrot).reshape(-1, 3, 6)
        imu = torch.cat([vacc, vrot], dim=2).reshape(-1, 27)
        out_imu.append(imu.clone())
        b += l

    print('Saving')
    os.makedirs(amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(amass_dir, 'pose.pt'))
    torch.save(out_tran, os.path.join(amass_dir, 'trans.pt'))
    torch.save(out_joint, os.path.join(amass_dir, 'gposition.pt'))
    torch.save(out_lposition, os.path.join(amass_dir, 'lposition.pt'))
    torch.save(out_imu, os.path.join(amass_dir, 'sparse.pt'))
    print('Synthetic AMASS dataset is saved at', amass_dir)

vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
vi_mask_3 = torch.tensor([411, 1961, 5424])

def vector_cross_matrix(x: torch.Tensor):
    r"""
    Get the skew-symmetric matrix :math:`[v]_\times\in so(3)` for each vector3 `v`. (torch, batch)

    :param x: Tensor that can reshape to [batch_size, 3].
    :return: The skew-symmetric matrix in shape [batch_size, 3, 3].
    """
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)

def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)

def axis_angle_to_rotation_matrix(a: torch.Tensor):
    r"""
    Turn axis-angles into rotation matrices. (torch, batch)

    :param a: Axis-angle tensor that can reshape to [batch_size, 3].
    :return: Rotation matrix of shape [batch_size, 3, 3].
    """
    axis, angle = normalize_tensor(a.view(-1, 3), return_norm=True)
    axis[torch.isnan(axis) | torch.isinf(axis)] = 0
    i_cube = torch.eye(3, device=a.device).expand(angle.shape[0], 3, 3)
    c, s = angle.cos().view(-1, 1, 1), angle.sin().view(-1, 1, 1)
    r = c * i_cube + (1 - c) * torch.bmm(axis.view(-1, 3, 1), axis.view(-1, 1, 3)) + s * vector_cross_matrix(axis)
    return r

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

def process_totalcapture_3sip():
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'

    sparses, accs , poses, pose6d, trans, gpositions, lpositions = [], [], [], [], [], [], []
    height = []
    for file in sorted(os.listdir(raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([4, 0, 1])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([4, 0, 1])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        ori = utils_transform.matrot2sixd(  # N, 3, 6
            ori.reshape(-1, 3, 3)
        ).reshape(acc.shape[0], -1, 6)
        pose_rm = pose
        pose_aa = pose
        pose = utils_transform.aa2sixd(pose.reshape(-1,3)).reshape(-1,24,6)
        pose = pose[:,:22,:]#.reshape(-1,132)    # N, 22, 6
        sparse = torch.cat([acc, ori], dim=2)#.reshape(acc.shape[0], 27)  # N, 27
        sparses.append(sparse)
        accs.append(acc)
        pose6d.append(pose)  # N, 22, 6
        poses.append(pose_aa)
        pose_rm = art.math.axis_angle_to_rotation_matrix(pose_rm).view(-1, 24, 3, 3)
        _, pos, _ = body_model.forward_kinematics(pose_rm, tran=None, calc_mesh=True)
        lpositions.append(pos[:,:22])

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue   # no SMPL poses
            f = open(os.path.join(raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['Head', 'RightFoot', 'LeftFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            height.append(max(pos[:,0,1]))
            trans.append(pos[:, 3] - pos[:1, 3])   # N, 3 calculate translation using spine's position

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

        # remove acceleration bias
    for iacc, pose, tran, sparse in zip(accs, poses, trans, sparses):
        pose = axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        _, gpp, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        gpositions.append(gpp[:, [15, 18, 19]].reshape(-1, 9))
        vacc = _syn_acc(vert[:, vi_mask_3])
        for imu_id in range(3):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d
        sparse[..., :3] = iacc

    sparse_, pose_, trans_, gposition_, lposition_ = [], [], [], [], []
    for idx in range(len(sparses)):
        sparse_.append(sparses[idx].reshape(-1, 27))
        pose_.append(pose6d[idx])
        trans_.append(trans[idx])
        lposition_.append(lpositions[idx])
        gposition_.append(gpositions[idx])

    os.makedirs(totalcapture_dir, exist_ok=True)
    torch.save({'sparse': sparse_, 'pose': pose_, 'trans': trans_,
                'gposition': gposition_, 'lposition': lposition_},
               os.path.join(totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', totalcapture_dir)

if __name__ == '__main__':
    process_amass()
    process_totalcapture_3sip()
