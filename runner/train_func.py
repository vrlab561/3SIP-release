import torch
import torch.nn.functional as F
from utils import utils_transform
from utils import tools
import torch.nn as nn

def update_lr(
    nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
):
    if nb_iter > lr_anneal_steps:
        current_lr = min_lr
    else:
        current_lr = max_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def eudistance(pred,gt):
    loss = torch.sqrt(torch.sum((gt-pred).pow(2),dim=-1)).mean()
    return loss

def velocityLoss(loss_func, pred, gt, interval=1):
    '''
    pred: shape=(batch, seq_len, joint_num, 3)
    '''
    batch, seq_len = pred.shape[0], pred.shape[1]
    pred = pred.reshape(batch, seq_len, -1)
    gt = gt.reshape(batch, seq_len, -1)
    target_vel = gt[:, interval::interval, :] - gt[:, :-interval:interval, :]
    pred_vel = pred[:, interval::interval, :] - pred[:, :-interval:interval, :]
    velocity_loss = loss_func(target_vel, pred_vel)
    return velocity_loss

def jitter(predicted_position,fps=60):
    pred_jitter = (
        (
                (
                        predicted_position[:,3:]
                        - 3 * predicted_position[:,2:-1]
                        + 3 * predicted_position[:,1:-2]
                        - predicted_position[:,:-3]
                )
                #* (fps ** 3)
        )
        .norm(dim=2)
        .mean()
    )
    return pred_jitter

def train_step_vqcm(
    pose_target,    # final pose target     B,N,132
    model,
    optimizer,
    nb_iter,
    total_iter,
    max_lr,
    min_lr,
    device,
    lr_anneal_steps,
    body_model,loss_func,datatype="xsens",sparse=None,
    cm_weight=0.1,fp_weight=1.0,bv_weight=0.0,slp_weight=1.0,cp_weight=0.0  #with slp
):

    pose_target = pose_target.to(device)
    batch, seq = pose_target.shape[0], pose_target.shape[1]
    pose_pred, ze_y, zq_y = model(y=pose_target)

    fp_loss = loss_func(pose_target, pose_pred)
    l_embedding = loss_func(ze_y.detach(), zq_y)
    l_commitment = loss_func(ze_y, zq_y.detach())
    codebook_matching_loss = l_embedding + l_commitment * 0.25
    #slp_loss = torch.zeros([1],device=fp_loss.device)
    if datatype == "xsens":
        out_pose = pose_pred.reshape(-1, 84)
        rots = pose_target.reshape(-1, 84)

        global_orientation = out_pose[:, :6]
        joint_rotation = out_pose[:, 6:]
        #sparse_ori = sparse.view(-1, 3, 12)[..., 3:]  # n,3,6
        batch_size, seq_len = sparse.shape[0], sparse.shape[1]
        sparse_6d = sparse.view(batch_size, seq_len, 3, -1)[..., 3:]
        joint_p = body_model(joint_rotation, global_orientation, sparse_6d.reshape(-1,3,6)).reshape(batch,seq,-1)#.reshape(-1, 23, 3)

        global_orientation = rots[:, :6]
        joint_rotation = rots[:, 6:]
        joint_t = body_model(joint_rotation, global_orientation, sparse_6d.reshape(-1,3,6)).reshape(batch,seq,-1)#.reshape(-1, 23, 3)
        slp_loss = loss_func(joint_p, joint_t)
    else:
        out_pose = pose_pred.reshape(-1, 132)
        rots = pose_target.reshape(-1, 132)

        out_pose = tools.sixd2aa(out_pose.reshape(-1, 6)).reshape(out_pose.shape[0], -1).float()
        rots = tools.sixd2aa(rots.reshape(-1, 6)).reshape(rots.shape[0], -1).float()

        zero_trans = torch.zeros((1, 3), dtype=torch.float32).expand(rots.shape[0], -1).cuda()

        joint_p = body_model(out_pose[:, 3:], out_pose[:, :3], trans=zero_trans).Jtr[:, :22, :].reshape(batch,seq,-1)
        joint_t = body_model(rots[:, 3:], rots[:, :3], trans=zero_trans).Jtr[:, :22, :].reshape(batch,seq,-1)
        slp_loss = loss_func(joint_p, joint_t)
        #slp_loss = torch.zeros([1]).cuda()

    velocity_error_scale = 20#10
    # velocity loss
    global_velocity_loss_1 = velocityLoss(loss_func=loss_func, pred=joint_p,
                                        gt=joint_t) * velocity_error_scale
    bv_loss = global_velocity_loss_1

    global_velocity_loss_3 = velocityLoss(loss_func=loss_func, pred=joint_p, gt=joint_t,
                                        interval=3) * velocity_error_scale / 3
    bv_loss += global_velocity_loss_3

    global_velocity_loss_5 = velocityLoss(loss_func=loss_func, pred=joint_p, gt=joint_t,
                                        interval=5) * velocity_error_scale / 5
    bv_loss += global_velocity_loss_5

    cp_loss = jitter(joint_p)

    loss = codebook_matching_loss*cm_weight + fp_loss*fp_weight + slp_loss*slp_weight + bv_loss*bv_weight + cp_loss*cp_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer, current_lr = update_lr(
        nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
    )

    return loss.item(), codebook_matching_loss.item(), fp_loss.item(), bv_loss.item(), slp_loss.item(), cp_loss.item(),\
        optimizer, current_lr

def train_step_tp_vqcm_search(
    sparse,         # for pose estimate     B,N,27
    v0,  # for offset optimize   B,N,9
    body_v_target,   # body velocity target   B,N,3
    pose_target,    # final pose target     B,N,132
    sparse_v_target,   # sparse velocity target  B,N,9
    sparse_p_target,  # sparse local position target B,N,9
    model,
    optimizer,
    nb_iter,
    total_iter,
    max_lr,
    min_lr,
    device,
    lr_anneal_steps,
    body_model,loss_func,datatype="xsens",wsearch=None
):
    init_pose = pose_target[:, :1, :].repeat(1, pose_target.shape[1], 1)
    batch_size = sparse.shape[0]
    seq_len = sparse.shape[1]
    sparse_6d = sparse.view(batch_size, seq_len, 3, -1)[..., 3:]  # B, N, 3, 6
    sparse_acc = sparse.view(batch_size, seq_len, 3, -1)[..., :3]  # B, N, 3, 3
    sparse_ori = utils_transform.sixd2matrot(sparse_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 9)  # B, N, 3, 9
    sparse = torch.cat([sparse_ori, sparse_acc], dim=-1).reshape(batch_size, seq_len, -1)  # B, N, 36

    motion_input = torch.cat([sparse, v0, init_pose], dim=-1).to(device)

    pose_target = pose_target.to(device)

    with torch.no_grad():
        fine_pose, _, fine_pose_vq = wsearch.divide_online(motion_input)

    motion = model(fine_pose, fine_pose_vq)
    pose_target = pose_target[:, 1:, :]
    pose_fusion = motion[:, 1:, :]

    loss = loss_func(pose_target, pose_fusion)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']

    return loss.item(), 0.0, 0.0, 0.0, 0.0, 0.0, optimizer, current_lr

def train_step_tp_vqcm(
    sparse,         # for pose estimate     B,N,27
    v0,  # for offset optimize   B,N,9
    body_v_target,   # body velocity target   B,N,3
    pose_target,    # final pose target     B,N,132
    sparse_v_target,   # sparse velocity target  B,N,9
    sparse_p_target,  # sparse local position target B,N,9
    model,
    optimizer,
    nb_iter,
    total_iter,
    max_lr,
    min_lr,
    device,
    lr_anneal_steps,
    body_model,loss_func,datatype="xsens",wsearch=None,
    cm_weight=0.1,fp_weight=1.0,bv_weight=0.1,slp_weight=0.0,cp_weight=1.0  #xsens nofusion
):
    init_pose = pose_target[:, :1, :].repeat(1, pose_target.shape[1], 1)
    batch_size = sparse.shape[0]
    seq_len = sparse.shape[1]
    sparse_6d = sparse.view(batch_size, seq_len, 3, -1)[..., 3:]  # B, N, 3, 6
    sparse_acc = sparse.view(batch_size, seq_len, 3, -1)[..., :3]  # B, N, 3, 3
    sparse_ori = utils_transform.sixd2matrot(sparse_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 9)  # B, N, 3, 9
    sparse = torch.cat([sparse_ori, sparse_acc], dim=-1).reshape(batch_size, seq_len, -1)  # B, N, 36

    motion_input = torch.cat([sparse, v0, init_pose], dim=-1).to(device)

    body_v_target = body_v_target.to(device)
    pose_target = pose_target.to(device)
    sparse_v_target = sparse_v_target.to(device)
    sparse_p_target = sparse_p_target.to(device)

    motion, ze_y, zq_y, ze_f = model(motion_input, y=pose_target)
    pose_target = pose_target[:, 1:, :]
    codebook_matching_loss = loss_func(ze_y, ze_f)
    output = motion[:, 1:, :]

    if datatype == "xsens":
        body_v_pred = output[..., 84:84+12]
        pose_pred = output[..., :84]
        pose_contract = output[..., -84:]

        cp_loss = loss_func(pose_target, pose_contract)
        fp_loss = loss_func(pose_target, pose_pred)
        out_pose = pose_pred.reshape(-1, 84)
        rots = pose_target.reshape(-1, 84)

        global_orientation = out_pose[:, :6]
        joint_rotation = out_pose[:, 6:]
        joint_p = body_model(joint_rotation, global_orientation, sparse_6d[:,1:].reshape(-1,3,6)).reshape(-1, 23, 3)

        global_orientation = rots[:, :6]
        joint_rotation = rots[:, 6:]
        joint_t = body_model(joint_rotation, global_orientation, sparse_6d[:,1:].reshape(-1,3,6)).reshape(-1, 23, 3)
        slp_loss = loss_func(joint_p, joint_t)
    else:
        body_v_pred = output[...,132:132+12]
        pose_pred = output[...,:132]
        pose_contract = output[..., -132:]
        
        out_pose = pose_pred.reshape(-1, 132)
        rots = pose_target.reshape(-1, 132)

        cp_loss = loss_func(pose_target, pose_contract)
        fp_loss = loss_func(pose_target, pose_pred)
        out_pose = tools.sixd2aa(out_pose.reshape(-1, 6)).reshape(out_pose.shape[0], -1).float()
        rots = tools.sixd2aa(rots.reshape(-1, 6)).reshape(rots.shape[0], -1).float()
        
        zero_trans = torch.zeros((1, 3), dtype=torch.float32).expand(rots.shape[0], -1).cuda()
        joint_p = body_model(out_pose[:, 3:], out_pose[:, :3], trans=zero_trans).Jtr[:, :22, :]
        joint_t = body_model(rots[:, 3:], rots[:, :3], trans=zero_trans).Jtr[:, :22, :]
        slp_loss = loss_func(joint_p, joint_t)

    bv_loss = loss_func(torch.cat([body_v_target, sparse_v_target], dim=-1), body_v_pred)

    loss = codebook_matching_loss*cm_weight + fp_loss*fp_weight + slp_loss*slp_weight + bv_loss*bv_weight + cp_loss*cp_weight

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']

    return loss.item(), codebook_matching_loss.item(), fp_loss.item(), bv_loss.item(), slp_loss.item(), cp_loss.item(),\
        optimizer, current_lr

def square_loss1(pred,gt):
    loss = torch.mean((gt - pred).pow(2), dim=0).sum()
    return loss