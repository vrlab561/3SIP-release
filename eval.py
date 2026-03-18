import glob
import os
import articulate as art
import torch
from utils.evaluator import Evaluator
from tqdm import tqdm
import math
import random
import numpy as np
from model.models import *
from data_loaders.dataloader import *
from human_body_prior.body_model.body_model import BodyModel as BM
from utils import utils_transform
from utils.metrics import get_metric_function, mpgre
from utils.parser_util import sample_args
from utils.evaluator import *
from human_body_prior.tools.rotation_tools import aa2matrot,local2global_pose
import time
device = torch.device("cuda")
smpl_path = './body_models/smpl_male.pkl'
smpl_path_root = "./body_models/"
dataset_path_amass = './datasets/AMASSIMU_/' 
dataset_path_xsens = './datasets/work_/'
model_path_amass = '.'
w_path_amass = "."
w_path_xsens = "."     
model_path_xsens = "."
pretrained_3sip = True
totalcapture = False
"""
    amass, xsens
"""
datatype = "amass"
if datatype == "amass":
    out_feats = 132
    code_seq = 10
else:
    out_feats = 84
    code_seq = 30

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)
METERS_TO_CENTIMETERS = 100.0

pred_metrics = [
    "mpjre",
    #"mpgre",
    "mpjpe",
    "mpjve",
    "pred_jitter",
]
gt_metrics = [
    "gt_jitter",
]
all_metrics = pred_metrics + gt_metrics

RADIANS_TO_DEGREES = 360.0 / (2 * math.pi)  # 57.2958 grads
metrics_coeffs = {
    "mpjre": RADIANS_TO_DEGREES,
    "mpgre": RADIANS_TO_DEGREES,
    "mpjpe": METERS_TO_CENTIMETERS,
    "mpjve": METERS_TO_CENTIMETERS,
    "pred_jitter": 0.01,
    "gt_jitter": 0.01
}

def get_body_model(support_dir, type="xsens"):
    if type == "xsens":
        body_model = XsensModel().to(device)
        return body_model.eval()
    else:
        subject_gender = "male"
        bm_fname = os.path.join(
            support_dir, "smplh/{}/model.npz".format(subject_gender)
        )
        dmpl_fname = os.path.join(
            support_dir, "dmpls/{}/model.npz".format(subject_gender)
        )
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        ).to(device)
        body_model = body_model.eval()
        return body_model

def evaluate_smpl(
    metrics,motion_pred,
    motion_gt,
    trans_gt,
    body_model,
    head_motion,
    fps,
    totalcapture = False
):
    # Get the  prediction from the model
    model_rot_input = (
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())
        .reshape(motion_pred.shape[0], -1)
        .float()
    )

    model_rot_gt = (
        utils_transform.sixd2aa(motion_gt.reshape(-1, 6).detach())
        .reshape(motion_gt.shape[0], -1)
        .float()
    )

    T_head2world = head_motion.clone()
    if not totalcapture:
        t_head2world = T_head2world[:, :3, 3].clone()
    else:
        t_head2world = T_head2world.clone()

    # Get the offset between the head and other joints using forward kinematic model
    body_pose_local = body_model(
        **{
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "trans": None
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]
    t_root2world = t_head2root + t_head2world
    # w.r.t first frame
    t_root2world = t_root2world-t_root2world[:1]

    predicted_body = body_model(
        **{
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "trans": t_root2world,
        }
    )
    predicted_position = predicted_body.Jtr[:, :22, :]

    # Get the predicted position and rotation
    predicted_angle = model_rot_input

    # Get the  ground truth position from the model
    gt_body = body_model(**{
            "pose_body": model_rot_gt[..., 3:66],
            "root_orient": model_rot_gt[..., :3],
            "trans": trans_gt,
        })
    gt_position = gt_body.Jtr[:, :22, :]

    gt_angle = model_rot_gt[:, 3:]
    gt_root_angle = model_rot_gt[:, :3]

    predicted_root_angle = predicted_angle[:, :3]
    predicted_angle = predicted_angle[:, 3:]

    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    eval_log = {}
    for metric in metrics:
        eval_log[metric] = (
            get_metric_function(metric)(
                predicted_position,
                predicted_angle,
                predicted_root_angle,
                gt_position,
                gt_angle,
                gt_root_angle,
                upper_index,
                lower_index,
                fps,
            )
            .cpu()
            .numpy()
        )

    torch.cuda.empty_cache()
    return eval_log

def evaluate_smpl_local(
    metrics,motion_pred,
    motion_gt,
    trans_gt,
    body_model,
    head_motion,
    fps,
    totalcapture = False
):
    # Get the  prediction from the model
    model_rot_input = (
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())
        .reshape(motion_pred.shape[0], -1)
        .float()
    )

    pred_local_matrot = aa2matrot(
        torch.tensor(model_rot_input).reshape(-1, 3)
    ).reshape(motion_pred.shape[0], -1, 9)
    pred_global_matrot = local2global_pose(
        pred_local_matrot, body_model.kintree_table[0][:22].long()
    )
    pred_rot_global = utils_transform.matrot2aa(pred_global_matrot.reshape(-1,3,3)).reshape(motion_pred.shape[0],-1)

    model_rot_gt = (
        utils_transform.sixd2aa(motion_gt.reshape(-1, 6).detach())
        .reshape(motion_gt.shape[0], -1)
        .float()
    )

    gt_local_matrot = aa2matrot(
        torch.tensor(model_rot_gt).reshape(-1, 3)
    ).reshape(motion_pred.shape[0], -1, 9)
    gt_global_matrot = local2global_pose(
        gt_local_matrot, body_model.kintree_table[0][:22].long()
    )
    gt_rot_global = utils_transform.matrot2aa(gt_global_matrot.reshape(-1,3,3)).reshape(motion_pred.shape[0],-1)

    predicted_body = body_model(
        **{
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "trans": None,
        }
    )
    predicted_position = predicted_body.Jtr[:, :22, :]

    # Get the predicted position and rotation
    predicted_angle = model_rot_input

    # Get the  ground truth position from the model
    gt_body = body_model(**{
            "pose_body": model_rot_gt[..., 3:66],
            "root_orient": model_rot_gt[..., :3],
            "trans": None,
        })
    gt_position = gt_body.Jtr[:, :22, :]

    gt_angle = model_rot_gt[:, 3:]
    gt_root_angle = model_rot_gt[:, :3]

    predicted_root_angle = predicted_angle[:, :3]
    predicted_angle = predicted_angle[:, 3:]

    upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_index = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    eval_log = {}
    for metric in metrics:
        if metric == "mpgre":
            eval_log[metric] = mpgre(pred_rot_global, gt_rot_global).item()
        else:
            eval_log[metric] = (
                get_metric_function(metric)(
                    predicted_position,
                    predicted_angle,
                    predicted_root_angle,
                    gt_position,
                    gt_angle,
                    gt_root_angle,
                    upper_index,
                    lower_index,
                    fps,
                )
                .cpu()
                .numpy()
            )

    torch.cuda.empty_cache()
    return eval_log

def evaluate_amass():
    fps = 60  # AMASS dataset requires 60 frames per second
    body_model = get_body_model(smpl_path_root, type="smpl")
    print("Loading dataset...")
    if not totalcapture:
        filename_list, all_info = load_data_trans(dataset_path_amass, "test")
        dataset = TestDatasetTrans(all_info, filename_list)
    else:
        dataset = TotalCapture(dataset_path_amass)
    log = {}
    for metric in all_metrics:
        log[metric] = 0
    if pretrained_3sip:
        model = ThreeSIP_divide_wwrapped(type=datatype, out_dim=out_feats, code_seq=code_seq)
        model.eval()
        state_dict_3sip = torch.load(model_path_amass, weights_only=False, map_location=device)
        model.ThreeSIP.load_state_dict(state_dict_3sip.state_dict(), strict=True)
        state_dict_w = torch.load(w_path_amass, weights_only=False, map_location=device)
        model.w.load_state_dict(state_dict_w.state_dict(), strict=True)
        print("fusion weight:")
        print(model.w.w)
        model.to(device)
    else:
        model = ThreeSIP_divide(type=datatype, out_dim=out_feats, code_seq=code_seq)  # 3sip
        model.eval()
        state_dict = torch.load(model_path_amass, weights_only=False, map_location=device)
        model.load_state_dict(state_dict.state_dict(), strict=True)
        model.to(device)
    cnt, t = 0, 0
    v0_frames = 8
    with torch.no_grad():
        for sample_index in tqdm(range(len(dataset))):
            rots = dataset[sample_index][0].cuda()
            sparse = dataset[sample_index][1].cuda()
            trans = dataset[sample_index][2].cuda()
            batch_size = sparse.shape[0]
            seq_len = sparse.shape[1]
            if totalcapture:
                gposition = dataset[sample_index][3].reshape(rots.shape[0], -1, 9).cuda()
            else:
                gposition = dataset[sample_index][3][..., [15, 20, 21], :].reshape(rots.shape[0], -1, 9).cuda()
            sparse_6d = sparse.view(batch_size, seq_len, 3, -1)[..., 3:]  # B, N, 3, 6
            sparse_acc = sparse.view(batch_size, seq_len, 3, -1)[..., :3]  # B, N, 3, 3
            sparse_ori = utils_transform.sixd2matrot(sparse_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 9)  # B, N, 3, 9
            sparse = torch.cat([sparse_ori, sparse_acc], dim=-1).reshape(batch_size, seq_len, -1)  # B, N, 36

            init_pose = rots[:, :1, :].repeat(1, rots.shape[1], 1)
            v0 = ((gposition[:, v0_frames:v0_frames + 1, ] - gposition[:, :1, ...]) / v0_frames * 60). \
                repeat(1, sparse.shape[1], 1).reshape(sparse.shape[0], -1, 9)  # B, N, 9
            motion_input = torch.cat([sparse, v0, init_pose], dim=-1).to(device)  # 27+9+132
            motion, cnt_, _, _ = model.online(motion_input)
            cnt += cnt_
            motion = motion[:, 1:, :]
            pose_fine = motion[..., :132]
            body_target_trans = (trans - trans[:, :1, :])[:, 1:, :]
            if not totalcapture:
                head_motion = dataset[sample_index][5][1:].cuda()
            else:
                head_motion = dataset[sample_index][3][0,1:,0].cuda()

            if not totalcapture:
                instance_log = evaluate_smpl(
                    metrics=all_metrics,
                    motion_pred=pose_fine[0],
                    motion_gt=rots[0, 1:, :],
                    trans_gt=body_target_trans[0],
                    body_model=body_model,
                    head_motion=head_motion,
                    fps=fps,totalcapture=totalcapture)
            else:
                instance_log = evaluate_smpl_local(
                    metrics=all_metrics,
                    motion_pred=pose_fine[0],
                    motion_gt=rots[0, 1:, :],
                    trans_gt=body_target_trans[0],
                    body_model=body_model,
                    head_motion=head_motion,
                    fps=fps, totalcapture=totalcapture)
            for key in instance_log:
                log[key] += instance_log[key]
    print("Metrics for the predictions")
    for metric in pred_metrics:
        print(metric, ":", log[metric] / len(dataset) * metrics_coeffs[metric])

def evaluate_xsens(model, dataset_: str, pose_evaluator=Evaluator(smpl_path=smpl_path, device=device),device_=device):
    total_error = []
    dataset = XsensTestDataset(dataset_dir=dataset_path_xsens,datasets=[dataset_])
    body_offset_error_cumulative_2s = 0
    cnt2s = 0
    body_offset_error_cumulative_5s = 0
    cnt5s = 0
    body_offset_error_cumulative_10s = 0
    cnt10s = 0
    with torch.no_grad():
        for sample_index in range(len(dataset)):
            rots = dataset[sample_index][0]
            sparse = dataset[sample_index][1]
            trans = dataset[sample_index][2].to(device)

            vel = dataset[sample_index][4]
            batch = rots.shape[0]
            seq_len = rots.shape[1]
            v0 = vel[:, [0], [1, 2, 3], :].reshape(batch, 1, 9).repeat(1, seq_len, 1)  # B, N, 9
            init_pose = rots[:, :1, :].repeat(1, rots.shape[1], 1)

            batch_size = sparse.shape[0]
            seq_len = sparse.shape[1]
            sparse_6d = sparse.view(batch_size, seq_len, 3, -1)[..., 3:]  # B, N, 3, 6
            sparse_acc = sparse.view(batch_size, seq_len, 3, -1)[..., :3]  # B, N, 3, 3
            sparse_ori = utils_transform.sixd2matrot(sparse_6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 9)  # B, N, 3, 9
            sparse = torch.cat([sparse_ori, sparse_acc], dim=-1).reshape(batch_size, seq_len, -1)  # B, N, 36

            motion_input = torch.cat([sparse, v0, init_pose], dim=-1).to(device)
            motion, _, _, _ = model.online(motion_input)
            output = motion[:, 1:, :]
            body_v_pred = output[..., 84:87]
            pose_fine = output[..., :84].squeeze().to(device_)

            body_target_trans = (trans - trans[:, :1, :])[:, 1:, :]
            body_pred_trans = torch.cumsum(body_v_pred, dim=1) / 60.0

            cnt2s_, cnt5s_, cnt10s_ = 0, 0, 0
            sparse_pred_seq_len = body_v_pred.shape[1]
            if sparse_pred_seq_len >= 120:
                body_pred_trans_ = body_pred_trans[:,120:,:]-body_pred_trans[:,:-120,:]
                body_target_trans_ = body_target_trans[:,120:,:]-body_target_trans[:,:-120,:]
                loss = torch.sqrt(torch.sum((body_pred_trans_[...,[0,2]] - body_target_trans_[...,[0,2]]).pow(2), dim=-1))
                body_offset_error_cumulative_2s += loss.mean().item()
                cnt2s += 1
                cnt2s_ = loss.mean()
            if sparse_pred_seq_len >= 300:
                body_pred_trans_ = body_pred_trans[:, 300:, :] - body_pred_trans[:, :-300, :]
                body_target_trans_ = body_target_trans[:, 300:, :] - body_target_trans[:, :-300, :]
                loss = torch.sqrt(torch.sum((body_pred_trans_[...,[0,2]] - body_target_trans_[...,[0,2]]).pow(2), dim=-1))
                body_offset_error_cumulative_5s += loss.mean().item()
                cnt5s += 1
                cnt5s_ = loss.mean()
            if sparse_pred_seq_len >= 600:
                body_pred_trans_ = body_pred_trans[:, 600:, :] - body_pred_trans[:, :-600, :]
                body_target_trans_ = body_target_trans[:, 600:, :] - body_target_trans[:, :-600, :]
                loss = torch.sqrt(torch.sum((body_pred_trans_[...,[0,2]] - body_target_trans_[...,[0,2]]).pow(2), dim=-1))
                body_offset_error_cumulative_10s += loss.mean().item()
                cnt10s += 1
                cnt10s_ = loss.mean()

            sparse_ori = sparse_6d.squeeze()[1:].to(device_)
            rots = rots.squeeze()[1:,:].to(device_)

            full_body_pose_pred = utils_transform._reduced_glb_6d_to_full_glb_mat_xsens(pose_fine[:, 6:],
                                                                                        pose_fine[:, :6], sparse_ori).to(device_)
            if dataset_ == "dip":
                full_body_pose_pred = utils_transform._glb_mat_xsens_to_glb_mat_smpl(full_body_pose_pred).to(device_)
                full_body_pose_gt = dataset[sample_index][5].squeeze()[1:,:].to(device_)
                err = pose_evaluator.eval_smpl(full_body_pose_pred, full_body_pose_gt)
            else:
                full_body_pose_gt = utils_transform._reduced_glb_6d_to_full_glb_mat_xsens(rots[:, 6:],
                                                                                          rots[:, :6], sparse_ori).to(device_)
                err = pose_evaluator.eval_xsens(full_body_pose_pred, full_body_pose_gt)

            total_error.append(err)

    total_error = torch.stack(total_error).mean(dim=0)
    print(total_error)
    if cnt2s > 0:
        print("cumulative body offset error(m) in 2s:{}".format(body_offset_error_cumulative_2s / cnt2s))
    if cnt5s > 0:
        print("cumulative body offset error(m) in 5s:{}".format(body_offset_error_cumulative_5s / cnt5s))
    if cnt10s > 0:
        print("cumulative body offset error(m) in 10s:{}".format(body_offset_error_cumulative_10s / cnt10s))
    return total_error

if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    assert datatype in ["amass","xsens"], "data type error"

    if datatype == "amass":
        evaluate_amass()
    else:
        print(model_path_xsens)
        smpl_model = get_body_model(smpl_path_root, type="smpl")

        if pretrained_3sip:
            model = ThreeSIP_divide_wwrapped(type=datatype, out_dim=out_feats, code_seq=code_seq)
            model.eval()
            state_dict_3sip = torch.load(model_path_xsens, weights_only=False, map_location=device)
            model.ThreeSIP.load_state_dict(state_dict_3sip.state_dict(), strict=True)
            state_dict_w = torch.load(w_path_xsens, weights_only=False, map_location=device)
            model.w.load_state_dict(state_dict_w.state_dict(), strict=True)
            print("fusion weight:")
            print(model.w.w)
            model.to(device)
        else:
            model = ThreeSIP_divide(type=datatype, out_dim=out_feats, code_seq=code_seq)
            model.eval()
            state_dict = torch.load(model_path_xsens, weights_only=False, map_location=device)
            model.load_state_dict(state_dict.state_dict(), strict=True)
            model.to(device)
        log = []
        datasets = ['dip', 'andy', 'unipd', 'cip', 'virginia']
        for ds in datasets:
            total_error = evaluate_xsens(model, ds)
            log.append([ds, total_error])

        print('-' * 75)
        print(f'{" " * 10:^10} | {"SIP Err":^18} | {"Global Angle Err":^18} | {"Joint Position Err":^18}')
        for (ds, err) in log:
            print('-' * 75)
            print(f'{ds:^10} | {"{}".format(round(err[0].item(), 2)):^18} | {"{}".format(round(err[1].item(), 2)):^18} | {"{}".format(round(err[2].item(), 2)):^18}')