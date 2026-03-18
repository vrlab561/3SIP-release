import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from data_loaders.dataloader import *
from runner.train_func import *
from human_body_prior.body_model.body_model import BodyModel as BM
from model.models import *
from utils.parser_util import train_args
from utils.skeleton import *

program = "3SIP"
smpl_path_root = "./body_models/"
dataset_path_xsens = './datasets/work/'
dataset_path_amass = './datasets/AMASSTC/'        
vqcm_path = "./saved_models/vqcm.pth"
threesip_path = "./saved_models/3SIP.pth"
"""
    amass xsens
"""
datatype = "xsens"
totalcapture = True
pretrained_3sip = True
pretrained_vqcm = True

loss_func = nn.MSELoss()
if datatype == "amass":
    seq_len = 60
    out_feats = 132
    code_seq = 10
else:
    seq_len = 300
    out_feats = 84
    code_seq = 30
print("seq_len:",seq_len)
device = torch.device("cuda")

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

def freeze_model(model, target):
    for (name, param) in model.named_parameters():
        if target in name:
            print("freeze "+name)
            param.requires_grad = False
    return model

def printAndSave(avg_loss,avg_lr,loss,current_lr,args,nb_iter,train_loss_list,model,
        avg_cm_loss=None, cm_loss=None, cm_loss_list=None,
        avg_fp_loss=None, fp_loss=None, fp_loss_list=None,
        avg_bv_loss=None, bv_loss=None, bv_loss_list=None,
        avg_slp_loss=None, slp_loss=None, slp_loss_list=None,
        avg_cp_loss=None, cp_loss=None, cp_loss_list=None,
        totalcapture=False):
    cnt = 0
    others = {}
    avg_loss += loss
    avg_lr += current_lr
    ifBreak = False
    if (nb_iter + 1) % args.log_interval == 0:
        avg_loss = avg_loss / args.log_interval
        avg_lr = avg_lr / args.log_interval
        print("Iter {} Summary: ".format(nb_iter + 1))
        print(f"\t lr: {current_lr} \t Training loss: {avg_loss}")
        train_loss_list.append(avg_loss)
        with open("./train_loss_{program}_{phase}.txt".format(program=program,phase=datatype), "w") as train_los:
            train_los.write(str(train_loss_list))
        avg_loss = 0
        avg_lr = 0

    if (nb_iter + 1) == args.num_steps:
        ifBreak = True

    if cm_loss is not None:
        avg_cm_loss += cm_loss
        if (nb_iter + 1) % args.log_interval == 0:
            avg_cm_loss = avg_cm_loss / args.log_interval
            print(f"\t codebook matching loss: {avg_cm_loss}")
            cm_loss_list.append(avg_cm_loss)
            with open("./cm_loss_{program}_{phase}.txt".format(program=program, phase=datatype), "w") as cm_los:
                cm_los.write(str(cm_loss_list))
            avg_cm_loss = 0
        others['avg_cm_loss'] = avg_cm_loss

    if fp_loss is not None:
        avg_fp_loss += fp_loss
        if (nb_iter + 1) % args.log_interval == 0:
            avg_fp_loss = avg_fp_loss / args.log_interval
            print(f"\t fine pose loss: {avg_fp_loss}")
            fp_loss_list.append(avg_fp_loss)
            with open("./fp_loss_{program}_{phase}.txt".format(program=program, phase=datatype), "w") as fp_los:
                fp_los.write(str(fp_loss_list))
            avg_fp_loss = 0
        others['avg_fp_loss'] = avg_fp_loss

    if bv_loss is not None:
        avg_bv_loss += bv_loss
        if (nb_iter + 1) % args.log_interval == 0:
            avg_bv_loss = avg_bv_loss / args.log_interval
            print(f"\t body velocity loss: {avg_bv_loss}")
            bv_loss_list.append(avg_bv_loss)
            with open("./bv_loss_{program}_{phase}.txt".format(program=program, phase=datatype), "w") as bv_los:
                bv_los.write(str(bv_loss_list))
            avg_bv_loss = 0
        others['avg_bv_loss'] = avg_bv_loss

    if slp_loss is not None:
        avg_slp_loss += slp_loss
        if (nb_iter + 1) % args.log_interval == 0:
            avg_slp_loss = avg_slp_loss / args.log_interval
            print(f"\t sparse local position loss: {avg_slp_loss}")
            slp_loss_list.append(avg_slp_loss)
            with open("./slp_loss_{program}_{phase}.txt".format(program=program, phase=datatype), "w") as slp_los:
                slp_los.write(str(slp_loss_list))
            avg_slp_loss = 0
        others['avg_slp_loss'] = avg_slp_loss

    if cp_loss is not None:
        avg_cp_loss += cp_loss
        if (nb_iter + 1) % args.log_interval == 0:
            avg_cp_loss = avg_cp_loss / args.log_interval
            print(f"\t contract pose loss: {avg_cp_loss}")
            cp_loss_list.append(avg_cp_loss)
            with open("./cp_loss_{program}_{phase}.txt".format(program=program, phase=datatype), "w") as cp_los:
                cp_los.write(str(cp_loss_list))
            avg_cp_loss = 0
        others['avg_cp_loss'] = avg_cp_loss

    nb_iter += 1
    return avg_loss, avg_lr, nb_iter, ifBreak, others#, train_loss

def update_lr_multistep(
    nb_iter, total_iter, max_lr, min_lr, optimizer, lr_anneal_steps
):
    if nb_iter > lr_anneal_steps:
        current_lr = min_lr
    else:
        current_lr = max_lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def train_vqcm(args, dataloader):
    xsens_model = get_body_model(smpl_path_root, type="xsens")
    smpl_model = get_body_model(smpl_path_root, type="smpl")
    model = ThreeSIP_vqcm(type=datatype, out_dim=out_feats, code_seq=code_seq)
    model.train()
    model.to(device)
    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0)
    )
    # initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0
    train_loss = []

    avg_slp_loss = 0.0  # sparse local position loss
    slp_loss_list = []

    avg_fp_loss = 0.0 # fine pose loss
    fp_loss_list = []

    avg_bv_loss = 0.0   # body velocity loss
    bv_loss_list = []

    avg_cm_loss = 0.0   # sparse velocity loss
    cm_loss_list = []

    avg_cp_loss = 0.0   # coarse pose loss
    cp_loss_list = []

    epoch = -1
    while (nb_iter + 1) < args.num_steps:
        epoch += 1
        print("epoch:{}".format(epoch))
        if datatype == "xsens":
            if epoch%10 == 0:
                torch_model = torch.jit.script(model)
                torch.jit.save(torch_model, './saved_models/{program}_{phase}_{iter}.pth'. \
                               format(program="vqcm", phase=datatype, iter=epoch))
            for (rot, sparse, _, _, _) in dataloader:
                rot = rot.squeeze()
                sparse = sparse.squeeze().cuda()
                loss, cm_loss, fp_loss, bv_loss, slp_loss, cp_loss, optimizer, current_lr = \
                    train_step_vqcm(
                        pose_target=rot.cuda(),
                        model=model,
                        optimizer=optimizer,
                        nb_iter=nb_iter,
                        total_iter=args.num_steps,
                        max_lr=args.lr,
                        min_lr=args.lr / 10.0,
                        device=device, body_model=xsens_model, loss_func=loss_func,
                        lr_anneal_steps=args.lr_anneal_steps,
                        datatype=datatype,sparse=sparse
                    )
                avg_loss, avg_lr, nb_iter, ifBreak, others = \
                    printAndSave(avg_loss, avg_lr, loss, current_lr, args, nb_iter, train_loss, model,
                                 avg_cm_loss=avg_cm_loss, cm_loss=cm_loss, cm_loss_list=cm_loss_list,
                                 avg_fp_loss=avg_fp_loss, fp_loss=fp_loss, fp_loss_list=fp_loss_list,
                                 avg_bv_loss=avg_bv_loss, bv_loss=bv_loss, bv_loss_list=bv_loss_list,
                                 avg_slp_loss=avg_slp_loss, slp_loss=slp_loss, slp_loss_list=slp_loss_list,
                                 avg_cp_loss=avg_cp_loss, cp_loss=cp_loss, cp_loss_list=cp_loss_list)
                if others is not None:
                    avg_cm_loss = others['avg_cm_loss']
                    avg_fp_loss = others['avg_fp_loss']
                    avg_bv_loss = others['avg_bv_loss']
                    avg_slp_loss = others['avg_slp_loss']
                    avg_cp_loss = others['avg_cp_loss']
                if ifBreak:
                    break
        else:
            if epoch%10 == 0:
                torch_model = torch.jit.script(model)
                torch.jit.save(torch_model, './saved_models/{program}_{phase}_{iter}.pth'. \
                                   format(program="vqcm", phase=datatype, iter=epoch))
            for (rot, _, _, _, _) in dataloader:
                rot = rot.squeeze()
                loss, cm_loss, fp_loss, bv_loss, slp_loss, cp_loss, optimizer, current_lr = \
                    train_step_vqcm(
                        pose_target=rot.cuda(),
                        model=model,
                        optimizer=optimizer,
                        nb_iter=nb_iter,
                        total_iter=args.num_steps,
                        max_lr=args.lr,
                        min_lr=args.lr / 10.0,
                        device=device, body_model=smpl_model, loss_func=loss_func,
                        lr_anneal_steps=args.lr_anneal_steps,
                        datatype=datatype
                    )
                avg_loss, avg_lr, nb_iter, ifBreak, others = \
                    printAndSave(avg_loss, avg_lr, loss, current_lr, args, nb_iter, train_loss, model,
                                 avg_cm_loss=avg_cm_loss, cm_loss=cm_loss, cm_loss_list=cm_loss_list,
                                 avg_fp_loss=avg_fp_loss, fp_loss=fp_loss, fp_loss_list=fp_loss_list,
                                 avg_bv_loss=avg_bv_loss, bv_loss=bv_loss, bv_loss_list=bv_loss_list,
                                 avg_slp_loss=avg_slp_loss, slp_loss=slp_loss, slp_loss_list=slp_loss_list,
                                 avg_cp_loss=avg_cp_loss, cp_loss=cp_loss, cp_loss_list=cp_loss_list,
                                 totalcapture=True)
                if others is not None:
                    avg_cm_loss = others['avg_cm_loss']
                    avg_fp_loss = others['avg_fp_loss']
                    avg_bv_loss = others['avg_bv_loss']
                    avg_slp_loss = others['avg_slp_loss']
                    avg_cp_loss = others['avg_cp_loss']
                if ifBreak:
                    break

def train_model(args, dataloader):
    xsens_model = get_body_model(smpl_path_root, type="xsens")
    smpl_model = get_body_model(smpl_path_root, type="smpl")
    Threesip = None
    if pretrained_3sip:
        model = ThreeSIP_divide_wsearch()
        Threesip = ThreeSIP_divide(type=datatype, out_dim=out_feats, code_seq=code_seq)
        Threesip.eval()
        state_dict = torch.load(threesip_path, weights_only=False, map_location=device)
        Threesip.load_state_dict(state_dict.state_dict(), strict=True)
        Threesip.to(device)
    elif pretrained_vqcm:
        model = ThreeSIP_divide(type=datatype, out_dim=out_feats, code_seq=code_seq)
        state_dict = torch.load(vqcm_path, weights_only=False, map_location=device)
        model.vqcm.load_state_dict(state_dict.state_dict(), strict=True)
        model = freeze_model(model,"vqcm")
    else:
        train_vqcm(args, dataloader)
        return

    model.train()
    model.to(device)
    print(
        "Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0)
    )
    # initialize optimizer
    if datatype == "xsens":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr
        )

    if pretrained_3sip:
        train_step = train_step_tp_vqcm_search
    else:
        train_step = train_step_tp_vqcm

    nb_iter = 0
    avg_loss = 0.0
    avg_lr = 0.0
    train_loss = []

    avg_slp_loss = 0.0  # sparse local position loss
    slp_loss_list = []

    avg_fp_loss = 0.0 # fine pose loss
    fp_loss_list = []

    avg_bv_loss = 0.0   # body velocity loss
    bv_loss_list = []

    avg_cm_loss = 0.0   # sparse velocity loss
    cm_loss_list = []

    avg_cp_loss = 0.0   # coarse pose loss
    cp_loss_list = []

    v0_frames = 8
    epoch = -1
    while (nb_iter + 1) < args.num_steps:
        epoch += 1
        print("epoch:{}".format(epoch))
        if pretrained_3sip:
            print(model.w)
        if datatype == "xsens":
            if pretrained_3sip:
                torch_model = torch.jit.script(model)
                torch.jit.save(torch_model, './saved_models/{program}_{phase}_{iter}.pth'. \
                               format(program=program, phase=datatype, iter=epoch))
            elif epoch % 10 == 0:
                torch_model = torch.jit.script(model)
                torch.jit.save(torch_model, './saved_models/{program}_{phase}_{iter}.pth'. \
                                   format(program=program, phase=datatype, iter=epoch))
            for (rot, sparse, trans, lposition, vel) in dataloader:
                batch = rot.shape[0]
                seq_len = rot.shape[1]
                v0 = vel[:,[0],[1,2,3],:].reshape(batch,1,9).repeat(1,seq_len,1).cuda() # B, N, 9
                sparse_g_vel = vel[...,[1,2,3],:].reshape(batch, seq_len, 9)[:,1:,:].cuda()
                body_vel_target = vel[...,0,:].reshape(batch, seq_len, 3)[:,1:,:].cuda()
                loss, cm_loss, fp_loss, bv_loss, slp_loss, cp_loss, optimizer, current_lr = \
                    train_step(
                        sparse=sparse.cuda(),
                        v0=v0,
                        body_v_target=body_vel_target,
                        pose_target=rot.cuda(),
                        sparse_v_target=sparse_g_vel,
                        sparse_p_target=lposition[:, 1:, :],
                        model=model,
                        optimizer=optimizer,
                        nb_iter=nb_iter,
                        total_iter=args.num_steps,
                        max_lr=args.lr,
                        min_lr=args.lr / 10.0,
                        device=device,body_model=xsens_model,loss_func=loss_func,
                        lr_anneal_steps=args.lr_anneal_steps,
                        datatype=datatype, wsearch=Threesip
                    )
                avg_loss, avg_lr, nb_iter, ifBreak, others = \
                    printAndSave(avg_loss, avg_lr, loss, current_lr, args, nb_iter, train_loss, model,
                                 avg_cm_loss=avg_cm_loss, cm_loss=cm_loss, cm_loss_list=cm_loss_list,
                                 avg_fp_loss=avg_fp_loss, fp_loss=fp_loss, fp_loss_list=fp_loss_list,
                                 avg_bv_loss=avg_bv_loss, bv_loss=bv_loss, bv_loss_list=bv_loss_list,
                                 avg_slp_loss=avg_slp_loss, slp_loss=slp_loss, slp_loss_list=slp_loss_list,
                                 avg_cp_loss=avg_cp_loss, cp_loss=cp_loss, cp_loss_list=cp_loss_list)
                if others is not None:
                    avg_cm_loss = others['avg_cm_loss']
                    avg_fp_loss = others['avg_fp_loss']
                    avg_bv_loss = others['avg_bv_loss']
                    avg_slp_loss = others['avg_slp_loss']
                    avg_cp_loss = others['avg_cp_loss']
                if ifBreak:
                    break
        else:
            if totalcapture or pretrained_3sip:
                torch_model = torch.jit.script(model)
                torch.jit.save(torch_model, './saved_models/{program}_{phase}_{iter}.pth'. \
                                   format(program=program, phase=datatype, iter=epoch))
            elif epoch%10 == 0:
                torch_model = torch.jit.script(model)
                torch.jit.save(torch_model, './saved_models/{program}_{phase}_{iter}.pth'. \
                               format(program=program, phase=datatype, iter=epoch))
            for (rot, sparse, trans, gposition, lposition) in dataloader:
                rot = rot.squeeze()
                sparse = sparse.squeeze()
                trans = trans.squeeze()
                gposition = gposition.squeeze()[...,[15,18,19],:].reshape(rot.shape[0],-1,9)
                lposition = lposition.squeeze().reshape(rot.shape[0],-1,66)
                v0 = ((gposition[:, v0_frames:v0_frames + 1, ] - gposition[:, :1, ...]) / v0_frames * 60). \
                    repeat(1, sparse.shape[1], 1).reshape(sparse.shape[0], -1, 9)  # B, N, 9
                sparse_g_vel = (gposition[:, 1:, :] - gposition[:, :-1, :]) * 60.0  # B, N, 9
                body_vel_target = (trans[:, 1:, :] - trans[:, :-1, :]) * 60.0
                loss, cm_loss, fp_loss, bv_loss, slp_loss, cp_loss, optimizer, current_lr = \
                    train_step(
                        sparse=sparse.cuda(),
                        v0=v0.cuda(),
                        body_v_target=body_vel_target,
                        pose_target=rot.cuda(),
                        sparse_v_target=sparse_g_vel,
                        sparse_p_target=lposition[:, 1:, :],
                        model=model,
                        optimizer=optimizer,
                        nb_iter=nb_iter,
                        total_iter=args.num_steps,
                        max_lr=args.lr,
                        min_lr=args.lr / 10.0,
                        device=device, body_model=smpl_model, loss_func=loss_func,
                        lr_anneal_steps=args.lr_anneal_steps,
                        datatype=datatype, wsearch=Threesip
                    )
                # print(scheduler.get_lr()[0])
                avg_loss, avg_lr, nb_iter, ifBreak, others = \
                    printAndSave(avg_loss, avg_lr, loss, current_lr, args, nb_iter, train_loss, model,
                                 avg_cm_loss=avg_cm_loss, cm_loss=cm_loss, cm_loss_list=cm_loss_list,
                                 avg_fp_loss=avg_fp_loss, fp_loss=fp_loss, fp_loss_list=fp_loss_list,
                                 avg_bv_loss=avg_bv_loss, bv_loss=bv_loss, bv_loss_list=bv_loss_list,
                                 avg_slp_loss=avg_slp_loss, slp_loss=slp_loss, slp_loss_list=slp_loss_list,
                                 avg_cp_loss=avg_cp_loss, cp_loss=cp_loss, cp_loss_list=cp_loss_list,
                                 totalcapture=totalcapture)
                if others is not None:
                    avg_cm_loss = others['avg_cm_loss']
                    avg_fp_loss = others['avg_fp_loss']
                    avg_bv_loss = others['avg_bv_loss']
                    avg_slp_loss = others['avg_slp_loss']
                    avg_cp_loss = others['avg_cp_loss']
                if ifBreak:
                    break

def main():
    args = train_args()

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    print("creating data loader...")

    if datatype == "xsens":
        dataset = XsensDataset(dataset_dir=dataset_path_xsens, seq_len=seq_len)
    elif totalcapture:
        dataset = AMASS(path=dataset_path_amass,input_motion_length=seq_len,
                        train_dataset_repeat_times=1)
    else:
        assert datatype == "amass", "data type error."
        rot, sparse, trans, gposition, lposition = load_data_trans(
            dataset_path_amass,
            "train",
            input_motion_length=seq_len
        )
        dataset = AMASS_hmd(
            rot, sparse, trans, gposition, lposition,
            seq_len,
            1
        )

    dataloader = get_dataloader(
        dataset, "train", batch_size=args.batch_size, num_workers=args.num_workers
    )

    train_model(args, dataloader)

if __name__ == "__main__":
    main()