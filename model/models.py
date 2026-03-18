import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import *
from utils.skeleton import *

class RNN_with_init(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int, n_init=132, n_rnn_layer=2,dropout=0.2,
                 bidirectional=False):
        super().__init__()
        self.motion_feats = n_input
        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(n_init, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden * n_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden * n_rnn_layer, 2 * (2 if bidirectional else 1) * n_rnn_layer * n_hidden)
        )
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)#nn.Linear(n_hidden * 2, n_output)
        self.dropout = nn.Dropout(dropout)
        self.nd = n_rnn_layer * (2 if bidirectional else 1)
        self.nh = n_hidden

    def forward(self, x):
        x_init = x[...,self.motion_feats:][:,0,:]
        x = x[...,:self.motion_feats]  # 30
        res = self.init_net(x_init).view(-1, 2, self.nd, self.nh).permute(1, 2, 0, 3).contiguous()
        h = res[0]
        c = res[1]
        x = F.relu(self.linear1(self.dropout(x)))
        x = self.rnn(x, (h, c))[0]

        x = self.linear2(x)
        return x

    def online(self, x, h=None, c=None):
        if h is not None:
            x = x[..., :self.motion_feats]  # 30
            x = F.relu(self.linear1(self.dropout(x)))
            x, (h,c) = self.rnn(x, (h, c))

            x = self.linear2(x)
            return x, h, c
        else:
            x_init = x[..., self.motion_feats:][:, 0, :]
            x = x[..., :self.motion_feats]  # 30
            res = self.init_net(x_init).view(-1, 2, self.nd, self.nh).permute(1, 2, 0, 3).contiguous()
            h = res[0]
            c = res[1]
            x = F.relu(self.linear1(self.dropout(x)))
            x, (h,c) = self.rnn(x, (h, c))

            x = self.linear2(x)
            return x, h, c

class pose_RNN(nn.Module):
    def __init__(self,n_input=36,n_init=84,n_output_joints=14,n_hidden=512,bidirectional=False):
        super(pose_RNN, self).__init__()
        self.n_output_joints = n_output_joints
        self.n_pose_output = self.n_output_joints * 6

        n_output = self.n_pose_output

        self.model = RNN_with_init(n_input=n_input, n_output=n_output, n_init=n_init, n_hidden=512,
                                   bidirectional=bidirectional)

    def forward(self,x):
        pose = self.model(x)
        return pose

    def online(self, x, h=None, c=None):
        pose, h, c = self.model.online(x,h,c)
        return pose, h, c

class vel_RNN(nn.Module):
    def __init__(self,n_input=36,n_init=9,n_output=12,n_hidden=512,bidirectional=False):    # n_output: sparse_v+body_v
        super(vel_RNN, self).__init__()

        self.model = RNN_with_init(n_input=n_input, n_output=n_output, n_init=n_init, n_hidden=512,
                                   bidirectional=bidirectional)

    def forward(self,x):
        pose = self.model(x)
        return pose

    def online(self, x, h=None, c=None):
        pose, h, c = self.model.online(x,h,c)
        return pose, h, c

class LinearEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, dropout):
        super(LinearEncoder, self).__init__()

        self.InputSize = input_size
        self.OutputSize = output_size

        self.Dropout = dropout

        self.L1 = nn.Linear(input_size, hidden1_size)
        self.L2 = nn.Linear(hidden1_size, hidden2_size)
        self.L3 = nn.Linear(hidden2_size, output_size)

    def forward(self, z):
        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L1(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L2(z)
        z = F.elu(z)

        z = F.dropout(z, self.Dropout, training=self.training)
        z = self.L3(z)

        return z

class ThreeSIP_vqcm(nn.Module):
    def __init__(self, type="xsens", out_dim=84, #out_dim=132 smpl
        code_seq=30,#code_seq=10,#code_seq=30,
        latent_dim=1024, codebook_channels=128, codebook_dim=16, drop_out=0.25):
        super(ThreeSIP_vqcm, self).__init__()

        self.motion_dim = out_dim
        self.code_seq = code_seq
        self.C = codebook_channels
        self.D = codebook_dim
        self.codebook_size = codebook_dim*codebook_channels

        self.embedding = nn.Embedding(codebook_channels, codebook_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_channels, 1.0 / codebook_channels)

        self.motion_encoder_target = LinearEncoder(code_seq*out_dim, latent_dim, latent_dim, self.codebook_size,
                                                   drop_out)
        self.motion_decoder = LinearEncoder(self.codebook_size, latent_dim, latent_dim, code_seq*out_dim, 0.0)

    def forward(self, y):   # 36(imu)+9(initv)+132/84(initpose)
        batch = y.shape[0]
        seq_len = y.shape[1]
        padding = self.code_seq-seq_len%self.code_seq
        if padding < self.code_seq:
            pad = torch.zeros([batch,padding,y.shape[-1]],device=y.device,dtype=torch.float32)
            if y is not None:
                y = torch.cat([y,pad],dim=1)

        if y is not None:   # training
            y = y.reshape(batch,-1,self.code_seq*self.motion_dim)
            n_m = y.shape[1]
            ze_y = self.motion_encoder_target(y).reshape(batch,n_m,-1,1,self.D)   # b, n/m, hid/D, 1, D

            emb = self.embedding.weight.data
            emb_broadcast = emb.reshape(1,1,1,self.C,self.D)

            distance = torch.sum((emb_broadcast - ze_y)**2, -1) # b, n/m, hid/D, C
            nearest_neighbour = torch.argmin(distance, -1)   # b, n/m, hid/D

            zq_y = self.embedding(nearest_neighbour).reshape(batch,n_m,-1)    # b, n/m, hid/D, D -> b, n/m, hid
            ze_y = ze_y.reshape(batch, n_m, -1)
            decoder_input = ze_y + (zq_y - ze_y).detach()
            y = self.motion_decoder(decoder_input)

            y = y.reshape(batch,-1,self.motion_dim)[:,:seq_len,:]

            return y, ze_y, zq_y

    def sample_by_zef(self, ze_f):
        batch = ze_f.shape[0]
        n_m = ze_f.shape[1]
        emb = self.embedding.weight.data
        emb_broadcast = emb.reshape(1, 1, 1, self.C, self.D)
        ze_f = ze_f.reshape(batch,n_m,-1,1,self.D)
        distance = torch.sum((emb_broadcast - ze_f) ** 2, -1)  # b, n/m, hid/D, C
        nearest_neighbour = torch.argmin(distance, -1)  # b, n/m, hid/D

        zq_f = self.embedding(nearest_neighbour).reshape(batch, n_m, -1)  # b, n/m, hid/D, D -> b, n/m, hid
        ze_f = ze_f.reshape(batch, n_m, -1)
        decoder_input = ze_f + (zq_f - ze_f).detach()
        y = self.motion_decoder(decoder_input)

        y = y.reshape(batch, -1, self.motion_dim)

        return y
from typing import Optional
class ThreeSIP_divide(nn.Module):
    def __init__(self, type="xsens", out_dim=84, #out_dim=132 smpl
        code_seq=30,#code_seq=10,#code_seq=30,
        latent_dim=1024, codebook_channels=128, codebook_dim=16, drop_out=0.25):
        super(ThreeSIP_divide, self).__init__()

        self.code_seq = code_seq
        self.motion_dim = out_dim
        self.Trans = vel_RNN(n_input=36, n_init=9, n_output=12)   # 27+9+9
        self.Fine = pose_RNN(n_input=36+12, n_init=out_dim, n_output_joints=out_dim // 6)

        self.codebook_size = codebook_dim * codebook_channels
        self.motion_encoder_estimation = LinearEncoder(code_seq*out_dim, latent_dim, latent_dim, self.codebook_size,
                                                   drop_out)
        self.vqcm = ThreeSIP_vqcm(out_dim=out_dim, code_seq=code_seq, latent_dim=latent_dim,
                                  codebook_channels=codebook_channels, codebook_dim=codebook_dim, drop_out=drop_out)

    def forward(self, x, y:Optional[torch.Tensor]=None):#def forward(self, x, y:Optional[torch.Tensor]=None):   # 36(imu)+9(initv)+132/84(initpose)
        batch = x.shape[0]
        seq_len = x.shape[1]
        sparse = x[...,:36]
        init_v = x[...,36:45]
        init_pose = x[...,45:]

        vels = self.Trans(torch.cat([sparse, init_v], dim=-1))

        fine_pose = self.Fine(torch.cat([sparse, vels, init_pose], dim=-1))

        padding = self.code_seq-seq_len%self.code_seq
        if padding < self.code_seq:
            pad = torch.zeros([batch,padding,fine_pose.shape[-1]],device=fine_pose.device,dtype=torch.float32)
            fine_pose = torch.cat([fine_pose,pad],dim=1)
            pad_sparse = torch.zeros([batch,padding,sparse.shape[-1]],device=sparse.device,dtype=torch.float32)
            sparse = torch.cat([sparse,pad_sparse],dim=1)
            if y is not None:
                y = torch.cat([y,pad],dim=1)
        fine_pose = fine_pose.reshape(batch,-1,self.code_seq*self.motion_dim)   # b,n/m,132/84*15
        sparse = sparse.reshape(batch,-1,self.code_seq*sparse.shape[-1])    # b,n/m,36*15

        if y is not None:   # training
            n_m = fine_pose.shape[1]
            y, ze_y, zq_y = self.vqcm(y)
            ze_f = self.motion_encoder_estimation(fine_pose)
            fine_pose = fine_pose.reshape(batch, -1, self.motion_dim)[:, :seq_len, :]
            fine_pose_ = self.vqcm.sample_by_zef(ze_f)[:, :seq_len, :]
            #fine_pose_ = self.pose_fusion(torch.cat([fine_pose,fine_pose_],dim=-1))

            return torch.cat([fine_pose, vels, fine_pose_], dim=-1), ze_y, zq_y, ze_f
        else:   # evaluation
            fine_pose = fine_pose.reshape(batch, -1, self.motion_dim)[:, :seq_len, :]
            return torch.cat([fine_pose, vels], dim=-1), torch.tensor([1]), torch.tensor([1]), torch.tensor([1])

    def divide_online(self, x, slide_window=30):
        batch = x.shape[0]
        seq_len = x.shape[1]
        sparse = x[..., :36]
        init_v = x[..., 36:45]
        init_pose = x[..., 45:]

        vels, fine_pose, fine_pose_vq = [], [], []
        h1, c1, h2, c2, h3, c3, h4, c4 = None, None, None, None, None, None, None, None
        start = 0
        while start < seq_len:
            sparse_ = sparse[:,start:start+slide_window]
            init_pose_ = init_pose[:,start:start+slide_window]
            init_v_ = init_v[:,start:start+slide_window]
            seq_len_ = sparse_.shape[1]

            vels_, h2, c2 = self.Trans.online(torch.cat([sparse_, init_v_], dim=-1),h2,c2)

            fine_pose_, h3, c3 = self.Fine.online(torch.cat([sparse_, vels_, init_pose_], dim=-1), h3, c3)

            padding = self.code_seq - seq_len_ % self.code_seq
            if padding < self.code_seq:
                pad = torch.zeros([batch, padding, fine_pose_.shape[-1]], device=fine_pose_.device, dtype=torch.float32)
                fine_pose_ = torch.cat([fine_pose_, pad], dim=1)

            fine_pose_ = fine_pose_.reshape(batch, -1, self.code_seq * self.motion_dim)  # b,n/m,132/84*15

            n_m = fine_pose_.shape[1]
            ze_f = self.motion_encoder_estimation(fine_pose_)
            fine_pose_ = fine_pose_.reshape(batch, -1, self.motion_dim)[:, :seq_len_, :]
            fine_pose_vq_ = self.vqcm.sample_by_zef(ze_f)[:, :seq_len_, :]   # b,s,n

            fine_pose_vq.append(fine_pose_vq_)
            fine_pose.append(fine_pose_)
            vels.append(vels_)
            start += slide_window
        fine_pose_vq = torch.cat(fine_pose_vq, dim=1)
        fine_pose = torch.cat(fine_pose, dim=1)
        vels = torch.cat(vels, dim=1)
        return fine_pose, vels, fine_pose_vq

    def online(self, x, slide_window=30):
        batch = x.shape[0]
        seq_len = x.shape[1]
        sparse = x[..., :36]
        init_v = x[..., 36:45]
        init_pose = x[..., 45:]

        vels, fine_pose = [], []
        h1, c1, h2, c2, h3, c3, h4, c4 = None, None, None, None, None, None, None, None
        start, cnt = 0, 0
        while start < seq_len:
            sparse_ = sparse[:,start:start+slide_window]
            init_pose_ = init_pose[:,start:start+slide_window]
            init_v_ = init_v[:,start:start+slide_window]
            seq_len_ = sparse_.shape[1]

            vels_, h2, c2 = self.Trans.online(torch.cat([sparse_, init_v_], dim=-1),h2,c2)

            fine_pose_, h3, c3 = self.Fine.online(torch.cat([sparse_, vels_, init_pose_], dim=-1), h3, c3)

            padding = self.code_seq - seq_len_ % self.code_seq
            if padding < self.code_seq:
                pad = torch.zeros([batch, padding, fine_pose_.shape[-1]], device=fine_pose_.device, dtype=torch.float32)
                fine_pose_ = torch.cat([fine_pose_, pad], dim=1)

            fine_pose_ = fine_pose_.reshape(batch, -1, self.code_seq * self.motion_dim)  # b,n/m,132/84*15

            n_m = fine_pose_.shape[1]
            ze_f = self.motion_encoder_estimation(fine_pose_)
            fine_pose_ = fine_pose_.reshape(batch, -1, self.motion_dim)[:, :seq_len_, :]
            fine_pose_vq = self.vqcm.sample_by_zef(ze_f)[:, :seq_len_, :]   # b,s,n

            c = 0.44
            fine_out = fine_pose_ * c + fine_pose_vq * (1-c)
            fine_pose.append(fine_out)
            vels.append(vels_)
            start += slide_window
            cnt += 1    # count fps
        fine_pose = torch.cat(fine_pose, dim=1)
        vels = torch.cat(vels, dim=1)
        return torch.cat([fine_pose, vels], dim=-1), cnt, torch.tensor([1]), torch.tensor([1])

class ThreeSIP_divide_wsearch(nn.Module):
    def __init__(self):
        super(ThreeSIP_divide_wsearch, self).__init__()

        self.w = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float32), requires_grad=True)

    def forward(self, fine_pose, fine_pose_vq):
        fusion = fine_pose * self.w + fine_pose_vq * (torch.tensor([1.0], dtype=torch.float32, device=fine_pose.device) - self.w)
        return fusion

class ThreeSIP_divide_wwrapped(nn.Module):
    def __init__(self, type="xsens", out_dim=84, code_seq=30):
        super(ThreeSIP_divide_wwrapped, self).__init__()

        self.ThreeSIP = ThreeSIP_divide(type=type, out_dim=out_dim, code_seq=code_seq,latent_dim=1024, codebook_channels=128, codebook_dim=16, drop_out=0.25)
        self.w = ThreeSIP_divide_wsearch()

    def forward(self, x):
        fine_pose, vels, fine_pose_vq = self.ThreeSIP.divide_online(x)
        fusion = self.w(fine_pose, fine_pose_vq)
        return torch.cat([fusion, vels], dim=-1), x.shape[1]/30.0, torch.tensor([1]), torch.tensor([1])

    def online(self, x):
        fine_pose, vels, fine_pose_vq = self.ThreeSIP.divide_online(x)
        fusion = self.w(fine_pose, fine_pose_vq)
        return torch.cat([fusion, vels], dim=-1), x.shape[1]/30.0, torch.tensor([1]), torch.tensor([1])