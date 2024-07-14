import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pesq import pesq
from joblib import Parallel, delayed
from DPIH import DPIH
import json
from env import AttrDict, build_env
from ptflops.flops_counter import get_model_complexity_info

def mag_pha_stft(y):
    mag = torch.sqrt((y[:, 0, :, :] ** 2 + y[:, 1, :, :] ** 2) + 1e-6)  # B,T,F
    mag = mag.permute(0,2,1)  # B,F,T
    pha = torch.angle(torch.complex(y[:, 0, :, :], y[:, 1, :, :]))  # B,T,F
    pha = pha.permute(0,2,1)  # B,F,T
    # Magnitude Compression
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)  # [B,F,T,C]
    return mag, pha, com

def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size() #[B,C,H,W]
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group, #[B,4,C/4,H,W]
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()  #[B,C/4,4,H,W]
    # reshape into orignal
    x = x.view(batch_size, channels, height, width) #[B,C,H,W]
    return x

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))


class LearnableSigmoid_1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))  #α. (in_features, 1) For each feature of the data, having a separate slope parameter
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x) #First scale the input using a learnable slope paramet


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True,
                 fre: int = 201,
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)   #torch_gn=Ture：nn.GroupNorm    torch_gn=Flase:GroupBatchnorm2d
        self.gate_treshold = gate_treshold
        self.lsigmoid = LearnableSigmoid_2d(fre, beta=2.0) # v1+fe 1.23


    def forward(self, x):
        gn_x = self.gn(x)   # x * self.weight + self.bias
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        x1 = gn_x * w_gamma
        reweigts = self.lsigmoid(x1.permute(0,1,3,2)).permute(0,1,3,2)  #W weight
        f_reweigts = torch.flip(reweigts, [1])  # F operation
        x_1 = reweigts * x  #X1
        x_2 = f_reweigts * x  #X2
        y2 = shuffle_channels(x_2, 4) #S operation
        y = x_1 + y2

        return y

class multiple_branch(nn.Module):
    def __init__(self, h, K=3, i=0):
        super(multiple_branch, self).__init__()

        self.padding = K // 2
        self.E_1x1 = nn.Conv2d(in_channels=h.dense_channel, out_channels=h.dense_channel, kernel_size=1)
        self.E_kxk = nn.Conv2d(in_channels=h.dense_channel, out_channels=h.dense_channel, kernel_size=K, padding=self.padding, groups=h.dense_channel)
        self.E_1x1_kxk = nn.Sequential(
            nn.Conv2d(in_channels=h.dense_channel, out_channels=h.dense_channel, kernel_size=1),
            nn.Conv2d(in_channels=h.dense_channel, out_channels=h.dense_channel, kernel_size=K, padding=self.padding, groups=h.dense_channel)
        )
        self.E_1x1_avg = nn.Sequential(
            nn.Conv2d(in_channels=h.dense_channel, out_channels=h.dense_channel, kernel_size=1),
            nn.MaxPool2d(kernel_size=K, stride=1, padding=self.padding)
        )
        self.Norm = nn.Sequential(
            SwitchNorm2d(h.dense_channel),
            nn.PReLU()
        )


    def forward(self, input):
        out = self.E_1x1(input)
        out += self.E_1x1_kxk(input)
        out += self.E_1x1_avg(input)
        out += self.E_kxk(input)
        out = self.Norm(out)

        return out


class multiple_branch4(nn.Module):
    def __init__(self, h, K=3, i=0):
        super(multiple_branch4, self).__init__()
        self.h = h.dense_channel // 4
        self.padding = K // 2
        self.E_1x1 = nn.Conv2d(in_channels=self.h, out_channels=self.h, kernel_size=1)
        self.E_kxk = nn.Conv2d(in_channels=self.h, out_channels=self.h, kernel_size=K,
                               padding=self.padding, groups=self.h)
        self.E_1x1_kxk = nn.Sequential(
            nn.Conv2d(in_channels=self.h, out_channels=self.h, kernel_size=1),
            nn.Conv2d(in_channels=self.h, out_channels=self.h, kernel_size=K, padding=self.padding,
                      groups=self.h)
        )
        self.E_1x1_avg = nn.Sequential(
            nn.Conv2d(in_channels=self.h, out_channels=self.h, kernel_size=1),
            nn.MaxPool2d(kernel_size=K, stride=1, padding=self.padding)
        )
        self.Norm = nn.Sequential(
            SwitchNorm2d(h.dense_channel),
            nn.PReLU()
        )

    def forward(self, input):
        input1, input2, input3, input4 = torch.chunk(input, 4, dim=1)
        out1 = self.E_1x1(input1)
        out2 = self.E_1x1_kxk(input2)
        out3 = self.E_1x1_avg(input3)
        out4 = self.E_kxk(input4)
        out = self.Norm(torch.cat((out1, out2, out3, out4), dim=1))

        return out

class SelectFusion(nn.Module):
    def __init__(self, N):
        super(SelectFusion, self).__init__()
        # Hyper-parameter
        self.N = N
        self.linear3 = nn.Conv2d(2 * N, N, kernel_size=(1, 1), bias=False)

    def forward(self, stft_feature, conv_feature):
        fusion_feature = self.linear3(torch.cat([stft_feature, conv_feature], dim=1))
        ratio_mask1 = torch.sigmoid(fusion_feature)
        ratio_mask2 = 1 - ratio_mask1
        conv_out = conv_feature * ratio_mask1
        stft_out = stft_feature * ratio_mask2
        fusion_out = conv_out + stft_out
        out = F.relu(stft_feature + conv_feature + fusion_out)

        return out

class SelectFusion1(nn.Module):
    def __init__(self, N):
        super(SelectFusion1, self).__init__()
        # Hyper-parameter
        self.N = N
        self.linear1 = nn.Conv2d(N, N, kernel_size=(1, 1), bias=False)
        self.linear2 = nn.Conv2d(N, N, kernel_size=(1, 1), bias=False)
        self.linear3 = nn.Conv2d(2 * N, N, kernel_size=(1, 1), bias=False)

    def forward(self, x, y):
        x_mask = torch.sigmoid(self.linear1(x))
        y_mask = torch.sigmoid(self.linear2(y))
        fusion_feature = self.linear3(torch.cat([x, y], dim=1))
        y_out = fusion_feature * y_mask
        x_out = fusion_feature * x_mask
        out = F.relu(x + y + y_out + x_out)

        return out

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            HardSwish(inplace=True),
        )

    def forward(self, x):
        b, c, t, f = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x

class multibranch_Block(nn.Module):
    def __init__(self, h, depth=4):
        super(multibranch_Block, self).__init__()

        self.depth = depth
        self.uplay = nn.ModuleList([])
        for i in range(self.depth):
            dense_conv = nn.Sequential(
                multiple_branch(h, K=3, i=i)
            )
            self.uplay.append(dense_conv)
        self.downlay = nn.ModuleList([])
        for i in range(self.depth):
            dense_conv = nn.Sequential(
                multiple_branch(h, K=5, i=i)
            )
            self.downlay.append(dense_conv)

        self.fsfb = SelectFusion(h.dense_channel)
        self.fuse = nn.Conv2d(5 * h.dense_channel, h.dense_channel, 1, bias=False)


    def forward(self, x):
        skip = [x]
        for i in range(self.depth):
            x = self.fsfb(self.uplay[i](x.contiguous()), self.downlay[i](x.contiguous()))
            skip.append(x)

        out = self.fuse(torch.cat(skip, dim=1))


        return out


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            SwitchNorm2d(h.dense_channel),
            nn.PReLU(h.dense_channel))

        self.dense_block = multibranch_Block(h, depth=4)  # [b, h.dense_channel, ndim_time, h.n_fft//2+1]
        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=201)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            SwitchNorm2d(h.dense_channel),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 24, T, F]
        x = self.dense_block(x)  # [b, 64, T, F]
        x = self.SRU(x)

        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = multibranch_Block(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            SwitchNorm2d(out_channel),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(h.n_fft // 2 + 1, beta=h.beta)

        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=100)


    def forward(self, x):
        x = self.dense_block(x)
        x = self.SRU(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x

class Complex_Decoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(Complex_Decoder, self).__init__()
        self.dense_block = multibranch_Block(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            SwitchNorm2d(out_channel),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        return x

class NoisyDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(NoisyDecoder, self).__init__()
        self.dense_block = multibranch_Block(h, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            SwitchNorm2d(out_channel),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=100)


    def forward(self, x):
        x = self.dense_block(x)
        x = self.SRU(x)
        x = self.mask_conv(x)
        return x

class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = multibranch_Block(h, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2)),
            SwitchNorm2d(h.dense_channel),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 1))

        self.SRU = SRU(h.dense_channel, group_num=4, gate_treshold=0.5, fre=100)


    def forward(self, x):
        x = self.dense_block(x)
        x = self.SRU(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

class MN_Net(nn.Module):
    def __init__(self, h, num_tscblocks=4):
        super(MN_Net, self).__init__()
        self.h = h
        self.num_tscblocks = num_tscblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.DPIH = DPIH(input_size=64, output_size=64, num_layers=4)
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)
        self.noisy_decoder = NoisyDecoder(h, out_channel=2)

    def forward(self, noisy_mag, noisy_pha):
        noisy_mag = noisy_mag.unsqueeze(-1).permute(0, 3, 2, 1)
        noisy_pha = noisy_pha.unsqueeze(-1).permute(0, 3, 2, 1)
        x_mix = torch.cat((noisy_mag, noisy_pha), dim=1)
        x = self.dense_encoder(x_mix)

        x = self.DPIH(x)
        denoised_mag = (noisy_mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        denoised_com = torch.stack((denoised_mag * torch.cos(denoised_pha),
                                    denoised_mag * torch.sin(denoised_pha)), dim=-1)

        noisy = self.noisy_decoder(x)

        return denoised_mag, denoised_pha, denoised_com, noisy.permute(0, 3, 2, 1)


def metric_loss(metric_ref, metrics_gen):
    loss = 0
    for metric_gen in metrics_gen:
        metric_loss = F.mse_loss(metric_ref, metric_gen.flatten())
        loss += metric_loss

    return loss


def phase_losses(phase_r, phase_g, h):
    dim_freq = h.n_fft // 2 + 1
    dim_time = phase_r.size(-1)

    gd_matrix = (torch.triu(torch.ones(dim_freq, dim_freq), diagonal=1) - torch.triu(torch.ones(dim_freq, dim_freq),
                                                                                     diagonal=2) - torch.eye(
        dim_freq)).to(phase_g.device)
    gd_r = torch.matmul(phase_r.permute(0, 2, 1), gd_matrix)
    gd_g = torch.matmul(phase_g.permute(0, 2, 1), gd_matrix)

    iaf_matrix = (torch.triu(torch.ones(dim_time, dim_time), diagonal=1) - torch.triu(torch.ones(dim_time, dim_time),
                                                                                      diagonal=2) - torch.eye(
        dim_time)).to(phase_g.device)
    iaf_r = torch.matmul(phase_r, iaf_matrix)
    iaf_g = torch.matmul(phase_g, iaf_matrix)

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(gd_r - gd_g))
    iaf_loss = torch.mean(anti_wrapping_function(iaf_r - iaf_g))

    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def pesq_score(utts_r, utts_g, h):
    pesq_score = Parallel(n_jobs=30)(delayed(eval_pesq)(
        utts_r[i].squeeze().cpu().numpy(),
        utts_g[i].squeeze().cpu().numpy(),
        h.sampling_rate)
                                     for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        # error can happen due to silent period
        pesq_score = -1

    return pesq_score


if __name__ == '__main__':
    x = torch.randn(1, 2, 201, 161)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name')
    parser.add_argument('--input_clean_wavs_dir')
    parser.add_argument('--input_noisy_wavs_dir')
    parser.add_argument('--input_training_file',)
    parser.add_argument('--input_validation_file')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--run_logs')
    parser.add_argument('--config')
    parser.add_argument('--training_epochs')
    parser.add_argument('--stdout_interval')
    parser.add_argument('--checkpoint_interval')
    parser.add_argument('--summary_interval')
    parser.add_argument('--validation_interval')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    model = MN_Net(h)
    out2 = model(x)

    # test Calculate the force
    flops2, params2 = get_model_complexity_info(model, (2, 201, 161), print_per_layer_stat=True)
    print("%s %s" % (flops2, params2))