"""
使用 RAFT 光流进行视频动态质量筛选（自包含，无需导入 RAFT 包）
"""
import argparse
import csv
import fcntl
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_FILTER_ROOT = str(Path(__file__).resolve().parents[1]) 
if _FILTER_ROOT not in sys.path:
    sys.path.insert(0, _FILTER_ROOT)

from filter.filter_utils import unique_output_name, worker_syspath_setup


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# RAFT 模型代码（内联）
# ============================================================

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


# --- extractor ---

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)
        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x


# --- corr ---

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


# --- update ---

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


# --- RAFT ---

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # if self.args.alternate_corr:
        #     corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        # else:
        #     corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


# ─── RAFT 加载 ────────────────────────────────────────────────────────────────

def load_raft_model(model_path, small=False):
    """加载 RAFT 模型"""
    args = argparse.Namespace(
        small=small,
        mixed_precision=False,
        alternate_corr=False,
    )
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.module
    model.to(DEVICE)
    model.eval()
    return model


def frame_to_tensor(frame):
    """BGR frame (H,W,3) -> tensor (1,3,H,W)"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


@torch.no_grad()
def compute_flow_magnitude(model, frame1, frame2, iters=20):
    """计算两帧之间的光流幅度均值"""
    image1 = frame_to_tensor(frame1)
    image2 = frame_to_tensor(frame2)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    _, flow_up = model(image1, image2, iters=iters, test_mode=True)
    flow = padder.unpad(flow_up[0]).cpu().numpy()  # (2, H, W)
    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    return float(np.mean(mag))


def compute_motion_score(model, video_path, sampling_fps=2, iters=20, max_size=512):
    """
    使用 RAFT 光流计算视频运动分数。

    运动分数参考（RAFT 光流，像素位移均值）：
        - < 0.5:  几乎静止
        - 0.5 ~ 2.0: 低动态（talking head、慢移动）
        - 2.0 ~ 6.0: 中等动态（正常视频）
        - 6.0 ~ 15.0: 高动态（运动场景、快速镜头）
        - > 15.0: 极高动态（剧烈抖动、场景频繁切换）

    Returns:
        dict: {"motion_score": float, "frame_scores": list} 或 None
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return None

    frame_interval = max(1, int(fps / min(max(sampling_fps, 1), fps)))
    frame_scores = []
    prev_frame = None
    frame_count = -1

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        if frame_count % frame_interval != 0:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if prev_frame is None:
            prev_frame = frame
            continue

        try:
            score = compute_flow_magnitude(model, prev_frame, frame, iters=iters)
            frame_scores.append(score)
        except Exception as e:
            print(f"  光流计算异常: {e}")

        prev_frame = frame

    cap.release()
    if not frame_scores:
        return None

    return {
        "motion_score": round(float(np.mean(frame_scores)), 4),
        "frame_scores": frame_scores,
    }


# ─── CSV Pipeline ────────────────────────────────────────────────────────────

MOTION_FIELDS = ["motion_reject_reason", "motion_score"]


def process_segment(args, model, item, qualified_dir):
    """处理单个视频片段：计算运动分数并判定。保留输入 CSV 全部列，追加运动过滤结果。"""
    row = dict(item)
    source_video = row.get(args.video_path_column, "")
    # 输入行是源视频路径；输出 video_path 在我们这里也表示"当前处理视频"的最终落点。
    # 不再保留重复的 source_video_path 字段。
    row["video_path"] = ""
    row["motion_reject_reason"] = ""
    row["motion_score"] = ""

    seg_path = Path(source_video)
    if not seg_path.is_file():
        row["motion_reject_reason"] = "file_not_found"
        return row

    # 输出文件名加源视频 hash 短码，避免不同源同名时冲突
    out_name = unique_output_name(seg_path, suffix=".mp4")
    out_path = Path(qualified_dir) / out_name

    # 已存在则跳过（断点续跑 / 重新跑）
    if out_path.exists():
        print(f"  跳过(已存在): {out_name}")
        row["video_path"] = str(out_path)
        return row

    result = compute_motion_score(
        model, str(seg_path),
        sampling_fps=args.sampling_fps,
        iters=args.raft_iters,
        max_size=args.max_size,
    )
    if result is None:
        row["motion_reject_reason"] = "compute_failed"
        return row

    score = result["motion_score"]
    row["motion_score"] = score

    if score < args.min_motion:
        row["motion_reject_reason"] = f"too_static_{score}"
        print(f"  不通过: {seg_path.name} | score={score:.4f} < {args.min_motion}")
        return row
    if score > args.max_motion:
        row["motion_reject_reason"] = f"too_dynamic_{score}"
        print(f"  不通过: {seg_path.name} | score={score:.4f} > {args.max_motion}")
        return row

    if args.copy_mode == "symlink":
        out_path.symlink_to(seg_path.resolve())
    elif args.copy_mode == "copy":
        shutil.copy2(str(seg_path), str(out_path))
    else:
        out_path = seg_path
    row["video_path"] = str(out_path)
    print(f"  通过: {seg_path.name} | score={score:.4f}")
    return row


# ─── Worker ─────────────────────────────────────────────────────────────────
class Worker:
    def __init__(self):
        worker_syspath_setup()  # Ray worker 端把 data_process/ 加进 sys.path
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qualified_dir = self.output_dir / "qualified_videos"
        self.qualified_dir.mkdir(parents=True, exist_ok=True)
        self.all_csv = self.output_dir / "3_motion_filter_all.csv"
        self.qual_csv = self.output_dir / "3_motion_filter_qualified.csv"
        with open(args.input_csv, encoding="utf-8") as f:
            input_fields = csv.DictReader(f).fieldnames or []
        # 不再额外加 source_video_path；video_path 表示当前处理视频的路径
        self.fieldnames = list(input_fields) + [c for c in ["video_path"] + MOTION_FIELDS if c not in input_fields]
        for p in (self.all_csv, self.qual_csv):
            if not p.exists():
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        # 断点续跑：加载已处理的 video_path（输出路径就是当前处理视频的最终落点）
        self.processed = set()
        if self.all_csv.exists():
            with open(self.all_csv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    vp = r.get("video_path", "")
                    if vp:
                        self.processed.add(vp)
        print(f"加载 RAFT 模型: {args.raft_model}")
        self.model = load_raft_model(args.raft_model, small=args.raft_small)
        print(f"RAFT 模型加载完成. 已处理 {len(self.processed)} 条。")

    def _append_csv(self, filepath, row):
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore").writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, item):
        seg_path = item.get(args.video_path_column, "")
        if not seg_path:
            row = dict(item)
            row.update({k: "" for k in MOTION_FIELDS})
            return row
        if seg_path in self.processed:
            row = dict(item)
            row.update({k: "" for k in MOTION_FIELDS})
            return row
        row = process_segment(args, self.model, item, str(self.qualified_dir))
        self._append_csv(self.all_csv, row)
        if not row.get("motion_reject_reason"):
            self._append_csv(self.qual_csv, row)
        self.processed.add(seg_path)
        return row

# ─── CLI ─────────────────────────────────────────────────────────────────
RAFT_CKPT_DEFAULT = str(Path(__file__).resolve().parent.parent / "third_part" / "raft-things.pth")
parser = argparse.ArgumentParser(description="RAFT 光流运动评分过滤")
parser.add_argument("--input_csv", default="/mnt/workspace/lihaoran/data_process/test/result/0617/filter_shotcut/data_5000_all/2_cutshot_qualified.csv")
parser.add_argument("--output_dir", default="/mnt/workspace/lihaoran/data_process/test/result/0617/motion_filter_raft/data_5000_all/")
parser.add_argument("--video_path_column", default="video_path")
parser.add_argument("--raft_model", default=RAFT_CKPT_DEFAULT)
parser.add_argument("--raft_small", action="store_true")
parser.add_argument("--raft_iters", type=int, default=20)
parser.add_argument("--min_motion", type=float, default=10)
parser.add_argument("--max_motion", type=float, default=200)
parser.add_argument("--sampling_fps", type=int, default=2)
parser.add_argument("--max_size", type=int, default=512)
parser.add_argument("--copy_mode", choices=["symlink", "copy", "none"], default="symlink")

# Ray
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_num_gpus", type=float, default=0.2)
parser.add_argument("--ray_num_cpus", type=float, default=0)
parser.add_argument("--concurrency", type=int, default=10)
args = parser.parse_args()


if __name__ == "__main__":

    # ── 去重：按 video_path_column 去重，避免重复处理 ──
    import pandas as pd
    df = pd.read_csv(args.input_csv, encoding="utf-8")
    before_cnt = len(df)
    df = df.drop_duplicates(subset=[args.video_path_column], keep="first")
    after_cnt = len(df)
    if before_cnt != after_cnt:
        print(f"去重: {before_cnt} → {after_cnt} (去除 {before_cnt - after_cnt} 条重复)")
    deduped_csv = args.input_csv.replace(".csv", "_deduped.csv")
    df.to_csv(deduped_csv, index=False)

    if not args.ray:
        # ── 本地模式 ──
        samples = list(csv.DictReader(open(deduped_csv, encoding="utf-8")))
        total = len(samples)
        print(f"读取 {total} 条记录 from {deduped_csv}")
        worker = Worker()
        for i, item in enumerate(samples, 1):
            print(f"[{i}/{total}] {item.get(args.video_path_column, '')}")
            worker(item)
        print("完成")

    else:
        # ── Ray 分布式模式 ──
        import ray
        ray.init(address="auto",
                 runtime_env={"env_vars": 
                                {"PYTHONPATH": _FILTER_ROOT,
                                 "DATA_PROCESS_ROOT": _FILTER_ROOT},
                            }
                )

        from ray.data import ActorPoolStrategy
        samples = ray.data.read_csv(deduped_csv).materialize()
        target_parallelism = min(args.concurrency, samples.count())
        if samples.num_blocks() < target_parallelism:
            samples = samples.repartition(target_parallelism)
        predictions = samples.map(
            Worker,
            num_gpus=args.ray_num_gpus,
            num_cpus=max(args.ray_num_cpus, 1),
            compute=ActorPoolStrategy(size=target_parallelism),
        )
        predictions.take_all()
        print("Ray 完成")
