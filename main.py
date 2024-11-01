import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models
from models import BasicBlock3D
from tool import data_split, Load_data, shapley, combination
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import StepLR

class GLA_module(nn.Module):
    def __init__(self,
                 heads=6,
                 channel=512,
                 dropout_rate=0.0):
        super().__init__()

        self.num_heads = heads
        self.head_size = int(channel / self.num_heads)
        self.all_head_size = self.num_heads * self.head_size

        self.query = nn.Linear(channel, self.all_head_size)
        self.key = nn.Linear(channel, self.all_head_size)
        self.value = nn.Linear(channel, self.all_head_size)

        self.out = nn.Linear(channel, channel)
        self.dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, local_x, global_x):
        q_matrix = self.query(local_x)
        k_matrix = self.key(global_x)
        v_matrix = self.value(global_x)

        q_layer = self.transpose(q_matrix)
        k_layer = self.transpose(k_matrix)
        v_layer = self.transpose(v_matrix)

        similarity = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        similarity = similarity / math.sqrt(self.head_size)
        attention = self.softmax(similarity)

        attention = self.dropout(attention)
        fusion_layer = torch.matmul(attention, v_layer)

        fusion_layer = fusion_layer.permute(0, 2, 1, 3).contiguous()
        new_fusion_layer_shape = fusion_layer.size()[:-2] + (self.all_head_size,)
        fusion_layer = fusion_layer.view(*new_fusion_layer_shape)

        output = self.out(fusion_layer)
        output = self.dropout(output)

        return output

class convBlock(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(outplace)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class Feedforward(nn.Module):
    def __init__(self, inplace, outplace):
        super().__init__()

        self.conv1 = convBlock(inplace, outplace, kernel_size=1, padding=0)
        self.conv2 = convBlock(outplace, outplace, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ds_FCRN(nn.Module):
    def __init__(self, inplace,
                 patch_size=64,
                 nblock=4,
                 dropout_rate=0.2,
                 network='FCRN',
                 start_age=45,
                 end_age=82):
        """
            patch_size: the patch size of local pathway input, default is 64
            nblock: the number of blocks for the 3D glt module, default is 4
            dropout_rate: dropout rate, default is 0.2
            network: feature extraction network, default is FCRN. VGG8, VGG16, SFCN, ResNet18 is also available
            start_age: minimum age of subject, default is 45, please use your real data instead of the default values
            end_age: maximum age of subject, default is 82, please use your real data instead of the default values
        """

        super().__init__()
        self.age = torch.linspace(start_age, end_age, end_age-start_age+1).to("cuda:0")
        self.patch_size = patch_size
        self.step = int(patch_size // 2)
        self.nblock = nblock

        if network == 'FCRN':
            self.global_feature = models.FCRN(inplace)
            self.local_feature = models.FCRN(inplace)
            channel = 256
        elif network == 'VGG8':
            self.global_feature = models.VGG8(inplace)
            self.local_feature = models.VGG8(inplace)
            channel = 256
        elif network == 'VGG16':
            self.global_feature = models.VGG16(inplace)
            self.local_feature = models.VGG16(inplace)
            channel = 256
        elif network == 'SFCN':
            self.global_feature = models.SFCN(inplace)
            self.local_feature = models.SFCN(inplace)
            channel = 256
        elif network == 'Res18':
            self.global_feature = models.ResNet3D(BasicBlock3D, [2, 2, 2, 2])
            self.local_feature = models.ResNet3D(BasicBlock3D, [2, 2, 2, 2])
            channel = 256
        else:
            raise ValueError('% model does not supported!' % network)

        self.attention_list = nn.ModuleList()
        self.feedforward_list = nn.ModuleList()

        for n in range(nblock):
            attention_module = GLA_module(
                heads=8,
                channel=channel,
                dropout_rate=dropout_rate)
            self.attention_list.append(attention_module)

            feedforward_module = Feedforward(inplace=channel * 2,
                              outplace=channel)
            self.feedforward_list.append(feedforward_module)

        self.avg = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(in_channels=channel, out_channels=38, kernel_size=1, padding=0, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        _, _, H, W, D = input.size()
        output_list = []
        global_x = self.global_feature(input)
        global_output = F.softmax((torch.flatten(self.conv(self.dropout((self.avg(global_x)))), 1)), dim=1)
        global_output = torch.sum(global_output * self.age, axis=1)
        output_list = [global_output]

        B2, C2, H2, W2, D2 = global_x.size()
        global_xt = global_x.view(B2, C2, H2 * W2 * D2)
        global_xt = global_xt.permute(0, 2, 1)

        for y in range(0, H - self.step, self.step):
            for x in range(0, W - self.step, self.step):
                for z in range(0, D - self.step, self.step):
                    if y + self.patch_size < H and x + self.patch_size < W and z + self.patch_size < D:
                        local_input = input[:, :, y:y + self.patch_size, x:x + self.patch_size, z:z + self.patch_size]
                    elif y + self.patch_size > H and x + self.patch_size < W and z + self.patch_size < D:
                        local_input = input[:, :, H - self.patch_size:H, x:x + self.patch_size, z:z + self.patch_size]
                    elif y + self.patch_size < H and x + self.patch_size > W and z + self.patch_size < D:
                        local_input = input[:, :, y:y + self.patch_size, W - self.patch_size:W, z:z + self.patch_size]
                    elif y + self.patch_size < H and x + self.patch_size < W and z + self.patch_size > D:
                        local_input = input[:, :, y:y + self.patch_size, x:x + self.patch_size, D - self.patch_size:D]
                    local_x = self.local_feature(local_input)

                    for n in range(self.nblock):
                        B1, C1, H1, W1, D1 = local_x.size()
                        local_xt = local_x.view(B1, C1, H1 * W1 * D1)
                        local_xt = local_xt.permute(0, 2, 1)

                        fusion_x = self.attention_list[n](local_xt, global_xt)
                        fusion_x = fusion_x.permute(0, 2, 1)
                        fusion_x = fusion_x.view(B1, C1, H1, W1, D1)
                        fusion_x = torch.cat([fusion_x, local_x], 1)

                        fusion_x = self.feedforward_list[n](fusion_x)
                        local_x = local_x + fusion_x

                    fusion_outout = F.softmax((torch.flatten(self.conv(self.dropout((self.avg(local_x)))), 1)), dim=1)

                    fusion_outout = torch.sum(fusion_outout * self.age, axis=1)
                    output_list.append(fusion_outout)
        return output_list
