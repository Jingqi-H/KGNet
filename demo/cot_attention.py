#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jingqi Huang
# @Time : 2021/8/2 9:48

import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F


# https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/attention/CoTAttention.py
# https://mp.weixin.qq.com/s/30lYjPZ9ZJAY9td6oZR0OQ
# 解释的很清楚：https://zhuanlan.zhihu.com/p/405828451

class CoTAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            # theta: 有relu的卷积
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            # delta：没有激活函数的卷积
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        # 1 根据论文图示，k经过 kxk 卷积；q是x本身；v经过1x1卷积
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        # 2 获得权值
        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        # 3 更新v
        # print(att.shape, v.shape)  # torch.Size([50, 512, 49]) torch.Size([50, 512, 49])
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class AttAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            # theta: 有relu的卷积
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            # delta：没有激活函数的卷积
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, feature_map, attribution_map):
        bs, c, h, w = attribution_map.shape
        # 1 根据论文图示，k经过 kxk 卷积；q是x本身；v经过1x1卷积
        k1 = self.key_embed(attribution_map)  # bs,c,h,w
        v = self.value_embed(feature_map).view(bs, c, -1)  # bs,c,h,w

        # 2 获得权值
        y = torch.cat([k1, attribution_map], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        # 3 更新v
        # print(att.shape, v.shape)  # torch.Size([50, 512, 49]) torch.Size([50, 512, 49])
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        # fusion就是 +，这里的fusion也可以是其他形式
        return k1 + k2


if __name__ == '__main__':
    img_feat = torch.randn(50, 512, 7, 7)
    att_feat = torch.randn(50, 512, 7, 7)
    cot = AttAttention(dim=512, kernel_size=3)
    output = cot(img_feat, att_feat)
    print(output.shape)
