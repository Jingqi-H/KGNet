#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jingqi Huang
# @Time : 2021/12/4 8:39
import torch
import torch.nn as nn
import torch.nn.functional as F


criterion = nn.CrossEntropyLoss().cuda()
loss = criterion()