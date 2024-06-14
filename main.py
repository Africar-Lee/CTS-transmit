import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

from script import dataloader, utility, earlystopping
from model import models

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from torch_geometric.data import Data

# @TODO
# prepare data

# prepare model parameters
num_features = 4
num_classes = 1
model = models.GCN(num_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# start training
for epoch in range(200):
    optimizer.zero_grad()  # 清除梯度
    out = model(data)  # 前向传播
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# make prediction
model.eval()
out = model(data)
_, pred = out.max(dim=1)
