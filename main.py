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

import matplotlib.pyplot as plt
import seaborn as sns
import transbigdata as tbd

import networkx as nx

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from torch_geometric.data import Data

# @TODO
# prepare data
line,stop = tbd.getbusdata('北京',['1号线', '2号线', '5号线', '6号线', '7号线', '8号线', '9&房山线', '10号线',
       '13号线', '15号线', '昌平线', '首都机场线'])
line = line[line['line']!='大兴机场大巴首都机场线'].reset_index(drop=True)
stop = stop[stop['line']!='大兴机场大巴首都机场线'].reset_index(drop=True)
line = line[line['line']!='首都机场大巴昌平线'].reset_index(drop=True)
stop = stop[stop['line']!='首都机场大巴昌平线'].reset_index(drop=True)

line_data = {'线路名': [], '站点名': [], '顺序': []}
for i in range(len(stop)):
    线路名 = line_data['线路名']
    线路名.append(stop.loc[i, 'line'])
    站点名 = line_data['站点名']
    站点名.append(stop.loc[i, 'stationnames'])
    顺序 = line_data['顺序']
    顺序.append(stop.loc[i, 'id'])
    line_data.update({'线路名':线路名, '站点名':站点名, '顺序':顺序})
# 示例数据
# line_data = {
#     '线路名': ['1号线', '1号线', '1号线', '2号线', '2号线'],
#     '站点名': ['站点A', '站点B', '站点C', '站点B', '站点D'],
#     '顺序': [1, 2, 3, 1, 2]
# }


stop_data = {'站点名':list(stop['stationnames'].unique())}
# 示例数据
# stop_data = {
#     '站点名': ['站点A', '站点B', '站点C', '站点D'],
#     '其他信息': ['infoA', 'infoB', 'infoC', 'infoD']
# }

line_for_adj = pd.DataFrame(line_data)
stop_for_adj = pd.DataFrame(stop_data)

# 构建地铁网络图
G = nx.Graph()

# 添加节点
for station_for_adj in stop_for_adj['站点名']:
    G.add_node(station_for_adj)

# 添加边（根据站点顺序）
for line_name, group in line_for_adj.groupby('线路名'):
    sorted_stations = group.sort_values('顺序')['站点名'].tolist()
    for i in range(len(sorted_stations) - 1):
        G.add_edge(sorted_stations[i], sorted_stations[i + 1])

# 提取邻接矩阵
adj_matrix = nx.adjacency_matrix(G).todense()

# 读取csv
path = 'CTS-2024-dataset/'
csvfile = ['station_20230519.csv', 'station_20230520.csv']
stationList = utility.cstRawCsvData([path + csvfile[i]] for i in range(len(csvfile)))
station = stationList.getAllStationData()

# 数据预加载
from datetime import datetime

sta_info_for_adj = pd.DataFrame(columns=['station_name', 'abs_date', 'time_point', 'day_of_week', 'is_event', 'in_flow', 'out_flow'])
for i in range(len(station)):
    station_name = station.loc[i, '站点名']
    precise_time = datetime.strptime(station.loc[i, '开始时间'], '%Y/%m/%d %H:%M:%S')
    time_diff = precise_time - datetime.strptime('2023/5/19 00:00:00', '%Y/%m/%d %H:%M:%S')
    abs_date = time_diff.days + 1 # 从1开始编号，五月19是第一天
    time_point = (precise_time.hour * 60 + precise_time.minute) / 30
    day_of_week = precise_time.weekday() # Monday == 0 ... Sunday == 6
    event_condition = ((precise_time.day == 26 or precise_time.day == 27) and precise_time.month == 5) or ((precise_time.day == 25) and precise_time.month == 8)
    is_event = 1 if event_condition else 0
    in_flow = float(station.loc[i, '进站量'])
    out_flow = float(station.loc[i, '出站量'])
    sta_info_for_adj.loc[i] = [station_name, abs_date, time_point, day_of_week, is_event, in_flow, out_flow]

# 准备模型输入与loss标杆
# 找到所有非零元素的索引
adj_matrix = np.array(adj_matrix)
row, col = np.nonzero(adj_matrix)

# 将这些索引转换为 PyTorch 张量
edge_index = torch.tensor([row, col], dtype=torch.long)
data_input: Data = Data(x=torch.tensor(sta_info_for_adj[['abs_date', 'time_point', 'day_of_week', 'is_event']].values, dtype=torch.float),
                        edge_index=edge_index,
                        y=(torch.tensor(sta_info_for_adj['in_flow'].values, dtype=torch.float), torch.tensor(sta_info_for_adj['out_flow'].values, dtype=torch.float)))

# prepare model parameters
num_features = 4
num_classes = 2
model = models.GCN(num_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# start training
for epoch in range(200):
    optimizer.zero_grad()  # 清除梯度
    out1, out2 = model(data_input)  # 前向传播
    loss1 = criterion(out1, data_input.y[0])  # 计算损失
    loss2 = criterion(out2, data_input.y[1])
    loss = loss1 + loss2  # 加权求和
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# make prediction
model.eval()
pre_out1, pre_out2 = model(predic_data)
