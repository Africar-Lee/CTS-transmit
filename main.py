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

# station数据预处理函数
def station2staion_no_line(station):
    """ 同一站点在不同线路上都有数据，合并这些数据 """
    station_no_line = pd.DataFrame()
    for i in range(len(station['站点名'].unique())):
        spec_time_place_station = station[station['站点名'] == station['站点名'].unique()[i]]
        # spec_time_place_station = spec_time_place_station[['开始时间','线路名','站点名','进站量','出站量']]
        spec_time_place_station = spec_time_place_station.fillna(0)
        spec_time_place_station = spec_time_place_station.groupby(['日期','开始时间','结束时间']).agg({
            '站点名': 'first',  # 假设每个时间段内站点名是相同的，取第一个即可
            '进站量': 'sum',
            '出站量': 'sum'
        })
        spec_time_place_station = spec_time_place_station.reset_index()
        station_no_line = pd.concat([station_no_line, spec_time_place_station]).reset_index(drop=True)
    return station_no_line


# prepare data
line,stop = tbd.getbusdata('北京',['1号线', '2号线', '5号线', '6号线', '7号线', '8号线', '9&房山线','房山线', '10号线',
       '13号线', '15号线', '昌平线', '首都机场线'])
line = line[line['line']!='大兴机场大巴首都机场线'].reset_index(drop=True)
stop = stop[stop['line']!='大兴机场大巴首都机场线'].reset_index(drop=True)
line = line[line['line']!='首都机场大巴昌平线'].reset_index(drop=True)
stop = stop[stop['line']!='首都机场大巴昌平线'].reset_index(drop=True)
stop.loc[stop['stationnames'] == '清河站', 'stationnames'] = '清河'

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
station_csv_file = ['station_20230519.csv',
                    'station_20230520.csv',
                    'station_20230521.csv',
                    'station_20230522.csv',
                    'station_20230523.csv',
                    'station_20230524.csv',
                    'station_20230525.csv',]
station_input_csv_select = range(len(station_csv_file))
# stationList = utility.cstRawCsvData([path + station_csv_file[i] for i in station_input_csv_select])
station = utility.get_station_for_adj(stop_for_adj, [path + station_csv_file[i] for i in station_input_csv_select])
station = station2staion_no_line(station) # 合并分布在不同线路上的同一站点数据

total_input_sz = len(station)


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

categories_list = list(stop_for_adj['站点名'])
my_x = []
my_y = []
for i in sta_info_for_adj.groupby(["abs_date","time_point"]):
    temp = i[1]
    temp['station_name'] = pd.Categorical(temp['station_name'], categories=categories_list, ordered=True)
    temp = temp.sort_values(by=['station_name']).reset_index(drop=True)#特定日期特定时间的所有站点已知数据
    features = temp[['abs_date','time_point','day_of_week','is_event']].values
    labels = temp[['in_flow','out_flow']].values
    my_x.append(features)
    my_y.append(labels)

my_x = np.array(my_x)
my_y = np.array(my_y)


# 准备模型输入与loss标杆
# 找到所有非零元素的索引
adj_matrix = np.array(adj_matrix)
row, col = np.nonzero(adj_matrix)

# 将这些索引转换为 PyTorch 张量
edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
data_input: List[Data] = [Data(x=torch.tensor(my_x[epoch], dtype=torch.float),
                               edge_index=edge_index,
                               y=torch.tensor(my_y[epoch], dtype=torch.float))
                          for epoch in range(len(my_x))]


# prepare model parameters
num_features = 4
num_classes = 2
model = models.GCN(num_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# start training
duptimes = 20 # 暂定每个数据输入进模型训2次
total_input_sz = len(data_input)
test_set_sz = 48 # 预测集大小
train_set_sz = total_input_sz - test_set_sz
for epoch in range(duptimes * train_set_sz):
    optimizer.zero_grad()  # 清除梯度
    out_y = model(data_input[epoch % train_set_sz])  # 前向传播
    loss1 = criterion(out_y[0], data_input[epoch % train_set_sz].y[0])  # 计算损失
    loss2 = criterion(out_y[1], data_input[epoch % train_set_sz].y[1])
    loss = loss1 + loss2  # 加权求和
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# make prediction
model.eval()
predict_data: List[Data] = data_input[train_set_sz:]

predict_in_flow = []
predict_out_flow = []
for piece in predict_data:
    pre_out = model(piece)
    predict_in_flow.append(pre_out[:, 0])
    predict_out_flow.append(pre_out[:, 1])

# 按照GPT建议简单写了下性能评估
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 将预测值和真实值转换为 numpy 数组以便使用 sklearn 库计算评估指标
true_in_flow = np.array([data.y[:, 0].detach().numpy() for data in predict_data])
true_out_flow = np.array([data.y[:, 1].detach().numpy() for data in predict_data])
predicted_in_flow = np.array([out.detach().numpy() for out in predict_in_flow])
predicted_out_flow = np.array([out.detach().numpy() for out in predict_out_flow])

# 计算 MAE 和 RMSE
mae_in = mean_absolute_error(true_in_flow, predicted_in_flow)
rmse_in = np.sqrt(mean_squared_error(true_in_flow, predicted_in_flow))

mae_out = mean_absolute_error(true_out_flow, predicted_out_flow)
rmse_out = np.sqrt(mean_squared_error(true_out_flow, predicted_out_flow))

print(f"MAE for In-Flow: {mae_in}, RMSE for In-Flow: {rmse_in}")
print(f"MAE for Out-Flow: {mae_out}, RMSE for Out-Flow: {rmse_out}")
