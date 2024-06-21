import torch.nn.functional as F
from torch.nn import Module
import torch.nn as nn
import torch
from model import layers

class GCN(Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = layers.GCNConvLayer(num_features, 24)
        self.conv2 = layers.GCNConvLayer(24, 16)
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # node_mask = data.node_mask

        # if node_mask is not None:
        #     x = x * node_mask.unsqueeze(-1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        x = F.relu(x)
        return x#F.log_softmax(x, dim=1)
    
class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, time_steps, temporal_channels, spatial_channels, kernel_size,out_channels):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.temporal1 = layers.TemporalConvLayer(in_channels, temporal_channels, kernel_size)
        self.spatial = layers.SpatialConvLayer(temporal_channels, spatial_channels)
        self.temporal2 = layers.TemporalConvLayer(spatial_channels, in_channels, kernel_size)
        self.fc = torch.nn.Linear(num_nodes * in_channels * time_steps, num_nodes * out_channels * time_steps)

    def forward(self, x, edge_index):
        x = self.temporal1(x)
        x = self.spatial(x, edge_index)
        x = self.temporal2(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.num_nodes, self.time_steps, -1)  # Reshape back to (batch, N, T, F)
        return x
