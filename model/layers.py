import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConvLayer(MessagePassing):
    """GCNConv Layer, copied from pyG official website"""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    
class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size[0] // 2, 0))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (batch, N, T, F) -> (batch, F, T, N)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (batch, F, T, N) -> (batch, N, T, F)
        return x

class SpatialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialConvLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: (batch, N, T, F)
        # edge_index: (2, E)
        batch_size, num_nodes, time_steps, num_features = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch, N, T, F) -> (batch, T, N, F)
        x = x.reshape(-1, num_nodes, num_features)  # (batch * T, N, F)
        
        # Flatten edge_index
        edge_index = edge_index.repeat(time_steps * batch_size, 1)  # Repeat edge_index for each time step and batch
        edge_index = edge_index.view(2, -1)  # (2, E * batch_size * time_steps)
        
        x = x.reshape(-1, num_features)  # (batch * T * N, F)
        x = self.gcn(x, edge_index)
        x = x.view(batch_size, time_steps, num_nodes, -1)  # (batch, T, N, F)
        x = x.permute(0, 2, 1, 3)  # (batch, T, N, F) -> (batch, N, T, F)
        return x
