import torch.nn.functional as F
from torch.nn import Module
from model import layers

class GCN(Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = layers.GCNConvLayer(num_features, 24)
        self.conv2 = layers.GCNConvLayer(24, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)