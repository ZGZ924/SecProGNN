import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, Linear, global_mean_pool
class GlobalMeanPool(nn.Module):
    def __init__(self):
        super(GlobalMeanPool, self).__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)

class GraphSELayer(nn.Module):
    def __init__(self, node_channels, reduction=16):
        super(GraphSELayer, self).__init__()
        self.node_channels = node_channels
        self.reduction = reduction
        # 将 GlobalMeanPool 类作为一个模块添加
        self.global_pool = GlobalMeanPool()
        self.fc = nn.Sequential(
            nn.Linear(node_channels, node_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(node_channels // reduction, node_channels, bias=False),
            nn.Sigmoid()
        )
        
        

    def forward(self, x, batch):
        mean_pool = self.global_pool(x, batch)
        scale = self.fc(mean_pool)
        scale = torch.repeat_interleave(scale, batch.bincount(), dim=0)
        return x * scale.view(-1, self.node_channels)


# 集成SELayer的GNN模型
class SecProGNN(torch.nn.Module):
    def __init__(self, input_features, hidden_channels, num_classes):
        super(SecProGNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(input_features, hidden_channels)
        self.se1 = GraphSELayer(hidden_channels)  # SELayer after GraphConv
 
        self.conv2 = GraphConv(hidden_channels, hidden_channels * 2)
        self.se2 = GraphSELayer(hidden_channels * 2)  # SELayer after GraphConv
        
        self.conv3 = GraphConv(hidden_channels * 2, hidden_channels * 4)
        self.se3 = GraphSELayer(hidden_channels * 4)  # SELayer after GraphConv    
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # Convolutional layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.se1(x, batch)  


        # Convolutional layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.se2(x, batch)  

        # Convolutional layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.se3(x, batch)  


        # Global mean pooling
        x = global_mean_pool(x, batch)
        
        x = self.classifier(x)



        return x
