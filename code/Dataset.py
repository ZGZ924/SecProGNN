import os
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)


class BALFInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, train_size=0.8):
        self.train_size = train_size
        super(BALFInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BALF_A.txt', 'BALF_node_labels.txt', 'BALF_graph_indicator.txt', 'BALF_graph_labels.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 如果数据集可以从网上下载，应在此方法中实现下载逻辑。
        pass


    def process(self):
        # 读取数据文件
        data_list = []
        
        data_dir = os.path.join(self.root, 'raw')
        
        adjacency_list = np.genfromtxt(os.path.join(data_dir, "BALF_A.txt"), dtype=np.int64, delimiter=',') - 1
        
        node_labels = np.genfromtxt(os.path.join(data_dir, "BALF_node_labels.txt"), dtype=np.int64) - 1
        
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "BALF_graph_indicator.txt"), dtype=np.int64) - 1
        
        graph_labels = np.genfromtxt(os.path.join(data_dir, "BALF_graph_labels.txt"), dtype=np.int64)
        node_features = pd.read_csv(os.path.join(data_dir, "BALF_node_features.csv")).values
        # 确定节点标签的总数（类别数）
        num_classes = np.max(node_labels) + 1
        
        # 创建稀疏邻接矩阵并将其转换为 CSR 格式
        num_nodes = len(node_labels)
        sparse_adjacency = sp.coo_matrix(
        (np.ones(len(adjacency_list)), (adjacency_list[:, 0], adjacency_list[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
        ).tocsr()
        
        # 处理每个图
        unique_graph_ids = np.unique(graph_indicator)

        for graph_id in unique_graph_ids:
            mask = graph_indicator == graph_id
            node_labels_graph = node_labels[mask]
            graph_label = graph_labels[graph_id]
            node_features_graph = node_features[mask]  # 获取当前图的节点特征
                                    
            # 使用 CSR 矩阵进行切片操作
            adjacency_graph = sparse_adjacency[mask][:, mask].tocoo()
            
            # 使用 PyTorch 的 F.one_hot 函数将节点标签转换为独热编码
            node_labels_graph = torch.tensor(node_labels_graph, dtype=torch.long)
            node_labels_one_hot = F.one_hot(node_labels_graph, num_classes=num_classes).to(torch.float)
            node_features_graph = torch.tensor(node_features_graph, dtype=torch.float)
            x = torch.cat((node_labels_one_hot, node_features_graph), dim=1)  # 合并独热编码和节点特征

            

             # 创建 PyG Data 对象
            edge_index = torch.tensor([adjacency_graph.row, adjacency_graph.col], dtype=torch.long)
            x = x
            y = torch.tensor([graph_label], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            data_list.append(data)

        # 将数据保存为 PyTorch 文件
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.data.y)

    def split_data(self, train_size):
        # 这个方法可以根据需要调整，以生成训练集和测试集的索引
        num_graphs = self.len()
        indices = np.arange(num_graphs)
        return train_test_split(indices, train_size=train_size, random_state=1234)
