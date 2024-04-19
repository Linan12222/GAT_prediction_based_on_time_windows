import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
from tqdm import tqdm

class StreamToLogger(object):
    """
    自定义流，用于同时将输出信息发送到标准输出和文件。
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 这个函数在这里是为了兼容文件对象的接口
        self.terminal.flush()
        self.log.flush()

sys.stdout = StreamToLogger("console_output_GAT.txt")

class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_out_len, batch_size):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, seq_out_len * output_dim)  # 均值预测
        self.fc_sigma = torch.nn.Linear(hidden_dim, seq_out_len * output_dim)  # 标准差预测
        self.seq_out_len = seq_out_len
        self.output_dim = output_dim
        self.batch_size = batch_size

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 全局池化

        mu = self.fc_mu(x)
        sigma = F.softplus(self.fc_sigma(x))  # 使用softplus确保标准差为正

        # 重塑输出以符合多步预测的需求
        mu = mu.view(-1, self.seq_out_len, self.output_dim)
        sigma = sigma.view(-1, self.seq_out_len, self.output_dim)

        return mu, sigma




def build_small_graphs(x, y, look_back):
    graph_list = []
    for i in range(len(x)):
        num_nodes = len(x[i])
        edges = []
        for j in range(look_back, num_nodes):
            for k in range(1, look_back + 1):
                edges.append((j, j - k))  # 确保边是在同一个样本内部构建

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(x[i], dtype=torch.float)
        target = torch.tensor(y[i], dtype=torch.float)  # 单个样本的目标
        graph_list.append(Data(x=node_features, edge_index=edge_index, y=target))
    return graph_list



def split_sequence_graph(sequence, seq_in_len, seq_out_len):
    X, Y = [], []
    # 确保在迭代中留有足够的行来完成seq_out_len的抽取
    for i in range(len(sequence) - seq_in_len - seq_out_len + 1):
        seq_x = sequence.iloc[i:i + seq_in_len].values
        # 此处改动: 抽取从seq_in_len开始的seq_out_len行数据作为每个seq_y
        seq_y = sequence.iloc[i + seq_in_len:i + seq_in_len + seq_out_len].values
        X.append(seq_x)
        Y.append(seq_y)  # 这里seq_y应该自然地是[seq_out_len, 特征数]的形状
    return np.array(X), np.array(Y)



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_indices = y_true != 0  # 避免除以零
    return np.mean(np.abs((y_true[nonzero_indices] - y_pred[nonzero_indices]) / y_true[nonzero_indices])) * 100

def custom_collate(batch):
    graph_data_list = [item[0] for item in batch]
    index_list = [item[1] for item in batch]

    batched_graph = Batch.from_data_list(graph_data_list)
    index_tensor = torch.tensor(index_list, dtype=torch.long)

    return batched_graph, index_tensor


def calculate_interval_metrics(actuals, predictions, sigmas, coverage_factor=1.96):
    lower_bounds = predictions - coverage_factor * sigmas
    upper_bounds = predictions + coverage_factor * sigmas

    # 计算PICP
    in_interval = (actuals >= lower_bounds) & (actuals <= upper_bounds)
    PICP = np.mean(in_interval)

    # 计算ACIW
    ACIW = np.mean(upper_bounds - lower_bounds)

    # 计算PINAW
    data_range = actuals.max() - actuals.min()
    PINAW = ACIW / data_range if data_range != 0 else 0

    # 计算CWC
    CWC = PINAW * (1.0 - PICP) if PICP < 0.95 else PINAW * (1.0 - PICP + 0.1 * (PICP - 0.95))

    return PICP, ACIW, PINAW, CWC


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

data = pd.read_excel('17+18-已处理.xlsx')
data = data.iloc[1:, 1:]
look_back = 4
seq_in_len = 24*7  # 设置输入序列的长度
hidden_dim=64
output_dim = 14
epochs = 100
batch_size = 32
# 设置循环
seq_out_lens = [1, 4, 8, 13, 16]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for seq_out_len in seq_out_lens:
    print(f"Training for seq_out_len = {seq_out_len}")
    # 应用 split_sequence_graph 函数并创建小图
    x, y = split_sequence_graph(data, seq_in_len, seq_out_len)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # 为每个时间窗口创建一个小图
    train_graphs = build_small_graphs(train_x, train_y, look_back)
    test_graphs = build_small_graphs(test_x, test_y, look_back)

    # 使用 DataLoader 加载小图
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    # 用 x 的形状来初始化模型
    model = GATModel(input_dim=x.shape[-1], hidden_dim=hidden_dim, output_dim=output_dim, seq_out_len=seq_out_len,batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 初始化指标存储列表
    picp_scores, aciw_scores, pinaw_scores, cwc_scores = [], [], [], []

    # 运行模型 10 次
    num_runs = 10
    for run in range(num_runs):
        predictions = []
        sigmas = []
        actuals = []

        # 训练循环
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, batch_data in progress_bar:
                optimizer.zero_grad()
                batch_graphs = batch_data.to(device)
                mu, sigma = model(batch_graphs)

                targets = batch_data.y.to(device)
                targets = targets.view(mu.shape)  # 重塑目标以匹配输出形状
                loss = criterion(mu, targets)  # 可以根据需要修改损失函数
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                progress_bar.set_postfix({"Train Loss": f"{train_loss / (batch_idx + 1):.4f}"})

        # 测试循环
        model.eval()
        total_batches = len(test_loader)
        processed_batches = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                batch_graphs = batch_data.to(device)
                mu, sigma = model(batch_graphs)
                targets = batch_data.y.to(device)
                predictions.extend(mu.cpu().numpy())
                sigmas.extend(sigma.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
                processed_batches += 1

        # 计算评估指标
        PICP, ACIW, PINAW, CWC = calculate_interval_metrics(np.array(actuals), np.array(predictions),
                                                                    np.array(sigmas))

        # 存储指标
        picp_scores.append(PICP)
        aciw_scores.append(ACIW)
        pinaw_scores.append(PINAW)
        cwc_scores.append(CWC)

        # 打印当前迭代的评估指标
        print(f'Run {run + 1}: PICP: {PICP}, ACIW: {ACIW}, PINAW: {PINAW}, CWC: {CWC}')
        print('-' * 50)

    # 计算平均评估指标
    avg_picp = np.mean(picp_scores)
    avg_aciw = np.mean(aciw_scores)
    avg_pinaw = np.mean(pinaw_scores)
    avg_cwc = np.mean(cwc_scores)

    # 打印平均评估指标
    print('Average Metrics Over Runs')
    print('Average PICP:', avg_picp)
    print('Average ACIW:', avg_aciw)
    print('Average PINAW:', avg_pinaw)
    print('Average CWC:', avg_cwc)
    print('-' * 100)
