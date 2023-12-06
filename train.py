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
        self.fc = torch.nn.Linear(hidden_dim, seq_out_len * output_dim)
        self.seq_out_len = seq_out_len
        self.output_dim = output_dim
        self.batch_size = batch_size

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 全局池化

        x = self.fc(x)

        # 重塑输出以符合多步预测的需求
        x = x.view(-1, self.seq_out_len, self.output_dim)

        return x



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
    for i in range(len(sequence) - seq_in_len - seq_out_len + 1):
        seq_x = sequence.iloc[i:i + seq_in_len].values
        seq_y = sequence.iloc[i + seq_in_len:i + seq_in_len + seq_out_len].values
        X.append(seq_x)
        Y.append(seq_y)
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
    # 应用 split_sequence_graph 函数并创建小图
    x, y = split_sequence_graph(data, seq_in_len, seq_out_len)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # 为每个时间窗口创建一个小图
    train_graphs = build_small_graphs(train_x, train_y, look_back)
    test_graphs = build_small_graphs(test_x, test_y, look_back)

    # 使用 DataLoader 加载小图
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    print("Length of DataLoader:", len(train_loader))
    # 用 x 的形状来初始化模型
    model = GATModel(input_dim=x.shape[-1], hidden_dim=hidden_dim, output_dim=output_dim, seq_out_len=seq_out_len,batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 初始化指标存储列表
    mse_scores, rmse_scores, mae_scores, r2_scores, mape_scores = [], [], [], [], []

    # 运行模型 10 次
    num_runs = 10
    for run in range(num_runs):
        predictions = []
        actuals = []
        # 训练循环
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
            for batch_idx, batch_data in progress_bar:
                optimizer.zero_grad()
                batch_graphs = batch_data.to(device)
                outputs = model(batch_graphs)
                targets = batch_data.y.to(device)
                targets = targets.view_as(outputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"Train Loss": f"{train_loss / (batch_idx + 1):.4f}"})

        # 测试循环
        model.eval()
        test_loss = 0.0
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
        with torch.no_grad():
            for batch_idx, batch_data in progress_bar:
                batch_graphs = batch_data.to(device)
                outputs = model(batch_graphs)
                targets = batch_data.y.to(device)
                targets = targets.view_as(outputs)  # 重塑目标以匹配输出形状
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                progress_bar.set_postfix({"Test Loss": f"{test_loss / (batch_idx + 1):.4f}"})
                # 收集预测值和实际值，保持其三维结构
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        # 将预测和实际值转换为 NumPy 数组并重塑为二维
        predictions = np.array(predictions).reshape(-1, output_dim)
        actuals = np.array(actuals).reshape(-1, output_dim)

        # 确保预测值和实际值的尺寸一致
        assert predictions.shape == actuals.shape, "Inconsistent shapes between predictions and actuals"
        # 计算评估指标
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mape = mean_absolute_percentage_error(actuals, predictions)

        # 存储指标
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        mape_scores.append(mape)

        # 打印当前迭代的评估指标
        print(f'Run {run + 1}: MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R-squared: {r2}, MAPE: {mape}')
        print('-' * 50)

    # 计算平均评估指标
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    avg_mape = np.mean(mape_scores)

    # 打印平均评估指标
    print('Average Metrics Over Runs')
    print('Average MSE:', avg_mse)
    print('Average RMSE:', avg_rmse)
    print('Average MAE:', avg_mae)
    print('Average R-squared:', avg_r2)
    print('Average MAPE:', avg_mape)
    print('-' * 100)