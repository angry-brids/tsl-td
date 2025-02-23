import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# MLP模型定义
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=6):
        super(MLPClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# 数据增强方法
def add_noise(data, noise_level=0.1):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def random_transpose(data, prob=0.5):
    if np.random.random() < prob:
        return data.T
    return data


def add_random_peaks(data, n_peaks=2, peak_magnitude=1.0):
    data_copy = data.copy()
    idx = np.random.choice(len(data), n_peaks, replace=False)
    data_copy[idx] += peak_magnitude * np.random.randn(n_peaks)
    return data_copy


# 提取统计特征
def extract_stat_features(data):
    features = np.column_stack([
        np.mean(data, axis=1),
        np.std(data, axis=1),
        np.min(data, axis=1),
        np.max(data, axis=1),
        np.median(data, axis=1)
    ])
    return features


# 数据预处理和归一化
def preprocess_data(X, method='local'):
    if method == 'local':
        # 局部归一化（每个样本单独归一化）
        scaler = StandardScaler()
        X_scaled = np.array([scaler.fit_transform(x) for x in X])
    else:
        # 全局归一化
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
    return X_scaled


# 训练和评估函数
def train_and_evaluate(X, y, input_size, window_size, description):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # 初始化模型
    model = MLPClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()

    print(f"{description} - Accuracy: {accuracy:.4f}")
    return accuracy
