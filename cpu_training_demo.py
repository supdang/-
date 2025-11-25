#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU训练演示脚本
演示如何在代码中强制使用CPU进行训练
"""

import sys
import os

# 设置环境变量以限制线程数，减少内存使用
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("CPU训练演示")
print("=" * 60)

# 检查PyTorch和设备信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备数量: {torch.cuda.device_count()}")

# 强制使用CPU
device = torch.device('cpu')
print(f"当前使用设备: {device}")

# 创建极小的演示数据集
print("\n创建演示数据...")
n_samples = 50
n_features = 7
n_classes = 4

X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, n_classes, n_samples).astype(np.int64)

print(f"数据形状: X={X.shape}, y={y.shape}")

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# 创建数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 定义一个简单的线性模型（避免使用可能消耗更多资源的复杂层）
class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型并移动到CPU
model = SimpleModel(n_features, n_classes)
model = model.to(device)  # 确保模型在CPU上

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
print(f"模型设备: {next(model.parameters()).device}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(f"\n开始在CPU上训练...")
print(f"训练轮数: 5")
print(f"批次大小: 10")

# 训练循环
model.train()
for epoch in range(5):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        # 确保数据在CPU上
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"  Epoch [{epoch+1}/5], Loss: {avg_loss:.4f}")

print(f"\n✓ CPU训练演示完成！")
print(f"模型成功在CPU上完成训练")
print(f"模型已定义并可以在CPU上正常运行")

# 保存模型以证明训练完成
model_path = Path(__file__).parent / 'cpu_model_demo.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_size': n_features,
        'num_classes': n_classes
    }
}, model_path)

print(f"模型已保存到: {model_path}")

print("\n" + "=" * 60)
print("CPU训练演示完成！")
print("此演示证明了如何在代码中强制使用CPU进行训练")
print("=" * 60)

# 额外说明
print("\n关键要点：")
print("1. 使用 torch.device('cpu') 强制指定CPU设备")
print("2. 使用 .to(device) 将模型和数据移动到指定设备")
print("3. 在模型初始化时指定 device='cpu' 参数（如项目中的 ToolWearClassifier）")
print("4. 设置环境变量限制线程数以减少资源使用")
print("5. 在资源受限环境中使用更小的批次大小和更简单的模型架构")