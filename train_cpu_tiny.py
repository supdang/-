#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简CPU训练脚本
用于在极低资源环境下训练刀具磨损状态识别模型
"""

import sys
import os
# 设置环境变量以限制线程数
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
from sklearn.model_selection import train_test_split
import gc  # 垃圾回收

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger


class TinyLSTMModel(nn.Module):
    """极简LSTM模型"""
    def __init__(self, input_size=7, hidden_size=16, num_layers=1, num_classes=4):
        super(TinyLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 极简LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1)
        
        # 极简全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out


def create_tiny_demo_data():
    """创建极小规模演示数据"""
    print("创建极小规模演示数据...")
    
    # 使用极小的数据集
    n_samples = 200  # 极大减少样本数
    n_channels = 7   # 7个传感器通道
    window_size = 32 # 极小的窗口大小
    
    # 创建模拟传感器数据
    X = np.random.randn(n_samples, window_size, n_channels).astype(np.float32)
    
    # 添加非常简单的模式
    for i in range(n_channels):
        # 添加微小的趋势
        trend = np.linspace(0, 0.1, window_size)
        X[:, :, i] += trend * np.random.uniform(-0.1, 0.1)
    
    # 创建标签 (0:初期磨损, 1:正常磨损, 2:后期磨损, 3:失效状态)
    y = np.random.randint(0, 4, n_samples).astype(np.int64)
    
    print(f"  特征形状: {X.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  标签分布: {np.bincount(y)}")
    
    return X, y


def train_tiny_model():
    """训练极简模型"""
    print("=" * 60)
    print("极简刀具磨损状态识别模型 - CPU训练")
    print("=" * 60)
    
    # 设置日志
    logger = setup_logger(log_level="INFO")
    logger.info("开始极简CPU训练流程")
    
    try:
        # 1. 创建极小规模数据
        print("1. 创建极小规模演示数据...")
        X, y = create_tiny_demo_data()
        
        # 2. 数据分割
        print("\n2. 数据分割...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"  训练集形状: {X_train.shape}")
        print(f"  验证集形状: {X_val.shape}")
        print(f"  测试集形状: {X_test.shape}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # 创建数据加载器 - 使用极小批次
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # 极小批次
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # 3. 创建极简模型
        print("\n3. 创建极简LSTM模型...")
        device = torch.device('cpu')
        model = TinyLSTMModel(input_size=7, hidden_size=8, num_layers=1, num_classes=4)  # 更小的模型
        
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  使用设备: {device}")
        
        # 4. 设置优化器和损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 5. 训练模型
        print("\n4. 开始训练...")
        print(f"  训练轮数: 10 (极少量)")
        print(f"  批量大小: 4 (极小批量)")
        print(f"  学习率: 0.001")
        
        model.train()
        best_val_acc = 0.0
        
        for epoch in range(10):  # 极少量epoch
            # 训练
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_acc = train_correct / train_total
            
            # 验证
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = val_correct / val_total
            model.train()  # 切换回训练模式
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            print(f"  Epoch [{epoch+1}/10] - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        print(f"\n  训练完成！最佳验证准确率: {best_val_acc:.4f}")
        
        # 6. 测试模型
        print("\n5. 测试模型...")
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = test_correct / test_total
        print(f"  测试准确率: {test_acc:.4f}")
        
        # 7. 保存模型
        print("\n6. 保存模型...")
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "tiny_model.pth"
        
        # 保存模型状态
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': 7,
                'hidden_size': 8,
                'num_layers': 1,
                'num_classes': 4
            },
            'test_accuracy': test_acc
        }, model_path)
        
        print(f"  模型已保存到: {model_path}")
        
        # 8. 内存清理
        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
        del train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader
        del model, optimizer
        gc.collect()  # 强制垃圾回收
        
        print("\n" + "=" * 60)
        print("极简CPU训练完成！")
        print(f"模型已保存到: {model_path}")
        print("注意: 此模型仅用于验证CPU训练功能")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"错误: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("极简CPU训练脚本")
    print("此脚本专为极低资源环境设计")
    
    # 运行训练函数
    exit_code = train_tiny_model()
    
    sys.exit(exit_code)