#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级CPU训练脚本
用于在资源受限的环境下训练刀具磨损状态识别模型
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config.config import DataConfig, ModelConfig
from src.models.lstm_model import ToolWearClassifier, LSTMToolWearModel
from src.utils.logger import setup_logger


def create_small_demo_data():
    """创建小规模演示数据"""
    print("创建小规模演示数据...")
    
    # 使用更小的数据集以适应内存限制
    n_samples = 500  # 显著减少样本数
    n_channels = 7   # 7个传感器通道
    window_size = 64 # 更小的窗口大小
    
    # 创建模拟传感器数据
    X = np.random.randn(n_samples, window_size, n_channels).astype(np.float32)
    
    # 添加一些简单的模式到数据中以模拟不同磨损状态
    for i in range(n_channels):
        # 添加趋势以模拟磨损过程
        trend = np.linspace(0, 0.5, window_size)
        X[:, :, i] += trend * np.random.uniform(-0.2, 0.2)
        # 添加周期性信号
        X[:, :, i] += 0.1 * np.sin(np.linspace(0, 2*np.pi, window_size)) * np.random.uniform(0.5, 1.5)
    
    # 创建标签 (0:初期磨损, 1:正常磨损, 2:后期磨损, 3:失效状态)
    y = np.random.randint(0, 4, n_samples).astype(np.int64)
    
    print(f"  特征形状: {X.shape}")
    print(f"  标签形状: {y.shape}")
    print(f"  标签分布: {np.bincount(y)}")
    
    return X, y


def main():
    """主训练函数"""
    print("=" * 60)
    print("轻量级刀具磨损状态识别模型 - CPU训练")
    print("=" * 60)
    
    # 设置日志
    logger = setup_logger(log_level="INFO")
    logger.info("开始轻量级CPU训练流程")
    
    try:
        # 1. 创建小规模数据
        print("1. 创建小规模演示数据...")
        X, y = create_small_demo_data()
        
        # 2. 数据分割
        print("\n2. 数据分割...")
        # 使用更小的比例分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y  # 减少测试集比例
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train  # 验证集占训练集的25%
        )
        
        print(f"  训练集形状: {X_train.shape}")
        print(f"  验证集形状: {X_val.shape}")
        print(f"  测试集形状: {X_test.shape}")
        
        # 3. 创建简化模型
        print("\n3. 创建简化LSTM模型...")
        
        # 创建一个更小的模型以节省内存
        model = LSTMToolWearModel(
            input_size=7,           # 输入特征维度
            hidden_size=32,         # 显著减少隐藏层大小
            num_layers=1,           # 只使用1层LSTM
            num_classes=4,          # 4个磨损状态
            dropout_rate=0.1        # 较小的dropout
        )
        
        # 创建分类器，强制使用CPU
        classifier = ToolWearClassifier(model=model, device='cpu')
        
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  使用设备: {classifier.device}")
        
        # 4. 训练模型（使用更小的批次和更少的轮数）
        print("\n4. 开始训练...")
        print(f"  训练轮数: 20 (减少轮数以节省时间)")
        print(f"  批量大小: 8 (小批量以节省内存)")
        print(f"  学习率: 0.001")
        
        # 使用更小的批量大小和更少的训练轮数
        history = classifier.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            batch_size=8,           # 小批量
            num_epochs=20,          # 少量epoch
            learning_rate=0.001,
            patience=5              # 早停耐心值
        )
        
        print(f"\n  训练完成！最佳验证准确率: {classifier.best_val_acc:.4f}")
        
        # 5. 评估模型
        print("\n5. 评估模型...")
        test_results = classifier.evaluate((X_test, y_test))
        print(f"  测试准确率: {test_results['accuracy']:.4f}")
        
        # 6. 保存模型
        print("\n6. 保存模型...")
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "lightweight_model.pth"
        classifier.save_model(str(model_path))
        print(f"  模型已保存到: {model_path}")
        
        print("\n" + "=" * 60)
        print("轻量级CPU训练完成！")
        print(f"模型已保存到: {model_path}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        import traceback
        print(f"错误: {str(e)}")
        traceback.print_exc()
        return 1


def verify_cpu_usage():
    """验证是否使用CPU进行训练"""
    print("验证CPU使用情况...")
    
    # 检查PyTorch是否可用
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"设备列表: CPU (无GPU)")
    
    # 确保使用CPU
    device = torch.device('cpu')
    print(f"当前设备: {device}")
    
    return device


if __name__ == "__main__":
    print("轻量级CPU训练脚本")
    print("此脚本专为资源受限环境设计")
    
    # 验证CPU使用
    device = verify_cpu_usage()
    
    # 运行主训练函数
    exit_code = main()
    
    sys.exit(exit_code)