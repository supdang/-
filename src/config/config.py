#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置文件
包含所有系统配置参数和常量定义
"""

import os
from pathlib import Path

# 系统路径配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# 创建必要的目录
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

# 数据配置
class DataConfig:
    """数据相关配置"""
    # PHM2010数据集配置
    SAMPLE_RATE = 50000  # 采样频率50kHz
    WINDOW_SIZE = 1024   # 滑动窗口大小
    STEP_SIZE = 512      # 滑动步长
    
    # 传感器通道配置
    SENSOR_CHANNELS = {
        'force_x': 0,   # X方向切削力
        'force_y': 1,   # Y方向切削力
        'force_z': 2,   # Z方向切削力
        'vibration_x': 3,  # X方向振动
        'vibration_y': 4,  # Y方向振动
        'vibration_z': 5,  # Z方向振动
        'ae': 6         # 声发射信号
    }
    
    # 磨损状态分类
    WEAR_STATES = {
        0: '初期磨损',   # VBmax < 0.12mm
        1: '正常磨损',   # 0.12mm ≤ VBmax < 0.2mm
        2: '后期磨损',   # 0.2mm ≤ VBmax < 0.3mm
        3: '失效状态'    # VBmax ≥ 0.3mm
    }
    
    # 磨损阈值
    WEAR_THRESHOLDS = [0.12, 0.2, 0.3]

# 模型配置
class ModelConfig:
    """模型相关配置"""
    # LSTM网络结构
    INPUT_SIZE = 7      # 输入特征维度（7个传感器通道）
    HIDDEN_SIZE = 128   # LSTM隐藏层大小
    NUM_LAYERS = 2      # LSTM层数
    NUM_CLASSES = 4     # 分类数（4个磨损状态）
    DROPOUT_RATE = 0.2  # Dropout比率
    
    # 训练参数
    BATCH_SIZE = 32     # 批量大小
    NUM_EPOCHS = 100    # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    WEIGHT_DECAY = 1e-5    # 权重衰减
    
    # 早停机制
    PATIENCE = 15       # 早停 patience
    MIN_DELTA = 0.001   # 最小改善量
    
    # 模型保存
    MODEL_NAME = "tool_wear_lstm_model.pth"
    BEST_MODEL_NAME = "tool_wear_lstm_best_model.pth"

# 预处理配置
class PreprocessConfig:
    """预处理相关配置"""
    # 小波去噪
    WAVELET_NAME = 'db4'  # 小波基函数
    DECOMPOSITION_LEVEL = 4  # 分解层数
    
    # 归一化
    NORMALIZATION_METHOD = 'minmax'  # 归一化方法
    
    # 数据增强
    NOISE_LEVEL = 0.01    # 噪声水平
    SCALE_RANGE = (0.8, 1.2)  # 缩放范围

# GUI配置
class GUIConfig:
    """GUI相关配置"""
    # 窗口配置
    WINDOW_TITLE = "基于深度学习的刀具磨损状态识别与诊断系统"
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 900
    
    # 样式配置
    THEME = "Fusion"
    PRIMARY_COLOR = "#2E3440"
    SECONDARY_COLOR = "#3B4252"
    ACCENT_COLOR = "#5E81AC"
    
    # 图表配置
    PLOT_DPI = 100
    PLOT_FONTSIZE = 10
    PLOT_LINESTYLE = "-"
    PLOT_MARKER = "o"

# 日志配置
class LogConfig:
    """日志配置"""
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    LOG_FILE = LOGS_DIR / "app.log"
    LOG_ROTATION = "10 MB"
    LOG_RETENTION = "30 days"

# 系统配置
class SystemConfig:
    """系统配置"""
    # 系统信息
    APP_NAME = "ToolWearDiagnosis"
    VERSION = "1.0.0"
    AUTHOR = "xxx"
    
    # 性能配置
    MAX_WORKERS = 4     # 最大工作线程数
    MEMORY_LIMIT = "2GB"  # 内存限制
    
    # 调试配置
    DEBUG = False
    TEST_MODE = False

# 评估指标配置
class MetricsConfig:
    """评估指标配置"""
    # 主要评估指标
    PRIMARY_METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # 详细评估指标
    DETAILED_METRICS = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'specificity', 'auc', 'confusion_matrix'
    ]
    
    # 交叉验证配置
    CV_FOLDS = 5
    CV_RANDOM_STATE = 42
    
    # 测试配置
    TEST_SPLIT = 0.2
    VAL_SPLIT = 0.2
    RANDOM_STATE = 42

# 导出配置类
__all__ = [
    'DataConfig',
    'ModelConfig', 
    'PreprocessConfig',
    'GUIConfig',
    'LogConfig',
    'SystemConfig',
    'MetricsConfig',
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'LOGS_DIR',
    'CONFIG_DIR'
]
