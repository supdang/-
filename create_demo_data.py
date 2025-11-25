#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示数据生成脚本
生成用于演示的刀具磨损数据
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

from src.config.config import DataConfig, DATA_DIR
from src.utils.logger import setup_logger

logger = setup_logger()

def generate_demo_data():
    """生成演示数据"""
    
    logger.info("开始生成演示数据...")
    
    # 参数设置
    sample_rate = DataConfig.SAMPLE_RATE
    duration = 60  # 60秒数据
    n_samples = int(sample_rate * duration)
    n_channels = len(DataConfig.SENSOR_CHANNELS)
    
    # 初始化数据数组
    sensor_data = np.zeros((n_channels, n_samples))
    
    # 时间轴
    time_axis = np.linspace(0, duration, n_samples)
    
    # 生成每个通道的信号
    for channel_idx, channel_name in enumerate(DataConfig.SENSOR_CHANNELS.keys()):
        logger.info(f"生成通道 {channel_name} 的数据...")
        
        if 'force' in channel_name:
            # 切削力信号 - 包含基频和谐波
            base_freq = 173  # Hz (主轴转速10400 rpm / 60)
            
            # 基础信号
            base_signal = 2.0 * np.sin(2 * np.pi * base_freq * time_axis)
            
            # 添加谐波
            harmonic2 = 0.5 * np.sin(2 * np.pi * base_freq * 2 * time_axis)
            harmonic3 = 0.2 * np.sin(2 * np.pi * base_freq * 3 * time_axis)
            
            # 添加随机噪声
            noise = np.random.normal(0, 0.3, n_samples)
            
            # 组合信号
            sensor_data[channel_idx] = base_signal + harmonic2 + harmonic3 + noise
            
        elif 'vibration' in channel_name:
            # 振动信号 - 包含多个频率成分
            freqs = [100, 200, 400, 800]
            amplitudes = [1.0, 0.6, 0.4, 0.2]
            
            signal_combined = np.zeros(n_samples)
            for freq, amp in zip(freqs, amplitudes):
                signal_combined += amp * np.sin(2 * np.pi * freq * time_axis)
            
            # 添加噪声
            noise = np.random.normal(0, 0.2, n_samples)
            sensor_data[channel_idx] = signal_combined + noise
            
        elif 'ae' in channel_name:
            # 声发射信号 - 高频随机脉冲
            # 生成随机脉冲
            n_pulses = 1000
            pulse_positions = np.random.randint(0, n_samples, n_pulses)
            pulse_amplitudes = np.random.exponential(0.5, n_pulses)
            
            ae_signal = np.zeros(n_samples)
            for pos, amp in zip(pulse_positions, pulse_amplitudes):
                if pos < n_samples - 100:
                    pulse = amp * np.exp(-np.linspace(0, 5, 100))
                    ae_signal[pos:pos+100] += pulse
            
            # 添加高频噪声
            high_freq_noise = np.random.normal(0, 0.1, n_samples)
            sensor_data[channel_idx] = ae_signal + high_freq_noise
    
    # 模拟磨损过程 - 信号特征随时间变化
    logger.info("模拟磨损过程...")
    
    # 磨损进程（从0到0.4mm）
    wear_progression = np.linspace(0, 0.4, n_samples)
    
    # 根据磨损量调整信号特征
    for channel_idx, channel_name in enumerate(DataConfig.SENSOR_CHANNELS.keys()):
        # 磨损导致信号幅度增加
        amplitude_factor = 1 + wear_progression * 2.0
        sensor_data[channel_idx] *= amplitude_factor
        
        # 磨损导致噪声增加
        noise_level = 0.1 + wear_progression * 0.3
        noise = np.random.normal(0, noise_level, n_samples)
        sensor_data[channel_idx] += noise
        
        # 磨损导致频率成分变化
        if 'force' in channel_name:
            # 高频成分增加
            high_freq_component = 0.5 * wear_progression * np.sin(2 * np.pi * 1000 * time_axis)
            sensor_data[channel_idx] += high_freq_component
    
    # 生成磨损标签
    logger.info("生成磨损标签...")
    
    # 根据磨损量创建状态标签
    labels = np.zeros(n_samples, dtype=int)
    
    thresholds = DataConfig.WEAR_THRESHOLDS
    labels[wear_progression >= thresholds[2]] = 3  # 失效状态
    labels[(wear_progression >= thresholds[1]) & (wear_progression < thresholds[2])] = 2  # 后期磨损
    labels[(wear_progression >= thresholds[0]) & (wear_progression < thresholds[1])] = 1  # 正常磨损
    labels[wear_progression < thresholds[0]] = 0  # 初期磨损
    
    # 保存数据
    logger.info("保存数据...")
    
    # 创建数据目录
    DATA_DIR.mkdir(exist_ok=True)
    
    # 保存为CSV格式
    df = pd.DataFrame()
    
    # 添加传感器数据
    for i, channel_name in enumerate(DataConfig.SENSOR_CHANNELS.keys()):
        df[f'{channel_name}'] = sensor_data[i]
    
    # 添加磨损标签
    df['wear_state'] = labels
    df['wear_value'] = wear_progression
    df['time'] = time_axis
    
    # 保存CSV文件
    csv_path = DATA_DIR / "demo_tool_wear_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV数据已保存到: {csv_path}")
    
    # 保存为NumPy格式
    npz_path = DATA_DIR / "demo_tool_wear_data.npz"
    np.savez(npz_path,
             sensor_data=sensor_data,
             labels=labels,
             wear_values=wear_progression,
             time_axis=time_axis)
    logger.info(f"NumPy数据已保存到: {npz_path}")
    
    # 创建数据信息文件
    info_path = DATA_DIR / "demo_data_info.txt"
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("演示数据信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write(f"采样频率: {sample_rate} Hz\n")
        f.write(f"数据时长: {duration} 秒\n")
        f.write(f"样本数量: {n_samples}\n")
        f.write(f"通道数量: {n_channels}\n")
        f.write(f"传感器通道: {list(DataConfig.SENSOR_CHANNELS.keys())}\n")
        f.write(f"数据形状: {sensor_data.shape}\n")
        f.write(f"标签形状: {labels.shape}\n\n")
        
        f.write("标签分布:\n")
        label_counts = np.bincount(labels)
        state_names = ['初期磨损', '正常磨损', '后期磨损', '失效状态']
        for i, (count, name) in enumerate(zip(label_counts, state_names)):
            f.write(f"  {name}: {count} 样本 ({count/len(labels)*100:.1f}%)\n")
    
    logger.info(f"数据信息已保存到: {info_path}")
    
    # 生成可视化图表
    logger.info("生成可视化图表...")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('演示数据可视化', fontsize=16, fontweight='bold')
    
    # 1. 时域信号示例
    time_show = min(5, duration)  # 显示前5秒
    samples_show = int(sample_rate * time_show)
    time_axis_show = time_axis[:samples_show]
    
    for i, channel_name in enumerate(list(DataConfig.SENSOR_CHANNELS.keys())[:3]):
        axes[i, 0].plot(time_axis_show, sensor_data[i, :samples_show])
        axes[i, 0].set_title(f'{channel_name} 时域信号')
        axes[i, 0].set_xlabel('时间 (s)')
        axes[i, 0].set_ylabel('幅值')
        axes[i, 0].grid(True)
        axes[i, 0].legend([channel_name])
    
    # 2. 频域信号示例
    for i, channel_name in enumerate(list(DataConfig.SENSOR_CHANNELS.keys())[:3]):
        # 计算FFT
        fft_values = np.fft.fft(sensor_data[i])
        freqs = np.fft.fftfreq(len(sensor_data[i]), 1/sample_rate)
        
        # 只显示正频率部分
        positive_freq_idx = freqs > 0
        axes[i, 1].plot(freqs[positive_freq_idx], 
                       np.abs(fft_values[positive_freq_idx]))
        axes[i, 1].set_title(f'{channel_name} 频域信号')
        axes[i, 1].set_xlabel('频率 (Hz)')
        axes[i, 1].set_ylabel('幅值')
        axes[i, 1].set_xlim(0, 5000)  # 限制频率范围
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = DATA_DIR / "demo_data_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"可视化图表已保存到: {plot_path}")
    
    # 生成磨损过程图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('磨损过程分析', fontsize=16, fontweight='bold')
    
    # 1. 磨损值随时间变化
    axes[0, 0].plot(time_axis[::1000], wear_progression[::1000])  # 降采样显示
    axes[0, 0].set_title('磨损值变化')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('磨损值 (mm)')
    axes[0, 0].grid(True)
    
    # 添加阈值线
    for threshold in thresholds:
        axes[0, 0].axhline(y=threshold, color='r', linestyle='--', alpha=0.7)
    
    # 2. 标签分布
    label_counts = np.bincount(labels)
    state_names = ['初期磨损', '正常磨损', '后期磨损', '失效状态']
    colors = ['green', 'blue', 'orange', 'red']
    
    bars = axes[0, 1].bar(state_names, label_counts, color=colors, alpha=0.7)
    axes[0, 1].set_title('磨损状态分布')
    axes[0, 1].set_xlabel('磨损状态')
    axes[0, 1].set_ylabel('样本数量')
    
    # 添加数值标签
    for bar, count in zip(bars, label_counts):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. 信号能量随时间变化
    window_size = 1000
    energy_over_time = []
    time_windows = []
    
    for i in range(0, len(sensor_data[0]) - window_size, window_size):
        window = sensor_data[0, i:i+window_size]
        energy = np.sum(window**2) / window_size
        energy_over_time.append(energy)
        time_windows.append(time_axis[i + window_size//2])
    
    axes[1, 0].plot(time_windows, energy_over_time)
    axes[1, 0].set_title('信号能量变化')
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('能量')
    axes[1, 0].grid(True)
    
    # 4. 频谱变化（使用短时傅里叶变换）
    from scipy import signal as scipy_signal
    
    # 计算STFT
    f, t, Zxx = scipy_signal.stft(sensor_data[0], fs=sample_rate, 
                                   nperseg=1024, noverlap=512)
    
    # 只显示低频部分
    freq_mask = f < 2000
    im = axes[1, 1].pcolormesh(t, f[freq_mask], np.abs(Zxx[freq_mask]), 
                              shading='gouraud', cmap='viridis')
    axes[1, 1].set_title('时频分析')
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('频率 (Hz)')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('幅值')
    
    plt.tight_layout()
    
    # 保存磨损分析图表
    wear_plot_path = DATA_DIR / "demo_wear_analysis.png"
    plt.savefig(wear_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"磨损分析图表已保存到: {wear_plot_path}")
    
    logger.info("演示数据生成完成！")
    logger.info(f"生成的文件:")
    logger.info(f"  - CSV数据: {csv_path}")
    logger.info(f"  - NumPy数据: {npz_path}")
    logger.info(f"  - 数据信息: {info_path}")
    logger.info(f"  - 可视化图表: {plot_path}")
    logger.info(f"  - 磨损分析图表: {wear_plot_path}")
    
    return {
        'sensor_data': sensor_data,
        'labels': labels,
        'wear_values': wear_progression,
        'time_axis': time_axis,
        'file_paths': {
            'csv': csv_path,
            'npz': npz_path,
            'info': info_path,
            'plot': plot_path,
            'wear_plot': wear_plot_path
        }
    }

def main():
    """主函数"""
    try:
        demo_data = generate_demo_data()
        print("演示数据生成成功！")
        
        # 显示数据基本信息
        sensor_data = demo_data['sensor_data']
        labels = demo_data['labels']
        
        print(f"\n数据基本信息:")
        print(f"传感器数据形状: {sensor_data.shape}")
        print(f"标签数据形状: {labels.shape}")
        print(f"传感器通道数: {sensor_data.shape[0]}")
        print(f"采样点数: {sensor_data.shape[1]}")
        
        print(f"\n标签分布:")
        state_names = ['初期磨损', '正常磨损', '后期磨损', '失效状态']
        label_counts = np.bincount(labels)
        for i, (name, count) in enumerate(zip(state_names, label_counts)):
            percentage = count / len(labels) * 100
            print(f"  {name}: {count} 样本 ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"演示数据生成失败: {e}")
        logger.error(f"演示数据生成失败: {e}")

if __name__ == "__main__":
    main()
