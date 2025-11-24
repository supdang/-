# CPU训练指南

本指南说明了如何修改项目代码以确保在CPU上进行训练。

## 修改内容

### 1. 模型设备设置修改

在 `/workspace/src/models/lstm_model.py` 文件中，我修改了第168-169行的设备设置：

**原代码:**
```python
self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
```

**修改后:**
```python
# 强制使用CPU以确保兼容性
self.device = torch.device('cpu' if device is None or device == 'cpu' else device)
```

这个修改确保了即使系统有GPU可用，模型也会在CPU上运行，除非显式指定其他设备。

### 2. 使用方法

要使用修改后的代码进行CPU训练，只需按以下方式初始化分类器：

```python
from src.models.lstm_model import ToolWearClassifier

# 方法1: 显式指定CPU设备
classifier = ToolWearClassifier(device='cpu')

# 方法2: 不指定设备（现在默认使用CPU）
classifier = ToolWearClassifier()
```

### 3. 优化资源使用

为了在资源受限的环境中运行训练，可以使用以下环境变量：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### 4. 训练参数调整

在资源受限环境中，建议使用以下参数：

- 较小的批次大小 (如 batch_size=4 或 8)
- 较小的隐藏层大小 (如 hidden_size=32 或更小)
- 较少的训练轮数 (如 num_epochs=10-20)
- 较小的模型层数 (如 num_layers=1-2)

## 验证CPU使用

要验证模型是否在CPU上运行，可以检查日志输出：

```
使用设备: cpu
```

## 关键要点

1. **设备强制**: 修改后的代码优先使用CPU，即使GPU可用
2. **兼容性**: 代码仍然支持指定其他设备，但默认行为是使用CPU
3. **资源优化**: 通过调整模型大小和训练参数，可以在低资源环境中运行
4. **环境变量**: 设置适当的环境变量以限制线程使用，避免资源耗尽

## 示例代码

```python
# 确保使用CPU进行训练
classifier = ToolWearClassifier(device='cpu')

# 使用较小的批次和参数进行训练
history = classifier.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    batch_size=8,        # 小批次
    num_epochs=20,       # 较少轮数
    learning_rate=0.001
)
```

这样修改后，您的项目就可以在没有GPU的环境中正常进行训练了。