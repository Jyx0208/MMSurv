# 03 - 训练流程和配置

## 概述

本阶段将实现MMSurv项目的完整训练流水线，包括训练配置管理、损失函数实现、优化器配置、学习率调度、早停机制、交叉验证和模型保存等核心功能。

## 训练流程架构

### 核心组件
1. **训练配置管理** - 统一的参数配置系统
2. **数据加载器** - 高效的多模态数据加载
3. **损失函数** - 生存分析专用损失函数
4. **训练循环** - 主训练和验证循环
5. **早停机制** - 防止过拟合的早停策略
6. **交叉验证** - K折交叉验证支持
7. **模型保存** - 检查点和最佳模型保存
8. **日志记录** - 训练过程监控和记录

## 实现任务

### 任务1：实现训练配置系统

#### 1.1 创建配置管理模块
**文件位置**：`mmsurv/utils/config.py`

**实现要求**：
- 实现配置类`TrainingConfig`
- 支持YAML/JSON配置文件
- 参数验证和默认值设置
- 配置继承和覆盖机制

**关键功能**：
- 模型配置（模型类型、参数）
- 训练配置（学习率、批大小、轮数）
- 数据配置（数据路径、预处理参数）
- 优化器配置（优化器类型、参数）
- 调度器配置（学习率调度策略）

#### 1.2 创建配置文件模板
**文件位置**：`configs/`目录

**配置文件类型**：
- `configs/porpoise_default.yaml` - PORPOISE模型默认配置
- `configs/mcat_default.yaml` - MCAT模型默认配置
- `configs/motcat_default.yaml` - MOTCat模型默认配置
- `configs/amil_default.yaml` - AMIL模型默认配置
- `configs/deepset_default.yaml` - DeepSet模型默认配置
- `configs/cmta_default.yaml` - CMTA模型默认配置

#### 1.3 测试配置系统
**测试文件**：`tests/test_config.py`

**测试方法**：
```bash
# 创建配置测试脚本
cat > test_config_basic.py << 'EOF'
from mmsurv.utils.config import TrainingConfig
import yaml

# 测试配置加载
config_data = {
    'model': {
        'type': 'porpoise',
        'omic_input_dim': 50,
        'path_input_dim': 768,
        'n_classes': 4
    },
    'training': {
        'batch_size': 1,
        'learning_rate': 0.0001,
        'num_epochs': 100,
        'early_stopping_patience': 10
    }
}

# 保存测试配置
with open('test_config.yaml', 'w') as f:
    yaml.dump(config_data, f)

# 加载配置
config = TrainingConfig.from_yaml('test_config.yaml')
print(f"模型类型: {config.model.type}")
print(f"学习率: {config.training.learning_rate}")
print("配置系统测试通过")
EOF

python test_config_basic.py
```

### 任务2：实现损失函数

#### 2.1 创建生存分析损失函数
**文件位置**：`mmsurv/utils/loss_func.py`

**实现要求**：
- 实现`CrossEntropySurvLoss`类（交叉熵生存损失）
- 实现`NLLSurvLoss`类（负对数似然生存损失）
- 实现`CoxPHLoss`类（Cox比例风险损失）
- 支持离散时间生存分析
- 支持事件时间和删失处理

**关键功能**：
- 风险评分计算
- 生存概率估计
- 删失数据处理
- 梯度稳定性优化

#### 2.2 实现损失函数工具
**文件位置**：`mmsurv/utils/loss_utils.py`

**实现要求**：
- 实现`ce_loss()`函数（交叉熵损失）
- 实现`nll_loss()`函数（负对数似然损失）
- 实现`cox_loss()`函数（Cox损失）
- 实现损失权重计算
- 实现损失平滑和正则化

#### 2.3 测试损失函数
**测试文件**：`tests/test_loss_functions.py`

**测试方法**：
```bash
# 测试损失函数
python -c "
import torch
from mmsurv.utils.loss_func import CrossEntropySurvLoss, NLLSurvLoss

# 创建测试数据
batch_size = 4
n_classes = 4
hazards = torch.randn(batch_size, n_classes)  # 风险预测
S = torch.randn(batch_size, n_classes)  # 生存概率
Y = torch.randint(0, n_classes, (batch_size,))  # 真实标签
c = torch.randint(0, 2, (batch_size,))  # 删失指示器

# 测试交叉熵损失
ce_loss_fn = CrossEntropySurvLoss()
ce_loss = ce_loss_fn(hazards, S, Y, c)
print(f'交叉熵损失: {ce_loss.item():.4f}')

# 测试NLL损失
nll_loss_fn = NLLSurvLoss()
nll_loss = nll_loss_fn(hazards, S, Y, c)
print(f'NLL损失: {nll_loss.item():.4f}')

print('损失函数测试通过')
"
```

### 任务3：实现训练核心逻辑

#### 3.1 实现训练工具模块
**文件位置**：`mmsurv/utils/core_utils.py`

**实现要求**：
- 实现`EarlyStopping`类（早停机制）
- 实现`train()`函数（主训练函数）
- 实现`loop_survival()`函数（训练循环）
- 实现数据分割和交叉验证支持
- 实现模型保存和加载

**关键功能**：
- 训练和验证循环
- 损失计算和反向传播
- 梯度裁剪和优化
- 性能指标计算
- 检查点保存

#### 3.2 实现早停机制
**实现要求**：
- 监控验证损失或性能指标
- 可配置的耐心参数
- 最佳模型状态保存
- 训练恢复支持

#### 3.3 测试训练核心逻辑
**测试方法**：
```bash
# 测试早停机制
python -c "
from mmsurv.utils.core_utils import EarlyStopping
import torch

# 创建早停实例
early_stopping = EarlyStopping(patience=3, min_delta=0.001)

# 模拟训练过程
val_losses = [1.0, 0.8, 0.7, 0.72, 0.71, 0.70, 0.69]

for epoch, val_loss in enumerate(val_losses):
    should_stop = early_stopping(val_loss)
    print(f'Epoch {epoch}: val_loss={val_loss:.3f}, should_stop={should_stop}')
    if should_stop:
        print(f'早停触发，最佳损失: {early_stopping.best_score:.3f}')
        break

print('早停机制测试通过')
"
```

### 任务4：实现数据加载和预处理

#### 4.1 实现数据加载器
**文件位置**：`mmsurv/utils/data_utils.py`

**实现要求**：
- 实现`get_split_loader()`函数
- 支持多模态数据加载
- 实现数据增强和预处理
- 支持批处理和并行加载
- 内存优化和缓存机制

**关键功能**：
- 训练/验证/测试数据分割
- 批量数据加载
- 数据标准化和归一化
- 缺失值处理
- 数据平衡和采样

#### 4.2 实现数据分割工具
**实现要求**：
- K折交叉验证分割
- 分层采样支持
- 患者级别分割
- 数据泄露防护

#### 4.3 测试数据加载
**测试方法**：
```bash
# 测试数据加载器
python -c "
from mmsurv.utils.data_utils import get_split_loader
from mmsurv.datasets.dataset_survival import Generic_WSI_Survival_Dataset
import torch

# 创建模拟数据集
class MockDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'path_features': torch.randn(50, 768),
            'omic_features': torch.randn(50),
            'label': torch.randint(0, 4, (1,)).item(),
            'event_time': torch.rand(1).item(),
            'censorship': torch.randint(0, 2, (1,)).item()
        }

# 测试数据加载
dataset = MockDataset(100)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

batch = next(iter(loader))
print(f'批次大小: {len(batch[\"label\"])}')
print(f'病理特征形状: {batch[\"path_features\"].shape}')
print(f'基因组特征形状: {batch[\"omic_features\"].shape}')
print('数据加载器测试通过')
"
```

### 任务5：实现优化器和调度器

#### 5.1 实现优化器配置
**文件位置**：`mmsurv/utils/optimizer_utils.py`

**实现要求**：
- 实现`get_optimizer()`函数
- 支持多种优化器（Adam、SGD、AdamW等）
- 实现参数组分离（不同层不同学习率）
- 权重衰减和正则化配置

**支持的优化器**：
- Adam优化器
- AdamW优化器
- SGD优化器
- RMSprop优化器

#### 5.2 实现学习率调度器
**实现要求**：
- 实现`get_scheduler()`函数
- 支持多种调度策略
- 自适应学习率调整
- 预热和衰减策略

**支持的调度器**：
- StepLR（阶梯衰减）
- ExponentialLR（指数衰减）
- CosineAnnealingLR（余弦退火）
- ReduceLROnPlateau（自适应衰减）

#### 5.3 测试优化器和调度器
**测试方法**：
```bash
# 测试优化器配置
python -c "
from mmsurv.utils.optimizer_utils import get_optimizer, get_scheduler
from mmsurv.models.model_porpoise import PorpoiseMMF
import torch

# 创建模型
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)

# 测试优化器
optimizer = get_optimizer(
    model.parameters(),
    optimizer_type='adam',
    learning_rate=0.0001,
    weight_decay=1e-5
)
print(f'优化器类型: {type(optimizer).__name__}')
print(f'学习率: {optimizer.param_groups[0][\"lr\"]}')

# 测试调度器
scheduler = get_scheduler(
    optimizer,
    scheduler_type='step',
    step_size=30,
    gamma=0.1
)
print(f'调度器类型: {type(scheduler).__name__}')
print('优化器和调度器测试通过')
"
```

### 任务6：实现交叉验证框架

#### 6.1 实现交叉验证管理器
**文件位置**：`mmsurv/utils/cv_utils.py`

**实现要求**：
- 实现`CrossValidator`类
- 支持K折交叉验证
- 支持分层交叉验证
- 实现折叠结果聚合
- 支持并行训练（可选）

**关键功能**：
- 数据分割策略
- 折叠训练管理
- 结果收集和统计
- 最佳模型选择

#### 6.2 实现验证指标计算
**实现要求**：
- 实现C-Index计算
- 实现生存曲线评估
- 实现风险分层评估
- 统计显著性测试

#### 6.3 测试交叉验证
**测试方法**：
```bash
# 测试交叉验证框架
python -c "
from mmsurv.utils.cv_utils import CrossValidator
from sklearn.model_selection import KFold
import numpy as np

# 创建模拟数据
n_samples = 100
X = np.random.randn(n_samples, 50)
y = np.random.randint(0, 4, n_samples)

# 创建交叉验证器
cv = CrossValidator(n_splits=5, random_state=42)
folds = cv.split(X, y)

print(f'交叉验证折数: {len(list(folds))}')
print('交叉验证框架测试通过')
"
```

### 任务7：实现模型保存和加载

#### 7.1 实现检查点管理
**文件位置**：`mmsurv/utils/checkpoint_utils.py`

**实现要求**：
- 实现`save_checkpoint()`函数
- 实现`load_checkpoint()`函数
- 支持模型状态、优化器状态保存
- 实现最佳模型自动保存
- 支持训练恢复

**保存内容**：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 训练轮数和损失
- 随机数种子状态

#### 7.2 实现模型导出
**实现要求**：
- 支持PyTorch格式导出
- 支持ONNX格式导出（可选）
- 模型压缩和量化（可选）
- 推理优化

#### 7.3 测试模型保存和加载
**测试方法**：
```bash
# 测试模型保存和加载
python -c "
from mmsurv.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from mmsurv.models.model_porpoise import PorpoiseMMF
import torch
import os

# 创建模型和优化器
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 保存检查点
checkpoint_path = 'test_checkpoint.pth'
save_checkpoint({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 10,
    'loss': 0.5
}, checkpoint_path)

print(f'检查点已保存: {os.path.exists(checkpoint_path)}')

# 加载检查点
checkpoint = load_checkpoint(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f'加载轮数: {checkpoint[\"epoch\"]}')
print(f'加载损失: {checkpoint[\"loss\"]}')
print('模型保存和加载测试通过')

# 清理测试文件
os.remove(checkpoint_path)
"
```

### 任务8：实现训练监控和日志

#### 8.1 实现日志记录系统
**文件位置**：`mmsurv/utils/logger.py`

**实现要求**：
- 实现`Logger`类
- 支持多级别日志（DEBUG、INFO、WARNING、ERROR）
- 支持文件和控制台输出
- 实现训练指标记录
- 支持TensorBoard集成（可选）

**记录内容**：
- 训练和验证损失
- 性能指标（C-Index等）
- 学习率变化
- 模型参数统计
- 训练时间和资源使用

#### 8.2 实现进度监控
**实现要求**：
- 实现训练进度条
- 实时损失显示
- ETA估算
- 资源使用监控

#### 8.3 测试日志系统
**测试方法**：
```bash
# 测试日志系统
python -c "
from mmsurv.utils.logger import Logger
import os

# 创建日志器
logger = Logger('test_training', log_dir='./logs')

# 记录训练信息
logger.info('开始训练')
logger.log_metrics({
    'epoch': 1,
    'train_loss': 0.8,
    'val_loss': 0.7,
    'c_index': 0.65
})
logger.warning('学习率过高')
logger.info('训练完成')

# 检查日志文件
log_files = os.listdir('./logs')
print(f'日志文件数量: {len(log_files)}')
print('日志系统测试通过')
"
```

### 任务9：实现完整训练脚本

#### 9.1 创建主训练脚本
**文件位置**：`scripts/train.py`

**实现要求**：
- 命令行参数解析
- 配置文件加载
- 数据集初始化
- 模型创建和配置
- 训练循环执行
- 结果保存和报告

**命令行接口**：
```bash
python scripts/train.py \
    --config configs/porpoise_default.yaml \
    --data_dir /path/to/data \
    --output_dir ./results \
    --gpu 0 \
    --seed 42
```

#### 9.2 创建分布式训练脚本（可选）
**文件位置**：`scripts/train_distributed.py`

**实现要求**：
- 多GPU训练支持
- 数据并行处理
- 梯度同步
- 分布式保存

#### 9.3 测试完整训练流程
**测试方法**：
```bash
# 创建最小训练测试
cat > test_training_pipeline.py << 'EOF'
import torch
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.utils.core_utils import train, EarlyStopping
from mmsurv.utils.loss_func import CrossEntropySurvLoss
from torch.utils.data import DataLoader, TensorDataset

# 创建模拟数据
n_samples = 50
path_features = torch.randn(n_samples, 100, 768)
omic_features = torch.randn(n_samples, 50)
labels = torch.randint(0, 4, (n_samples,))
event_times = torch.rand(n_samples)
censorship = torch.randint(0, 2, (n_samples,))

# 创建数据集和加载器
dataset = TensorDataset(path_features, omic_features, labels, event_times, censorship)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

# 创建模型
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = CrossEntropySurvLoss()
early_stopping = EarlyStopping(patience=5)

# 简化训练循环测试
model.train()
for epoch in range(3):  # 只训练3轮测试
    total_loss = 0
    for batch_idx, (path_feat, omic_feat, label, event_time, censor) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 前向传播
        output = model(h_path=path_feat, h_omic=omic_feat)
        
        # 计算损失（简化版）
        loss = torch.nn.CrossEntropyLoss()(output, label)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx >= 2:  # 只处理前3个批次
            break
    
    avg_loss = total_loss / min(3, len(train_loader))
    print(f'Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}')

print('训练流程测试通过')
EOF

python test_training_pipeline.py
```

## 配置文件示例

### PORPOISE模型配置
**文件位置**：`configs/porpoise_default.yaml`

```yaml
# 模型配置
model:
  type: "porpoise"
  omic_input_dim: 50
  path_input_dim: 768
  n_classes: 4
  dropout: 0.25
  fusion_type: "bilinear"

# 训练配置
training:
  batch_size: 1
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 1e-5
  gradient_clip_norm: 1.0
  
# 早停配置
early_stopping:
  patience: 10
  min_delta: 0.001
  monitor: "val_loss"
  
# 优化器配置
optimizer:
  type: "adam"
  betas: [0.9, 0.999]
  eps: 1e-8
  
# 学习率调度
scheduler:
  type: "step"
  step_size: 30
  gamma: 0.1
  
# 数据配置
data:
  num_workers: 4
  pin_memory: true
  drop_last: false
  
# 交叉验证
cross_validation:
  n_splits: 5
  shuffle: true
  random_state: 42
```

## 测试和验证

### 单元测试

#### 运行训练组件测试
```bash
# 运行所有训练相关测试
python -m pytest tests/test_training*.py -v

# 运行特定组件测试
python -m pytest tests/test_config.py -v
python -m pytest tests/test_loss_functions.py -v
python -m pytest tests/test_core_utils.py -v
```

### 集成测试

#### 测试1：端到端训练流程
```bash
# 创建端到端训练测试
cat > test_e2e_training.py << 'EOF'
import torch
import tempfile
import os
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.utils.config import TrainingConfig
from mmsurv.utils.core_utils import train
from torch.utils.data import DataLoader, TensorDataset

# 创建临时目录
with tempfile.TemporaryDirectory() as temp_dir:
    # 创建模拟数据
    n_samples = 20
    path_features = torch.randn(n_samples, 50, 768)
    omic_features = torch.randn(n_samples, 50)
    labels = torch.randint(0, 4, (n_samples,))
    
    dataset = TensorDataset(path_features, omic_features, labels)
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)
    
    # 创建模型
    model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
    
    # 创建配置
    config = {
        'num_epochs': 2,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'save_dir': temp_dir
    }
    
    # 运行训练
    try:
        results = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        print(f"训练完成，最终损失: {results['final_loss']:.4f}")
        print("端到端训练测试通过")
    except Exception as e:
        print(f"训练测试失败: {e}")
EOF

python test_e2e_training.py
```

#### 测试2：交叉验证流程
```bash
# 测试交叉验证
python -c "
from mmsurv.utils.cv_utils import CrossValidator
from mmsurv.models.model_porpoise import PorpoiseMMF
import torch
import numpy as np

# 创建模拟数据
n_samples = 50
X = torch.randn(n_samples, 100, 768)  # 病理特征
y = torch.randint(0, 4, (n_samples,))  # 标签

# 创建交叉验证器
cv = CrossValidator(n_splits=3, random_state=42)

# 运行交叉验证
results = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f'Fold {fold+1}: 训练样本 {len(train_idx)}, 验证样本 {len(val_idx)}')
    
    # 模拟训练结果
    fold_result = {
        'fold': fold,
        'train_loss': np.random.uniform(0.5, 1.0),
        'val_loss': np.random.uniform(0.6, 1.1),
        'c_index': np.random.uniform(0.6, 0.8)
    }
    results.append(fold_result)

# 计算平均结果
avg_c_index = np.mean([r['c_index'] for r in results])
std_c_index = np.std([r['c_index'] for r in results])

print(f'平均C-Index: {avg_c_index:.3f} ± {std_c_index:.3f}')
print('交叉验证测试通过')
"
```

### 性能测试

#### 训练速度基准测试
```bash
# 创建训练速度测试
cat > benchmark_training_speed.py << 'EOF'
import torch
import time
from mmsurv.models.model_porpoise import PorpoiseMMF
from torch.utils.data import DataLoader, TensorDataset

def benchmark_training_speed(model, data_loader, num_epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        for batch_idx, (path_feat, omic_feat, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            
            # 前向传播
            output = model(h_path=path_feat, h_omic=omic_feat)
            loss = loss_fn(output, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if batch_idx >= 10:  # 限制批次数量
                break
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return total_time

# 创建测试数据
n_samples = 100
path_features = torch.randn(n_samples, 100, 768)
omic_features = torch.randn(n_samples, 50)
labels = torch.randint(0, 4, (n_samples,))

dataset = TensorDataset(path_features, omic_features, labels)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 创建模型
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)

# 运行基准测试
training_time = benchmark_training_speed(model, data_loader)
print(f"训练时间: {training_time:.2f}秒")
print(f"每轮平均时间: {training_time/5:.2f}秒")
print("训练速度基准测试完成")
EOF

python benchmark_training_speed.py
```

## 故障排除

### 常见问题

#### 问题1：内存不足
**症状**：CUDA out of memory或系统内存不足
**解决方案**：
1. 减少batch_size
2. 使用梯度累积
3. 启用混合精度训练
4. 优化数据加载器

#### 问题2：训练不收敛
**症状**：损失不下降或震荡
**解决方案**：
1. 调整学习率
2. 检查损失函数实现
3. 验证数据预处理
4. 使用梯度裁剪

#### 问题3：过拟合
**症状**：训练损失下降但验证损失上升
**解决方案**：
1. 增加正则化
2. 使用dropout
3. 减少模型复杂度
4. 增加训练数据

#### 问题4：训练速度慢
**症状**：每轮训练时间过长
**解决方案**：
1. 优化数据加载
2. 使用多进程
3. 启用混合精度
4. 优化模型架构

## 验证清单

完成本阶段后，请确认以下项目：

- [ ] 训练配置系统实现完成
- [ ] 损失函数实现并测试通过
- [ ] 训练核心逻辑实现完成
- [ ] 数据加载和预处理实现完成
- [ ] 优化器和调度器配置完成
- [ ] 交叉验证框架实现完成
- [ ] 模型保存和加载功能完成
- [ ] 训练监控和日志系统完成
- [ ] 完整训练脚本实现完成
- [ ] 端到端训练测试通过
- [ ] 交叉验证测试通过
- [ ] 性能基准测试完成

## 下一步

训练流程实现完成后，请继续阅读 [04-evaluation-visualization.md](./04-evaluation-visualization.md) 进行评估和可视化实现。