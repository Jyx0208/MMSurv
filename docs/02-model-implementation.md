# 02 - 模型架构实现

## 概述

本阶段将实现MMSurv项目支持的6种深度学习模型架构。每种模型都有其独特的设计理念和适用场景，需要分别实现并确保它们能够处理多模态生存预测任务。

## 模型架构概览

### 1. PORPOISE - 多模态融合模型
- **特点**：参数效率高，使用低秩双线性融合
- **适用场景**：资源受限环境，快速原型开发
- **核心技术**：门控多模态单元、低秩分解

### 2. MCAT - 多模态协同注意力Transformer
- **特点**：基于Transformer的协同注意力机制
- **适用场景**：高精度要求的研究环境
- **核心技术**：多头注意力、跨模态交互

### 3. MOTCat - 最优传输协同注意力
- **特点**：基于最优传输理论的模态对齐
- **适用场景**：前沿研究，多模态对齐要求高
- **核心技术**：最优传输、Wasserstein距离

### 4. AMIL - 注意力多实例学习
- **特点**：经典的注意力MIL架构
- **适用场景**：传统MIL任务，需要可解释性
- **核心技术**：注意力权重、实例聚合

### 5. DeepSet - 集合函数深度学习
- **特点**：基于集合函数的理论基础
- **适用场景**：需要数学可解释性的场景
- **核心技术**：置换不变性、集合函数

### 6. DeepAttnMISL - 深度注意力多实例学习
- **特点**：深度注意力机制的MIL扩展
- **适用场景**：复杂的多实例学习任务
- **核心技术**：深度注意力、多层聚合

## 实现任务

### 任务1：实现基础模型工具

#### 1.1 创建模型工具模块
**文件位置**：`mmsurv/models/model_utils.py`

**实现要求**：
- 实现通用的网络层构建函数
- 实现权重初始化工具
- 实现模型参数统计工具
- 实现激活函数和正则化工具

**关键功能**：
- `SNN_Block()` - 标准神经网络块
- `MLP_Block()` - 多层感知机块
- `Attn_Net()` - 基础注意力网络
- `Attn_Net_Gated()` - 门控注意力网络
- `initialize_weights()` - 权重初始化

#### 1.2 测试基础工具
**测试文件**：`tests/test_model_utils.py`

**测试内容**：
- 网络层构建测试
- 权重初始化测试
- 前向传播测试
- 参数统计测试

### 任务2：实现PORPOISE模型

#### 2.1 实现PORPOISE核心组件
**文件位置**：`mmsurv/models/model_porpoise.py`

**实现要求**：
- 实现`LRBilinearFusion`类（低秩双线性融合）
- 实现`BilinearFusion`类（标准双线性融合）
- 实现`PorpoiseAMIL`类（单模态版本）
- 实现`PorpoiseMMF`类（多模态版本）

**关键组件**：
- 门控多模态单元
- 低秩分解融合层
- 注意力聚合机制
- 生存预测头

#### 2.2 测试PORPOISE模型
**测试文件**：`tests/test_porpoise.py`

**测试内容**：
- 模型初始化测试
- 前向传播测试
- 多模态融合测试
- 输出维度验证

**测试方法**：
```bash
# 创建PORPOISE测试脚本
cat > test_porpoise_basic.py << 'EOF'
import torch
from mmsurv.models.model_porpoise import PorpoiseMMF

# 测试模型初始化
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 测试前向传播
batch_size = 1
path_features = torch.randn(batch_size, 100, 768)  # 100个patch，768维特征
omic_features = torch.randn(batch_size, 50)  # 50维基因组特征

output = model(h_path=path_features, h_omic=omic_features)
print(f"输出形状: {output.shape}")
print("PORPOISE模型测试通过")
EOF

python test_porpoise_basic.py
```

### 任务3：实现MCAT模型

#### 3.1 实现MCAT核心组件
**文件位置**：`mmsurv/models/model_coattn.py`

**实现要求**：
- 实现`MultiheadAttention`类
- 实现`TransLayer`类（Transformer层）
- 实现`MCAT_Surv`类（主模型）
- 实现跨模态注意力机制

**关键组件**：
- 多头自注意力
- 跨模态协同注意力
- 位置编码
- Transformer编码器

#### 3.2 测试MCAT模型
**测试文件**：`tests/test_mcat.py`

**测试内容**：
- Transformer层测试
- 多头注意力测试
- 跨模态交互测试
- 完整模型测试

### 任务4：实现MOTCat模型

#### 4.1 实现MOTCat核心组件
**文件位置**：`mmsurv/models/model_motcat.py`

**实现要求**：
- 实现最优传输计算模块
- 实现`MOTCAT_Surv`类
- 实现Wasserstein距离计算
- 实现跨模态对齐机制

**关键组件**：
- 最优传输求解器
- 跨模态对齐层
- 协同注意力机制
- 微批处理支持

#### 4.2 实现MOTCat工具
**文件位置**：`mmsurv/models/cmta_util.py`

**实现要求**：
- 实现最优传输工具函数
- 实现距离计算函数
- 实现批处理优化

#### 4.3 测试MOTCat模型
**测试文件**：`tests/test_motcat.py`

**测试内容**：
- 最优传输计算测试
- 模态对齐测试
- 微批处理测试
- 完整模型测试

**测试方法**：
```bash
# 测试最优传输功能
python -c "
import torch
from mmsurv.models.model_motcat import MOTCAT_Surv

# 创建测试数据
path_features = torch.randn(1, 100, 768)
omic_features = torch.randn(1, 50)

# 测试模型
model = MOTCAT_Surv(omic_input_dim=50, path_input_dim=768, n_classes=4)
output = model(h_path=path_features, h_omic=omic_features)
print(f'MOTCat输出形状: {output.shape}')
"
```

### 任务5：实现MIL模型系列

#### 5.1 实现基础MIL模型
**文件位置**：`mmsurv/models/model_set_mil.py`

**实现要求**：
- 实现`MIL_Sum_FC_surv`类（DeepSet模型）
- 实现`MIL_Attention_FC_surv`类（AMIL模型）
- 实现`MIL_Cluster_FC_surv`类（聚类MIL模型）
- 支持不同的聚合策略

**关键组件**：
- 实例级特征提取
- 注意力权重计算
- 包级特征聚合
- 生存预测层

#### 5.2 实现基因组模型
**文件位置**：`mmsurv/models/model_genomic.py`

**实现要求**：
- 实现`SNN`类（基因组神经网络）
- 支持多种基因组数据类型
- 实现特征选择和降维
- 支持正则化和dropout

#### 5.3 测试MIL模型
**测试文件**：`tests/test_mil_models.py`

**测试内容**：
- DeepSet模型测试
- AMIL模型测试
- 聚类MIL测试
- 基因组模型测试

### 任务6：实现CMTA模型

#### 6.1 实现CMTA核心组件
**文件位置**：`mmsurv/models/model_cmta.py`

**实现要求**：
- 实现`CMTA`类（跨模态Transformer注意力）
- 实现多尺度特征融合
- 实现自适应权重机制
- 支持动态模态选择

#### 6.2 测试CMTA模型
**测试文件**：`tests/test_cmta.py`

**测试内容**：
- 跨模态注意力测试
- 多尺度融合测试
- 自适应权重测试
- 完整模型测试

## 模型集成和测试

### 任务7：创建模型工厂

#### 7.1 实现模型选择器
**文件位置**：`mmsurv/models/__init__.py`

**实现要求**：
- 实现模型工厂函数
- 支持动态模型选择
- 统一模型接口
- 参数验证和错误处理

**关键功能**：
```python
def get_model(model_type, **kwargs):
    """根据模型类型返回相应的模型实例"""
    pass
```

#### 7.2 测试模型工厂
**测试方法**：
```bash
# 测试所有模型的初始化
python -c "
from mmsurv.models import get_model

models = ['porpoise', 'mcat', 'motcat', 'amil', 'deepset', 'cmta']
for model_name in models:
    try:
        model = get_model(model_name, omic_input_dim=50, path_input_dim=768, n_classes=4)
        print(f'{model_name}: 初始化成功')
    except Exception as e:
        print(f'{model_name}: 初始化失败 - {e}')
"
```

### 任务8：模型性能基准测试

#### 8.1 创建性能测试脚本
**文件位置**：`scripts/benchmark_models.py`

**测试内容**：
- 模型参数数量统计
- 前向传播时间测试
- 内存使用量测试
- GPU利用率测试

#### 8.2 运行性能基准
**测试方法**：
```bash
# 创建基准测试脚本
cat > benchmark_test.py << 'EOF'
import torch
import time
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.models.model_coattn import MCAT_Surv

def benchmark_model(model, name, path_features, omic_features):
    # 参数统计
    params = sum(p.numel() for p in model.parameters())
    
    # 时间测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = model(h_path=path_features, h_omic=omic_features)
    end_time = time.time()
    
    print(f"{name}:")
    print(f"  参数数量: {params:,}")
    print(f"  平均推理时间: {(end_time - start_time) / 10:.4f}s")
    print(f"  输出形状: {output.shape}")
    print()

# 测试数据
path_features = torch.randn(1, 100, 768)
omic_features = torch.randn(1, 50)

# 测试模型
porpoise = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
benchmark_model(porpoise, "PORPOISE", path_features, omic_features)
EOF

python benchmark_test.py
```

### 任务9：模型兼容性测试

#### 9.1 接口一致性测试
**测试内容**：
- 所有模型的输入输出格式一致性
- 参数传递兼容性
- 设备兼容性（CPU/GPU）
- 批处理兼容性

#### 9.2 创建兼容性测试
**测试方法**：
```bash
# 测试所有模型的接口一致性
python -c "
import torch
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.models.model_set_mil import MIL_Attention_FC_surv

# 测试数据
path_features = torch.randn(2, 50, 768)  # batch_size=2
omic_features = torch.randn(2, 50)

# 测试PORPOISE
porpoise = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
out1 = porpoise(h_path=path_features, h_omic=omic_features)

# 测试AMIL
amil = MIL_Attention_FC_surv(omic_input_dim=50, path_input_dim=768, n_classes=4)
out2 = amil(h_path=path_features, h_omic=omic_features)

print(f'PORPOISE输出: {out1.shape}')
print(f'AMIL输出: {out2.shape}')
print('接口兼容性测试通过')
"
```

## 测试和验证

### 单元测试

#### 运行模型单元测试
```bash
# 运行所有模型测试
python -m pytest tests/test_model*.py -v

# 运行特定模型测试
python -m pytest tests/test_porpoise.py -v
python -m pytest tests/test_mcat.py -v
```

### 集成测试

#### 测试1：端到端模型流水线
```bash
# 创建端到端测试
cat > test_e2e_models.py << 'EOF'
import torch
from mmsurv.models.model_porpoise import PorpoiseMMF

# 模拟真实数据维度
batch_size = 4
num_patches = 200
path_dim = 768
omic_dim = 50
n_classes = 4

# 创建测试数据
path_features = torch.randn(batch_size, num_patches, path_dim)
omic_features = torch.randn(batch_size, omic_dim)

# 测试模型
model = PorpoiseMMF(
    omic_input_dim=omic_dim,
    path_input_dim=path_dim,
    n_classes=n_classes
)

# 前向传播
model.eval()
with torch.no_grad():
    output = model(h_path=path_features, h_omic=omic_features)
    
print(f"输入 - 病理特征: {path_features.shape}")
print(f"输入 - 基因组特征: {omic_features.shape}")
print(f"输出 - 预测结果: {output.shape}")
print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
print("端到端测试通过")
EOF

python test_e2e_models.py
```

#### 测试2：梯度流测试
```bash
# 测试梯度反向传播
python -c "
import torch
from mmsurv.models.model_porpoise import PorpoiseMMF

# 创建模型和数据
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
path_features = torch.randn(1, 100, 768, requires_grad=True)
omic_features = torch.randn(1, 50, requires_grad=True)
target = torch.randn(1, 4)

# 前向传播
output = model(h_path=path_features, h_omic=omic_features)
loss = torch.nn.MSELoss()(output, target)

# 反向传播
loss.backward()

# 检查梯度
path_grad_norm = path_features.grad.norm().item()
omic_grad_norm = omic_features.grad.norm().item()

print(f'病理特征梯度范数: {path_grad_norm:.6f}')
print(f'基因组特征梯度范数: {omic_grad_norm:.6f}')
print('梯度流测试通过' if path_grad_norm > 0 and omic_grad_norm > 0 else '梯度流测试失败')
"
```

### 性能验证

#### 内存效率测试
```bash
# 安装内存监控
pip install psutil

# 创建内存测试
cat > memory_efficiency_test.py << 'EOF'
import torch
import psutil
import os
from mmsurv.models.model_porpoise import PorpoiseMMF

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# 基准内存
base_memory = get_memory_usage()
print(f"基准内存使用: {base_memory:.2f} MB")

# 创建模型
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
model_memory = get_memory_usage()
print(f"模型加载后内存: {model_memory:.2f} MB")
print(f"模型内存占用: {model_memory - base_memory:.2f} MB")

# 测试推理
path_features = torch.randn(1, 100, 768)
omic_features = torch.randn(1, 50)

with torch.no_grad():
    output = model(h_path=path_features, h_omic=omic_features)
    
inference_memory = get_memory_usage()
print(f"推理后内存: {inference_memory:.2f} MB")
print(f"推理内存增量: {inference_memory - model_memory:.2f} MB")
EOF

python memory_efficiency_test.py
```

## 故障排除

### 常见问题

#### 问题1：模型初始化失败
**症状**：参数错误或维度不匹配
**解决方案**：
1. 检查输入维度配置
2. 验证模型参数设置
3. 确认依赖包版本

#### 问题2：CUDA内存不足
**症状**：RuntimeError: CUDA out of memory
**解决方案**：
1. 减少batch size
2. 使用梯度检查点
3. 优化模型架构

#### 问题3：梯度消失或爆炸
**症状**：训练不收敛或loss为NaN
**解决方案**：
1. 调整学习率
2. 使用梯度裁剪
3. 检查权重初始化

## 验证清单

完成本阶段后，请确认以下项目：

- [ ] 所有6种模型架构实现完成
- [ ] 模型工具和基础组件实现完成
- [ ] 所有模型单元测试通过
- [ ] 模型接口一致性验证通过
- [ ] 性能基准测试完成
- [ ] 内存效率测试通过
- [ ] 梯度流测试通过
- [ ] 端到端模型测试通过

## 下一步

模型实现完成后，请继续阅读 [03-training-pipeline.md](./03-training-pipeline.md) 进行训练流程实现。