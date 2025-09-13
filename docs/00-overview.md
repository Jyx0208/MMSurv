# 00 - 项目概述和环境准备

## 项目概述

### 项目背景
MMSurv 是一个多模态深度学习框架，专门用于癌症患者的离散时间生存预测。该项目结合了全切片病理图像(WSI)和基因组数据，通过多种先进的深度学习模型进行生存分析。

### 核心特性
- **多模态融合**：整合病理图像特征和基因组数据(RNA、DNA、CNV)
- **离散时间生存分析**：将连续生存时间转换为离散时间区间进行预测
- **多种模型架构**：支持6种不同的深度学习模型
- **端到端流程**：从数据预处理到模型评估的完整pipeline

### 支持的模型
1. **PORPOISE** - 多模态融合模型，参数效率高
2. **MCAT** - 多模态协同注意力Transformer
3. **MOTCat** - 基于最优传输的协同注意力模型
4. **AMIL** - 注意力引导的多实例学习
5. **DeepSet** - 基于集合函数的深度学习
6. **DeepAttnMISL** - 深度注意力多实例学习

## 环境准备

### 系统要求
- **操作系统**：Ubuntu 22.04 或类似Linux发行版
- **Python版本**：3.10
- **GPU**：建议使用NVIDIA GPU（如RTX 4090）
- **CUDA**：与PyTorch 2.3兼容的CUDA版本

### 环境配置步骤

#### 步骤1：创建项目目录
```bash
# 创建项目根目录
mkdir MMSurv
cd MMSurv
```

#### 步骤2：创建Conda环境
```bash
# 创建Python 3.10环境
conda create -n mmsurv python=3.10 -y

# 激活环境
conda activate mmsurv
```

#### 步骤3：安装基础依赖
```bash
# 升级pip
pip install --upgrade pip

# 安装PyTorch（根据CUDA版本调整）
pip install torch==2.3.0 torchvision torchaudio
```

#### 步骤4：创建项目结构
```bash
# 创建主要目录结构
mkdir -p mmsurv/{models,datasets,utils}
mkdir -p {datasets_csv,dummy_data,results,splits,scripts}
```

### 测试环境配置

#### 测试1：验证Python环境
```bash
# 检查Python版本
python --version
# 预期输出：Python 3.10.x

# 检查conda环境
conda info --envs
# 应该看到mmsurv环境
```

#### 测试2：验证PyTorch安装
```bash
# 创建测试脚本
echo "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" > test_pytorch.py

# 运行测试
python test_pytorch.py

# 预期输出：
# PyTorch版本: 2.3.0
# CUDA可用: True (如果有GPU)
```

#### 测试3：验证目录结构
```bash
# 检查目录结构
tree -L 2
# 或者使用
find . -type d -maxdepth 2

# 应该看到完整的目录结构
```

### 依赖包清单

项目需要以下主要依赖包：
- `torch==2.3.0` - 深度学习框架
- `numpy==1.23.4` - 数值计算
- `pandas==1.4.3` - 数据处理
- `scikit-learn` - 机器学习工具
- `scikit-survival` - 生存分析
- `h5py` - HDF5文件处理
- `tensorboardx` - 训练监控
- `pot==0.9.3` - 最优传输

### 故障排除

#### 问题1：CUDA版本不兼容
**症状**：PyTorch无法检测到GPU
**解决方案**：
1. 检查CUDA版本：`nvidia-smi`
2. 安装对应版本的PyTorch
3. 参考PyTorch官网的安装指南

#### 问题2：依赖包冲突
**症状**：包安装失败或版本冲突
**解决方案**：
1. 使用虚拟环境隔离依赖
2. 按照指定版本安装包
3. 必要时重新创建环境

#### 问题3：内存不足
**症状**：训练时出现OOM错误
**解决方案**：
1. 减小batch_size
2. 使用梯度累积
3. 考虑使用更小的模型

## 下一步

环境配置完成后，请继续阅读 [01-data-preparation.md](./01-data-preparation.md) 进行数据准备和预处理。

## 验证清单

在进入下一阶段前，请确认以下项目：

- [ ] Python 3.10环境已创建并激活
- [ ] PyTorch 2.3.0已正确安装
- [ ] CUDA功能正常（如果使用GPU）
- [ ] 项目目录结构已创建
- [ ] 所有环境测试均通过

完成以上验证后，即可开始数据准备阶段。