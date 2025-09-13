# MMSurv 多模态生存分析项目使用指南

## 📋 项目简介

MMSurv是一个基于多模态多实例学习（MIL）的生存预测项目，专门用于癌症患者的生存分析。该项目整合了以下多种先进模型：

- **AMIL** (Attention-based Multiple Instance Learning)
- **DeepSet** (深度集合学习)
- **DeepAttnMISL** (深度注意力多实例学习)
- **PORPOISE** (病理与基因组多模态融合模型)
- **MCAT** (多模态共注意力Transformer)
- **MOTCat** (多模态最优传输共注意力Transformer)

### 🎯 核心功能
- 整合全切片数字病理图像(WSI)和基因数据
- 支持离散时间生存预测
- 多模态数据融合分析
- 5折交叉验证评估

---

## 🔧 环境准备

### 系统要求
- **操作系统**: Ubuntu 22.04 (推荐)
- **GPU**: NVIDIA显卡 (测试环境: RTX 4090)
- **内存**: 建议16GB以上
- **存储**: 建议50GB以上可用空间

### 软件依赖
- **Python**: 3.10
- **PyTorch**: 2.3.0
- **CUDA**: 对应PyTorch版本的CUDA支持
- **Conda**: 用于环境管理

### Python包依赖
```bash
# 核心依赖
torch==2.3.0
numpy==1.23.4
pandas==1.4.3
h5py
scikit-learn
scikit-survival
tensorboardx
pot==0.9.3  # Python Optimal Transport
```

---

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/ezgiogulmus/MMSurv.git
cd MMSurv
```

### 2. 创建虚拟环境
```bash
# 创建conda环境
conda create -n mmsurv python=3.10 -y
conda activate mmsurv

# 升级pip
pip install --upgrade pip
```

### 3. 安装依赖
```bash
# 安装项目及其依赖
pip install -e .
```

### 4. 验证安装
```bash
# 检查PyTorch是否支持GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

---

## 🚀 运行方式

### 快速开始（使用虚拟数据）

#### 步骤1: 生成虚拟数据
```bash
cd mmsurv
python create_dummydata.py
```
- 生成30个患者的虚拟数据
- 包含RNA、DNA、CNV基因数据和病理图像特征
- 自动创建训练/验证/测试数据分割

#### 步骤2: 保存聚类ID
```bash
python save_cluster_ids.py dummy --patch_dir ./dummy_data/coords_dir/
```
- 对病理图像补丁进行K-means聚类
- 生成聚类标识文件用于后续训练

#### 步骤3: 运行模型训练
```bash
# 基本运行命令
python main.py --data_name dummy --feats_dir ./dummy_data/feats_dir/ --omics rna,dna,cnv --model_type porpoise
```

### 完整参数说明

#### 数据相关参数
```bash
--data_name           # 数据集名称 (必需)
--feats_dir           # 特征文件目录 (必需)
--dataset_dir         # 数据集CSV文件目录 (默认: ./datasets_csv)
--results_dir         # 结果保存目录 (默认: ./results)
--split_dir           # 数据分割目录 (默认: ./splits)
```

#### 模型参数
```bash
--model_type          # 模型类型:
                      # deepset, amil, deepattnmisl, mcat, motcat, porpoise, cmta
--mode                # 模态选择:
                      # omic(仅基因), path(仅病理), pathomic(多模态)
--omics               # 基因数据类型: rna,dna,cnv
--fusion              # 融合方式: None, concat, bilinear
--n_classes           # 生存时间分类数 (默认: 4)
```

#### 训练参数
```bash
--max_epochs          # 最大训练轮数 (默认: 20)
--batch_size          # 批次大小 (默认: 1)
--lr                  # 学习率 (默认: 2e-4)
--bag_loss            # 损失函数: nll_surv, cox_surv, ce_surv
--k                   # 交叉验证折数 (默认: 5)
--seed                # 随机种子 (默认: 1)
```

### 使用配置文件
```bash
# 创建配置文件 config.json
{
    "model_type": "porpoise",
    "mode": "pathomic",
    "omics": "rna,dna,cnv",
    "max_epochs": 50,
    "lr": 1e-4
}

# 使用配置文件运行
python run_mmsurv.py --run_config_file config.json --data_name dummy --feats_dir ./dummy_data/feats_dir/
```

---

## 💡 使用示例

### 示例1: 多模态生存预测
```bash
# 使用PORPOISE模型进行多模态分析
python main.py \
    --data_name dummy \
    --feats_dir ./dummy_data/feats_dir/ \
    --omics rna,dna,cnv \
    --model_type porpoise \
    --mode pathomic \
    --max_epochs 30 \
    --bag_loss nll_surv
```

### 示例2: 仅使用病理图像
```bash
# 仅使用病理数据进行生存预测
python main.py \
    --data_name dummy \
    --feats_dir ./dummy_data/feats_dir/ \
    --model_type amil \
    --mode path \
    --max_epochs 25
```

### 示例3: 仅使用基因数据
```bash
# 仅使用基因数据进行生存预测
python main.py \
    --data_name dummy \
    --omics rna,dna \
    --model_type deepset \
    --mode omic \
    --max_epochs 20
```

### 预期输出
训练完成后，在`results/`目录下会生成：
- **模型文件**: 训练好的模型权重
- **性能指标**: C-index、准确率等
- **实验日志**: 训练过程记录
- **可视化结果**: 生存曲线、混淆矩阵等

---

## ❗ 常见问题

### 1. CUDA内存不足
**错误**: `CUDA out of memory`
**解决**:
- 减小`--batch_size`参数
- 使用`--model_size_wsi small`和`--model_size_omic small`
- 增加梯度累积步数`--gc 64`

### 2. 数据格式错误
**错误**: `File not found` 或 `KeyError`
**解决**:
- 确保特征文件路径正确`--feats_dir`
- 检查CSV文件格式是否正确
- 验证基因数据列名格式

### 3. 模型训练不收敛
**解决**:
- 调整学习率`--lr` (建议1e-4到1e-3)
- 增加训练轮数`--max_epochs`
- 尝试不同的损失函数`--bag_loss`

### 4. 依赖包冲突
**解决**:
```bash
# 重新创建干净环境
conda create -n mmsurv_clean python=3.10 -y
conda activate mmsurv_clean
pip install --upgrade pip
pip install -e .
```

### 5. 虚拟数据生成失败
**解决**:
- 确保有足够的磁盘空间
- 检查目录权限
- 手动创建必要目录: `mkdir -p dummy_data/feats_dir dummy_data/coords_dir`

---

## 📁 项目结构说明

```
MMSurv/
├── mmsurv/                    # 主要代码目录
│   ├── models/               # 模型实现
│   │   ├── model_porpoise.py # PORPOISE模型
│   │   ├── model_mcat.py     # MCAT模型
│   │   ├── model_motcat.py   # MOTCat模型
│   │   └── ...
│   ├── datasets/             # 数据集处理
│   │   └── dataset_survival.py
│   ├── utils/                # 工具函数
│   │   ├── utils.py
│   │   └── core_utils.py
│   ├── main.py               # 主训练脚本
│   ├── arguments.py          # 参数配置
│   ├── create_dummydata.py   # 虚拟数据生成
│   └── save_cluster_ids.py   # 聚类ID保存
├── datasets_csv/             # CSV数据文件
├── splits/                   # 数据分割文件
├── results/                  # 实验结果
├── dummy_data/               # 虚拟数据
│   ├── feats_dir/           # 特征文件
│   └── coords_dir/          # 坐标文件
├── run_mmsurv.py            # 运行脚本
└── setup.py                 # 安装配置
```

---

## 🎓 后续学习建议

### 1. 深入理解模型原理
- **阅读论文**: 项目主页提供了相关论文链接
- **模型对比**: 尝试不同模型了解各自特点
- **参数调优**: 系统性调整超参数

### 2. 准备真实数据
```bash
# 数据预处理流程
1. 使用CLAM库提取病理图像特征
2. 准备基因表达数据 (RNA, DNA, CNV)
3. 格式化生存数据 (生存时间+事件状态)
4. 创建数据分割文件
```

### 3. 模型改进和扩展
- **自定义模型**: 在`models/`目录下添加新模型
- **损失函数**: 修改或新增损失函数
- **特征工程**: 改进特征提取方法

### 4. 性能优化
- **GPU加速**: 优化数据加载和训练流程
- **分布式训练**: 支持多GPU并行训练
- **模型压缩**: 应用量化或剪枝技术

### 5. 参与贡献
- **报告问题**: 在GitHub Issues中提交bug
- **功能建议**: 提出新特性需求
- **代码贡献**: 提交Pull Request

### 6. 相关工具学习
- **CLAM**: 病理图像特征提取
- **scikit-survival**: 生存分析工具包
- **Python Optimal Transport**: 最优传输算法

---

## 📞 技术支持

如果在项目使用过程中遇到问题，可以：
1. 查看项目的GitHub Issues
2. 阅读相关论文了解算法细节
3. 参考项目中的示例代码
4. 检查数据格式和参数配置

*祝您使用愉快！*

---

*文档生成时间: 2025-09-07*
*项目版本: 基于GitHub最新版本*