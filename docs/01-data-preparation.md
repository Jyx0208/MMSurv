# 01 - 数据准备和预处理

## 概述

本阶段将实现MMSurv项目的数据处理模块，包括多模态数据的加载、预处理和数据集构建。项目需要处理病理图像特征、基因组数据和临床信息三种模态的数据。

## 数据模态说明

### 1. 病理图像数据
- **格式**：通过CLAM库提取的patch特征
- **维度**：768维特征向量
- **存储**：HDF5格式或pickle文件
- **用途**：表示组织病理学信息

### 2. 基因组数据
- **RNA表达数据**：基因表达谱
- **DNA突变数据**：基因突变信息
- **CNV数据**：拷贝数变异信息
- **格式**：CSV文件

### 3. 临床数据
- **生存时间**：患者生存月数
- **删失状态**：是否发生事件
- **患者ID**：唯一标识符
- **切片ID**：病理切片标识

## 实现任务

### 任务1：创建数据集基础类

#### 1.1 实现通用数据集类
**文件位置**：`mmsurv/datasets/dataset_generic.py`

**实现要求**：
- 创建`Generic_WSI_Survival_Dataset`基类
- 实现数据加载和预处理的通用方法
- 支持数据分割功能
- 实现数据标准化和归一化

**关键功能**：
- 生存时间离散化（将连续时间转换为离散区间）
- 数据分割保存和加载
- 统计信息计算
- 数据预处理pipeline

#### 1.2 测试通用数据集类
**测试文件**：`tests/test_dataset_generic.py`

**测试内容**：
- 数据加载功能测试
- 生存时间离散化测试
- 数据分割功能测试
- 数据预处理测试

**测试方法**：
```bash
# 创建测试文件并运行单元测试
python -m pytest tests/test_dataset_generic.py -v
```

### 任务2：实现生存分析数据集

#### 2.1 实现MIL生存数据集
**文件位置**：`mmsurv/datasets/dataset_survival.py`

**实现要求**：
- 继承通用数据集类
- 实现多实例学习(MIL)数据加载
- 支持多模态数据融合
- 实现cluster ID管理

**关键功能**：
- 病理图像特征加载
- 基因组数据整合
- 患者级别数据聚合
- 模态选择和配置

#### 2.2 实现数据分割类
**实现要求**：
- 创建`Generic_Split`类
- 支持训练/验证/测试分割
- 实现数据预处理和标准化
- 支持加权采样

#### 2.3 测试生存数据集
**测试文件**：`tests/test_dataset_survival.py`

**测试内容**：
- MIL数据加载测试
- 多模态数据融合测试
- 数据分割测试
- 标准化功能测试

### 任务3：实现数据预处理工具

#### 3.1 创建虚拟数据生成器
**文件位置**：`mmsurv/create_dummydata.py`

**实现要求**：
- 生成模拟的多模态数据
- 创建虚拟的生存数据
- 生成测试用的特征文件
- 支持不同数据规模配置

**功能模块**：
- 虚拟病理特征生成
- 虚拟基因组数据生成
- 虚拟临床数据生成
- 数据格式转换和保存

#### 3.2 实现cluster ID保存工具
**文件位置**：`mmsurv/save_cluster_ids.py`

**实现要求**：
- 从patch坐标文件提取cluster信息
- 生成cluster ID映射
- 保存为pickle格式
- 支持批量处理

#### 3.3 测试数据预处理工具
**测试方法**：
```bash
# 测试虚拟数据生成
python mmsurv/create_dummydata.py

# 验证生成的文件
ls -la datasets_csv/
ls -la dummy_data/

# 测试cluster ID生成
python mmsurv/save_cluster_ids.py dummy --patch_dir ./dummy_data/coords_dir/

# 验证cluster文件
ls -la datasets_csv/dummy_cluster_ids.pkl
```

### 任务4：实现文件处理工具

#### 4.1 创建文件工具模块
**文件位置**：`mmsurv/utils/file_utils.py`

**实现要求**：
- 实现pickle文件保存和加载
- 实现HDF5文件处理
- 实现CSV文件读写
- 实现文件路径管理

**关键功能**：
- `save_pkl()` - 保存pickle文件
- `load_pkl()` - 加载pickle文件
- `save_hdf5()` - 保存HDF5文件
- `load_hdf5()` - 加载HDF5文件

#### 4.2 创建核心工具模块
**文件位置**：`mmsurv/utils/utils.py`

**实现要求**：
- 实现目录检查和创建
- 实现数据加载工具
- 实现配置管理
- 实现日志记录

**关键功能**：
- `check_directories()` - 检查和创建目录
- `get_data()` - 数据加载主函数
- 配置文件解析
- 错误处理和日志

#### 4.3 测试文件处理工具
**测试文件**：`tests/test_file_utils.py`

**测试内容**：
- pickle文件操作测试
- HDF5文件操作测试
- 目录管理测试
- 数据加载测试

### 任务5：创建数据验证和测试

#### 5.1 数据完整性验证
**验证脚本**：`scripts/validate_data.py`

**验证内容**：
- 检查所有必需的数据文件
- 验证数据格式和维度
- 检查数据一致性
- 生成数据报告

#### 5.2 端到端数据流测试
**测试脚本**：`scripts/test_data_pipeline.py`

**测试流程**：
1. 生成虚拟数据
2. 加载和预处理数据
3. 创建数据分割
4. 验证数据加载器
5. 检查数据维度和格式

#### 5.3 性能基准测试
**测试内容**：
- 数据加载速度测试
- 内存使用量测试
- 批处理性能测试
- 多进程加载测试

## 测试和验证

### 单元测试

#### 创建测试目录结构
```bash
mkdir -p tests
touch tests/__init__.py
```

#### 运行所有数据相关测试
```bash
# 运行单元测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_dataset_survival.py::test_data_loading -v
```

### 集成测试

#### 测试1：完整数据流水线
```bash
# 生成测试数据
python mmsurv/create_dummydata.py

# 生成cluster IDs
python mmsurv/save_cluster_ids.py dummy --patch_dir ./dummy_data/coords_dir/

# 验证数据加载
python -c "from mmsurv.datasets.dataset_survival import MIL_Survival_Dataset; print('数据集加载成功')"
```

#### 测试2：数据维度验证
```bash
# 创建验证脚本
cat > validate_dimensions.py << 'EOF'
from mmsurv.datasets.dataset_survival import MIL_Survival_Dataset
import pandas as pd
import os

# 加载数据
df = pd.read_csv('datasets_csv/dummy.csv')
print(f"数据行数: {len(df)}")
print(f"数据列数: {len(df.columns)}")
print(f"列名: {list(df.columns)}")

# 检查特征文件
if os.path.exists('dummy_data/feats_dir'):
    print("特征目录存在")
else:
    print("特征目录不存在")
EOF

python validate_dimensions.py
```

#### 测试3：多模态数据一致性
```bash
# 检查各模态数据的患者ID一致性
python -c "
import pandas as pd
import zipfile

# 读取各模态数据
with zipfile.ZipFile('datasets_csv/dummy_rna.csv.zip') as z:
    rna_df = pd.read_csv(z.open('dummy_rna.csv'))
    
with zipfile.ZipFile('datasets_csv/dummy_dna.csv.zip') as z:
    dna_df = pd.read_csv(z.open('dummy_dna.csv'))
    
with zipfile.ZipFile('datasets_csv/dummy_cnv.csv.zip') as z:
    cnv_df = pd.read_csv(z.open('dummy_cnv.csv'))

print(f'RNA患者数: {len(rna_df)}')    
print(f'DNA患者数: {len(dna_df)}')
print(f'CNV患者数: {len(cnv_df)}')
"
```

### 性能测试

#### 内存使用测试
```bash
# 安装内存监控工具
pip install memory-profiler

# 创建内存测试脚本
echo "@profile
def test_memory():
    from mmsurv.datasets.dataset_survival import MIL_Survival_Dataset
    import pandas as pd
    df = pd.read_csv('datasets_csv/dummy.csv')
    dataset = MIL_Survival_Dataset(df=df, data_dir='dummy_data/feats_dir', cluster_id_path='datasets_csv/dummy_cluster_ids.pkl')
    return dataset

if __name__ == '__main__':
    test_memory()" > memory_test.py

# 运行内存测试
python -m memory_profiler memory_test.py
```

## 故障排除

### 常见问题

#### 问题1：数据文件缺失
**症状**：FileNotFoundError
**解决方案**：
1. 检查数据文件路径
2. 重新生成虚拟数据
3. 验证文件权限

#### 问题2：数据维度不匹配
**症状**：Shape mismatch错误
**解决方案**：
1. 检查特征维度配置
2. 验证数据预处理步骤
3. 确认模型输入要求

#### 问题3：内存不足
**症状**：MemoryError或OOM
**解决方案**：
1. 减少batch size
2. 使用数据流式加载
3. 优化数据预处理

## 验证清单

完成本阶段后，请确认以下项目：

- [ ] 通用数据集类实现完成
- [ ] 生存分析数据集实现完成
- [ ] 数据预处理工具实现完成
- [ ] 文件处理工具实现完成
- [ ] 虚拟数据生成成功
- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 数据维度验证通过
- [ ] 性能测试满足要求

## 下一步

数据准备完成后，请继续阅读 [02-model-implementation.md](./02-model-implementation.md) 进行模型架构实现。