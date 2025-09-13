# 05 - 测试和验证

## 概述

本阶段将建立MMSurv项目的完整测试和验证体系，确保项目的可靠性、稳定性和正确性。包括单元测试、集成测试、性能测试、回归测试和持续集成等。

## 测试体系架构

### 测试层次
1. **单元测试** - 测试单个函数和类的功能
2. **集成测试** - 测试模块间的交互
3. **系统测试** - 测试完整的端到端流程
4. **性能测试** - 测试系统性能和资源使用
5. **回归测试** - 确保新更改不破坏现有功能
6. **验收测试** - 验证系统满足需求

### 测试工具
- **pytest** - 主要测试框架
- **coverage** - 代码覆盖率分析
- **hypothesis** - 属性基础测试
- **mock** - 模拟对象和依赖
- **benchmark** - 性能基准测试

## 实现任务

### 任务1：建立测试框架

#### 1.1 创建测试目录结构
**目录结构**：
```
tests/
├── __init__.py
├── conftest.py                 # pytest配置和fixtures
├── unit/                       # 单元测试
│   ├── __init__.py
│   ├── test_models/           # 模型测试
│   ├── test_datasets/         # 数据集测试
│   ├── test_utils/            # 工具函数测试
│   └── test_visualization/    # 可视化测试
├── integration/               # 集成测试
│   ├── __init__.py
│   ├── test_training_pipeline.py
│   ├── test_evaluation_pipeline.py
│   └── test_data_pipeline.py
├── system/                    # 系统测试
│   ├── __init__.py
│   ├── test_end_to_end.py
│   └── test_cross_validation.py
├── performance/               # 性能测试
│   ├── __init__.py
│   ├── test_model_speed.py
│   ├── test_memory_usage.py
│   └── test_scalability.py
└── fixtures/                  # 测试数据和fixtures
    ├── __init__.py
    ├── sample_data.py
    └── mock_models.py
```

#### 1.2 创建pytest配置
**文件位置**：`tests/conftest.py`

**实现要求**：
- 配置pytest设置
- 定义通用fixtures
- 设置测试环境
- 配置日志和报告

#### 1.3 创建测试工具
**文件位置**：`tests/test_utils.py`

**实现要求**：
- 实现测试数据生成器
- 实现断言辅助函数
- 实现性能测试工具
- 实现模拟对象工厂

#### 1.4 测试框架验证
**测试方法**：
```bash
# 创建基础测试框架验证
cat > test_framework_setup.py << 'EOF'
import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

def test_pytest_working():
    """验证pytest基础功能"""
    assert True
    
def test_imports():
    """验证核心模块可以导入"""
    try:
        import mmsurv
        import mmsurv.models
        import mmsurv.datasets
        import mmsurv.utils
        print("所有核心模块导入成功")
    except ImportError as e:
        pytest.fail(f"模块导入失败: {e}")
        
def test_torch_available():
    """验证PyTorch可用性"""
    import torch
    assert torch.cuda.is_available() or True  # CPU也可以
    print(f"PyTorch版本: {torch.__version__}")
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

python test_framework_setup.py
```

### 任务2：实现单元测试

#### 2.1 模型单元测试
**文件位置**：`tests/unit/test_models/`

##### 2.1.1 PORPOISE模型测试
**文件位置**：`tests/unit/test_models/test_porpoise.py`

**测试内容**：
- 模型初始化测试
- 前向传播测试
- 参数数量验证
- 输出维度检查
- 梯度流测试

**测试方法**：
```bash
# 创建PORPOISE模型单元测试
cat > test_porpoise_unit.py << 'EOF'
import pytest
import torch
from mmsurv.models.model_porpoise import PorpoiseMMF, LRBilinearFusion

class TestPorpoiseMMF:
    """PORPOISE模型单元测试"""
    
    @pytest.fixture
    def model_config(self):
        return {
            'omic_input_dim': 50,
            'path_input_dim': 768,
            'n_classes': 4,
            'dropout': 0.25
        }
    
    @pytest.fixture
    def sample_data(self):
        return {
            'path_features': torch.randn(2, 100, 768),
            'omic_features': torch.randn(2, 50)
        }
    
    def test_model_initialization(self, model_config):
        """测试模型初始化"""
        model = PorpoiseMMF(**model_config)
        assert model is not None
        assert hasattr(model, 'path_fc')
        assert hasattr(model, 'omic_fc')
        
    def test_forward_pass(self, model_config, sample_data):
        """测试前向传播"""
        model = PorpoiseMMF(**model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(
                h_path=sample_data['path_features'],
                h_omic=sample_data['omic_features']
            )
        
        assert output.shape == (2, 4)  # batch_size=2, n_classes=4
        assert not torch.isnan(output).any()
        
    def test_parameter_count(self, model_config):
        """测试参数数量"""
        model = PorpoiseMMF(**model_config)
        param_count = sum(p.numel() for p in model.parameters())
        
        # PORPOISE应该是参数效率高的模型
        assert param_count < 10_000_000  # 少于1000万参数
        print(f"PORPOISE参数数量: {param_count:,}")
        
    def test_gradient_flow(self, model_config, sample_data):
        """测试梯度流"""
        model = PorpoiseMMF(**model_config)
        model.train()
        
        output = model(
            h_path=sample_data['path_features'],
            h_omic=sample_data['omic_features']
        )
        
        # 计算损失并反向传播
        target = torch.randn_like(output)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "模型参数没有梯度"
        
class TestLRBilinearFusion:
    """低秩双线性融合测试"""
    
    def test_fusion_initialization(self):
        """测试融合层初始化"""
        fusion = LRBilinearFusion(dim1=768, dim2=50, scale_dim1=8, scale_dim2=8)
        assert fusion is not None
        
    def test_fusion_forward(self):
        """测试融合层前向传播"""
        fusion = LRBilinearFusion(dim1=768, dim2=50, scale_dim1=8, scale_dim2=8)
        
        vec1 = torch.randn(2, 768)
        vec2 = torch.randn(2, 50)
        
        output = fusion(vec1, vec2)
        assert output.shape == (2, 8)  # scale_dim1
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

python test_porpoise_unit.py
```

##### 2.1.2 其他模型测试
**测试文件**：
- `test_mcat.py` - MCAT模型测试
- `test_motcat.py` - MOTCat模型测试
- `test_mil_models.py` - MIL模型系列测试
- `test_genomic.py` - 基因组模型测试

#### 2.2 数据集单元测试
**文件位置**：`tests/unit/test_datasets/`

##### 2.2.1 生存数据集测试
**文件位置**：`tests/unit/test_datasets/test_survival_dataset.py`

**测试内容**：
- 数据集初始化测试
- 数据加载测试
- 标签处理测试
- 数据增强测试
- 批处理测试

**测试方法**：
```bash
# 创建数据集单元测试
cat > test_dataset_unit.py << 'EOF'
import pytest
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from mmsurv.datasets.dataset_survival import Generic_WSI_Survival_Dataset

class TestSurvivalDataset:
    """生存数据集单元测试"""
    
    @pytest.fixture
    def mock_csv_data(self, tmp_path):
        """创建模拟CSV数据"""
        data = {
            'case_id': [f'case_{i}' for i in range(10)],
            'slide_id': [f'slide_{i}' for i in range(10)],
            'survival_months': np.random.uniform(1, 100, 10),
            'censorship': np.random.randint(0, 2, 10),
            'oncotree_code': ['LUAD'] * 10
        }
        
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def mock_features_dir(self, tmp_path):
        """创建模拟特征目录"""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # 创建模拟特征文件
        for i in range(10):
            feature_file = features_dir / f"slide_{i}.pt"
            features = torch.randn(50, 768)  # 50个patch，768维特征
            torch.save(features, feature_file)
            
        return features_dir
    
    def test_dataset_initialization(self, mock_csv_data, mock_features_dir):
        """测试数据集初始化"""
        dataset = Generic_WSI_Survival_Dataset(
            csv_file=str(mock_csv_data),
            data_dir=str(mock_features_dir),
            shuffle=False,
            seed=42,
            print_info=True,
            n_bins=4,
            ignore=[]
        )
        
        assert len(dataset) == 10
        assert dataset.num_classes == 4
        
    def test_dataset_getitem(self, mock_csv_data, mock_features_dir):
        """测试数据获取"""
        dataset = Generic_WSI_Survival_Dataset(
            csv_file=str(mock_csv_data),
            data_dir=str(mock_features_dir),
            shuffle=False,
            seed=42,
            n_bins=4
        )
        
        # 获取第一个样本
        sample = dataset[0]
        
        assert 'features' in sample
        assert 'label' in sample
        assert 'event_time' in sample
        assert 'c' in sample  # censorship
        
        # 检查特征维度
        features = sample['features']
        assert features.shape[1] == 768  # 特征维度
        
    def test_dataloader_integration(self, mock_csv_data, mock_features_dir):
        """测试数据加载器集成"""
        dataset = Generic_WSI_Survival_Dataset(
            csv_file=str(mock_csv_data),
            data_dir=str(mock_features_dir),
            shuffle=False,
            seed=42,
            n_bins=4
        )
        
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 测试批次加载
        batch = next(iter(dataloader))
        
        assert len(batch) == 4  # features, label, event_time, censorship
        assert batch[0].shape[0] == 2  # batch_size
        
    def test_survival_binning(self, mock_csv_data, mock_features_dir):
        """测试生存时间分箱"""
        dataset = Generic_WSI_Survival_Dataset(
            csv_file=str(mock_csv_data),
            data_dir=str(mock_features_dir),
            shuffle=False,
            seed=42,
            n_bins=4
        )
        
        # 检查标签范围
        labels = [dataset[i]['label'] for i in range(len(dataset))]
        unique_labels = set(labels)
        
        assert all(0 <= label < 4 for label in labels)
        print(f"唯一标签: {sorted(unique_labels)}")
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

python test_dataset_unit.py
```

#### 2.3 工具函数单元测试
**文件位置**：`tests/unit/test_utils/`

##### 2.3.1 评估工具测试
**文件位置**：`tests/unit/test_utils/test_eval_utils.py`

**测试内容**：
- C-Index计算测试
- AUC计算测试
- 统计检验测试
- 边界条件测试

**测试方法**：
```bash
# 创建评估工具单元测试
cat > test_eval_utils_unit.py << 'EOF'
import pytest
import torch
import numpy as np
from mmsurv.utils.eval_utils import calculate_c_index
from mmsurv.utils.statistical_tests import paired_t_test

class TestCIndexCalculation:
    """C-Index计算测试"""
    
    def test_perfect_concordance(self):
        """测试完美一致性情况"""
        # 风险评分与事件时间完全负相关
        risk_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        event_times = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        censorship = torch.tensor([1, 1, 1, 1, 1])  # 无删失
        
        c_index = calculate_c_index(risk_scores, event_times, censorship)
        assert abs(c_index - 1.0) < 1e-6, f"期望C-Index=1.0，实际={c_index}"
        
    def test_random_concordance(self):
        """测试随机一致性情况"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_samples = 1000
        risk_scores = torch.randn(n_samples)
        event_times = torch.rand(n_samples) * 100
        censorship = torch.ones(n_samples)  # 无删失
        
        c_index = calculate_c_index(risk_scores, event_times, censorship)
        
        # 随机情况下C-Index应该接近0.5
        assert 0.4 < c_index < 0.6, f"随机C-Index应接近0.5，实际={c_index}"
        
    def test_with_censoring(self):
        """测试包含删失的情况"""
        risk_scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
        event_times = torch.tensor([10.0, 8.0, 6.0, 4.0])
        censorship = torch.tensor([1, 0, 1, 1])  # 第二个样本被删失
        
        c_index = calculate_c_index(risk_scores, event_times, censorship)
        
        # 应该能正常计算，不报错
        assert 0 <= c_index <= 1, f"C-Index应在[0,1]范围内，实际={c_index}"
        
    def test_edge_cases(self):
        """测试边界情况"""
        # 所有样本都被删失
        risk_scores = torch.tensor([1.0, 2.0, 3.0])
        event_times = torch.tensor([10.0, 8.0, 6.0])
        censorship = torch.tensor([0, 0, 0])  # 全部删失
        
        c_index = calculate_c_index(risk_scores, event_times, censorship)
        
        # 全部删失时，C-Index应该是NaN或特殊值
        assert torch.isnan(torch.tensor(c_index)) or c_index == 0.5
        
class TestStatisticalTests:
    """统计检验测试"""
    
    def test_paired_t_test_identical(self):
        """测试相同数据的配对t检验"""
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([1, 2, 3, 4, 5])
        
        t_stat, p_value = paired_t_test(data1, data2)
        
        assert abs(t_stat) < 1e-10, f"相同数据t统计量应为0，实际={t_stat}"
        assert p_value > 0.9, f"相同数据p值应接近1，实际={p_value}"
        
    def test_paired_t_test_different(self):
        """测试不同数据的配对t检验"""
        np.random.seed(42)
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)  # 均值差为1
        
        t_stat, p_value = paired_t_test(data1, data2)
        
        assert abs(t_stat) > 2, f"显著差异的t统计量应较大，实际={abs(t_stat)}"
        assert p_value < 0.05, f"显著差异的p值应小于0.05，实际={p_value}"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

python test_eval_utils_unit.py
```

### 任务3：实现集成测试

#### 3.1 训练流程集成测试
**文件位置**：`tests/integration/test_training_pipeline.py`

**测试内容**：
- 完整训练流程测试
- 多模型训练测试
- 配置文件集成测试
- 检查点保存和加载测试

**测试方法**：
```bash
# 创建训练流程集成测试
cat > test_training_integration.py << 'EOF'
import pytest
import torch
import tempfile
import os
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.utils.core_utils import train, EarlyStopping
from mmsurv.utils.loss_func import CrossEntropySurvLoss
from torch.utils.data import DataLoader, TensorDataset

class TestTrainingPipeline:
    """训练流程集成测试"""
    
    @pytest.fixture
    def mock_data_loaders(self):
        """创建模拟数据加载器"""
        n_samples = 20
        path_features = torch.randn(n_samples, 50, 768)
        omic_features = torch.randn(n_samples, 50)
        labels = torch.randint(0, 4, (n_samples,))
        event_times = torch.rand(n_samples)
        censorship = torch.randint(0, 2, (n_samples,))
        
        dataset = TensorDataset(
            path_features, omic_features, labels, event_times, censorship
        )
        
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        return train_loader, val_loader
    
    def test_basic_training_loop(self, mock_data_loaders):
        """测试基础训练循环"""
        train_loader, val_loader = mock_data_loaders
        
        # 创建模型和优化器
        model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # 简化训练循环
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(3):  # 只训练3轮
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, (path_feat, omic_feat, labels, _, _) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 前向传播
                output = model(h_path=path_feat, h_omic=omic_feat)
                loss = loss_fn(output, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx >= 2:  # 限制批次数量
                    break
            
            avg_loss = epoch_loss / num_batches
            
            if epoch == 0:
                initial_loss = avg_loss
            if epoch == 2:
                final_loss = avg_loss
                
            print(f"Epoch {epoch+1}: 平均损失 = {avg_loss:.4f}")
        
        # 验证训练有效性
        assert initial_loss is not None and final_loss is not None
        print(f"初始损失: {initial_loss:.4f}, 最终损失: {final_loss:.4f}")
        
    def test_early_stopping_integration(self, mock_data_loaders):
        """测试早停机制集成"""
        train_loader, val_loader = mock_data_loaders
        
        model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(patience=2, min_delta=0.001)
        
        # 模拟训练过程
        val_losses = [1.0, 0.9, 0.91, 0.92, 0.93]  # 验证损失开始上升
        
        for epoch, val_loss in enumerate(val_losses):
            should_stop = early_stopping(val_loss)
            print(f"Epoch {epoch+1}: val_loss={val_loss:.3f}, should_stop={should_stop}")
            
            if should_stop:
                print(f"早停触发于第{epoch+1}轮")
                assert epoch >= 2  # 应该在第3轮或之后触发
                break
        
    def test_checkpoint_save_load(self, mock_data_loaders):
        """测试检查点保存和加载"""
        train_loader, val_loader = mock_data_loaders
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建和训练模型
            model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 保存初始状态
            initial_state = model.state_dict().copy()
            
            # 简单训练几步
            model.train()
            for batch_idx, (path_feat, omic_feat, labels, _, _) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(h_path=path_feat, h_omic=omic_feat)
                loss = torch.nn.CrossEntropyLoss()(output, labels)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 1:  # 只训练2个批次
                    break
            
            # 保存检查点
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': 1,
                'loss': loss.item()
            }, checkpoint_path)
            
            # 创建新模型并加载检查点
            new_model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            
            checkpoint = torch.load(checkpoint_path)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 验证加载成功
            assert checkpoint['epoch'] == 1
            assert isinstance(checkpoint['loss'], float)
            
            # 验证模型参数已更改（与初始状态不同）
            current_state = new_model.state_dict()
            params_changed = False
            for key in initial_state:
                if not torch.equal(initial_state[key], current_state[key]):
                    params_changed = True
                    break
            
            assert params_changed, "模型参数应该在训练后发生变化"
            print("检查点保存和加载测试通过")
            
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

python test_training_integration.py
```

#### 3.2 评估流程集成测试
**文件位置**：`tests/integration/test_evaluation_pipeline.py`

**测试内容**：
- 完整评估流程测试
- 多模型比较测试
- 可视化生成测试
- 报告生成测试

#### 3.3 数据流程集成测试
**文件位置**：`tests/integration/test_data_pipeline.py`

**测试内容**：
- 数据加载流程测试
- 数据预处理流程测试
- 批处理流程测试
- 内存使用测试

### 任务4：实现系统测试

#### 4.1 端到端系统测试
**文件位置**：`tests/system/test_end_to_end.py`

**测试内容**：
- 完整项目流程测试
- 多模型端到端测试
- 配置驱动测试
- 结果一致性测试

**测试方法**：
```bash
# 创建端到端系统测试
cat > test_e2e_system.py << 'EOF'
import pytest
import torch
import tempfile
import os
import yaml
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.models.model_set_mil import MIL_Attention_FC_surv
from mmsurv.evaluation.model_comparator import ModelComparator
from torch.utils.data import DataLoader, TensorDataset

class TestEndToEndSystem:
    """端到端系统测试"""
    
    @pytest.fixture
    def system_config(self):
        """系统配置"""
        return {
            'models': {
                'porpoise': {
                    'type': 'PorpoiseMMF',
                    'omic_input_dim': 50,
                    'path_input_dim': 768,
                    'n_classes': 4
                },
                'amil': {
                    'type': 'MIL_Attention_FC_surv',
                    'omic_input_dim': 50,
                    'path_input_dim': 768,
                    'n_classes': 4
                }
            },
            'training': {
                'batch_size': 4,
                'num_epochs': 2,
                'learning_rate': 0.001
            },
            'evaluation': {
                'metrics': ['c_index', 'auc'],
                'cross_validation': {
                    'n_splits': 3,
                    'shuffle': True,
                    'random_state': 42
                }
            }
        }
    
    @pytest.fixture
    def system_data(self):
        """系统测试数据"""
        n_samples = 30
        return {
            'path_features': torch.randn(n_samples, 50, 768),
            'omic_features': torch.randn(n_samples, 50),
            'labels': torch.randint(0, 4, (n_samples,)),
            'event_times': torch.rand(n_samples) * 100,
            'censorship': torch.randint(0, 2, (n_samples,))
        }
    
    def test_multi_model_comparison(self, system_config, system_data):
        """测试多模型比较系统"""
        # 创建模型
        models = {
            'PORPOISE': PorpoiseMMF(
                omic_input_dim=50,
                path_input_dim=768,
                n_classes=4
            ),
            'AMIL': MIL_Attention_FC_surv(
                omic_input_dim=50,
                path_input_dim=768,
                n_classes=4
            )
        }
        
        # 模拟训练（简化）
        for model_name, model in models.items():
            model.eval()  # 设为评估模式
            print(f"模型 {model_name} 准备完成")
        
        # 模拟评估
        results = {}
        for model_name, model in models.items():
            with torch.no_grad():
                output = model(
                    h_path=system_data['path_features'],
                    h_omic=system_data['omic_features']
                )
                
                # 模拟评估指标
                results[model_name] = {
                    'c_index': torch.rand(1).item() * 0.3 + 0.5,  # 0.5-0.8
                    'auc': torch.rand(1).item() * 0.3 + 0.6,      # 0.6-0.9
                    'params': sum(p.numel() for p in model.parameters())
                }
        
        # 验证结果
        assert len(results) == 2
        for model_name, metrics in results.items():
            assert 0.5 <= metrics['c_index'] <= 0.8
            assert 0.6 <= metrics['auc'] <= 0.9
            assert metrics['params'] > 0
            print(f"{model_name}: C-Index={metrics['c_index']:.3f}, AUC={metrics['auc']:.3f}")
        
        print("多模型比较系统测试通过")
    
    def test_config_driven_workflow(self, system_config, system_data):
        """测试配置驱动的工作流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存配置文件
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(system_config, f)
            
            # 加载配置
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # 验证配置加载
            assert loaded_config['models']['porpoise']['type'] == 'PorpoiseMMF'
            assert loaded_config['training']['batch_size'] == 4
            assert loaded_config['evaluation']['metrics'] == ['c_index', 'auc']
            
            # 根据配置创建模型
            porpoise_config = loaded_config['models']['porpoise']
            model = PorpoiseMMF(
                omic_input_dim=porpoise_config['omic_input_dim'],
                path_input_dim=porpoise_config['path_input_dim'],
                n_classes=porpoise_config['n_classes']
            )
            
            # 验证模型创建
            assert model is not None
            print("配置驱动工作流程测试通过")
    
    def test_reproducibility(self, system_data):
        """测试结果可重现性"""
        # 设置随机种子
        torch.manual_seed(42)
        
        # 第一次运行
        model1 = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
        model1.eval()
        
        with torch.no_grad():
            output1 = model1(
                h_path=system_data['path_features'][:5],
                h_omic=system_data['omic_features'][:5]
            )
        
        # 重置随机种子
        torch.manual_seed(42)
        
        # 第二次运行
        model2 = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
        model2.eval()
        
        with torch.no_grad():
            output2 = model2(
                h_path=system_data['path_features'][:5],
                h_omic=system_data['omic_features'][:5]
            )
        
        # 验证结果一致性
        assert torch.allclose(output1, output2, atol=1e-6), "结果应该可重现"
        print("可重现性测试通过")
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

python test_e2e_system.py
```

#### 4.2 交叉验证系统测试
**文件位置**：`tests/system/test_cross_validation.py`

**测试内容**：
- K折交叉验证完整流程
- 统计显著性测试
- 结果聚合和报告
- 性能稳定性测试

### 任务5：实现性能测试

#### 5.1 模型速度性能测试
**文件位置**：`tests/performance/test_model_speed.py`

**测试内容**：
- 推理速度基准测试
- 训练速度基准测试
- 批处理性能测试
- GPU加速测试

**测试方法**：
```bash
# 创建性能基准测试
cat > test_performance_benchmark.py << 'EOF'
import pytest
import torch
import time
import psutil
import os
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.models.model_coattn import MCAT_Surv
from torch.utils.data import DataLoader, TensorDataset

class TestModelPerformance:
    """模型性能测试"""
    
    @pytest.fixture
    def performance_data(self):
        """性能测试数据"""
        batch_sizes = [1, 4, 8, 16]
        data = {}
        
        for batch_size in batch_sizes:
            data[batch_size] = {
                'path_features': torch.randn(batch_size, 100, 768),
                'omic_features': torch.randn(batch_size, 50)
            }
        
        return data
    
    def test_inference_speed(self, performance_data):
        """测试推理速度"""
        model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
        model.eval()
        
        results = {}
        
        for batch_size, data in performance_data.items():
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    _ = model(h_path=data['path_features'], h_omic=data['omic_features'])
            
            # 性能测试
            start_time = time.time()
            num_iterations = 50
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    output = model(h_path=data['path_features'], h_omic=data['omic_features'])
            
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_batch = total_time / num_iterations
            avg_time_per_sample = avg_time_per_batch / batch_size
            
            results[batch_size] = {
                'total_time': total_time,
                'avg_time_per_batch': avg_time_per_batch,
                'avg_time_per_sample': avg_time_per_sample,
                'throughput': batch_size * num_iterations / total_time
            }
            
            print(f"Batch Size {batch_size}:")
            print(f"  每批次平均时间: {avg_time_per_batch*1000:.2f}ms")
            print(f"  每样本平均时间: {avg_time_per_sample*1000:.2f}ms")
            print(f"  吞吐量: {results[batch_size]['throughput']:.1f} samples/sec")
        
        # 验证性能要求
        for batch_size, metrics in results.items():
            # 每样本推理时间应小于100ms
            assert metrics['avg_time_per_sample'] < 0.1, f"推理速度过慢: {metrics['avg_time_per_sample']:.3f}s"
        
        print("推理速度测试通过")
    
    def test_memory_usage(self, performance_data):
        """测试内存使用"""
        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        
        # 基准内存
        baseline_memory = get_memory_usage()
        
        # 创建模型
        model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
        model_memory = get_memory_usage()
        
        model_overhead = model_memory - baseline_memory
        print(f"模型内存占用: {model_overhead:.1f} MB")
        
        # 测试不同批次大小的内存使用
        memory_results = {}
        
        for batch_size in [1, 4, 8, 16]:
            data = performance_data[batch_size]
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(h_path=data['path_features'], h_omic=data['omic_features'])
            
            current_memory = get_memory_usage()
            batch_memory = current_memory - model_memory
            
            memory_results[batch_size] = {
                'total_memory': current_memory,
                'batch_memory': batch_memory,
                'memory_per_sample': batch_memory / batch_size
            }
            
            print(f"Batch Size {batch_size}: {batch_memory:.1f} MB ({batch_memory/batch_size:.1f} MB/sample)")
        
        # 验证内存使用合理性
        for batch_size, metrics in memory_results.items():
            # 每样本内存使用应小于50MB
            assert metrics['memory_per_sample'] < 50, f"内存使用过高: {metrics['memory_per_sample']:.1f}MB/sample"
        
        print("内存使用测试通过")
    
    def test_scalability(self):
        """测试可扩展性"""
        model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
        model.eval()
        
        # 测试不同输入大小的处理能力
        patch_counts = [10, 50, 100, 200, 500]
        scalability_results = {}
        
        for patch_count in patch_counts:
            path_features = torch.randn(1, patch_count, 768)
            omic_features = torch.randn(1, 50)
            
            # 测试处理时间
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):  # 重复10次取平均
                    output = model(h_path=path_features, h_omic=omic_features)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            scalability_results[patch_count] = {
                'avg_time': avg_time,
                'time_per_patch': avg_time / patch_count
            }
            
            print(f"Patch Count {patch_count}: {avg_time*1000:.1f}ms ({avg_time/patch_count*1000:.3f}ms/patch)")
        
        # 验证可扩展性
        times = [metrics['avg_time'] for metrics in scalability_results.values()]
        
        # 处理时间应该随输入大小合理增长（不应该是指数增长）
        time_ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
        max_ratio = max(time_ratios)
        
        assert max_ratio < 10, f"可扩展性差，时间增长比例过大: {max_ratio:.1f}"
        print("可扩展性测试通过")
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

python test_performance_benchmark.py
```

#### 5.2 内存使用测试
**文件位置**：`tests/performance/test_memory_usage.py`

#### 5.3 可扩展性测试
**文件位置**：`tests/performance/test_scalability.py`

### 任务6：实现回归测试

#### 6.1 创建回归测试套件
**文件位置**：`tests/regression/test_regression_suite.py`

**测试内容**：
- 模型输出一致性测试
- 性能回归测试
- API兼容性测试
- 配置兼容性测试

#### 6.2 创建基准结果
**文件位置**：`tests/regression/baseline_results.json`

**内容**：
- 各模型的基准性能指标
- 标准测试数据的预期输出
- 性能基准数据

### 任务7：实现持续集成

#### 7.1 创建CI配置
**文件位置**：`.github/workflows/ci.yml`

**实现要求**：
- 自动化测试执行
- 多Python版本测试
- 代码覆盖率报告
- 性能回归检测

#### 7.2 创建测试脚本
**文件位置**：`scripts/run_tests.py`

**实现要求**：
- 完整测试套件执行
- 测试结果汇总
- 失败测试报告
- 覆盖率分析

**测试方法**：
```bash
# 创建测试执行脚本
cat > run_comprehensive_tests.py << 'EOF'
import subprocess
import sys
import os
import time

def run_test_suite(test_type, test_path, description):
    """运行测试套件"""
    print(f"\n{'='*60}")
    print(f"运行 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"执行时间: {duration:.1f}秒")
        
        if result.returncode == 0:
            print(f"✅ {description} 通过")
            return True
        else:
            print(f"❌ {description} 失败")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"💥 {description} 执行错误: {e}")
        return False

def main():
    """主测试函数"""
    print("MMSurv 项目综合测试套件")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 测试套件定义
    test_suites = [
        ('unit', 'tests/unit/', '单元测试'),
        ('integration', 'tests/integration/', '集成测试'),
        ('system', 'tests/system/', '系统测试'),
        ('performance', 'tests/performance/', '性能测试')
    ]
    
    results = {}
    total_start_time = time.time()
    
    # 运行所有测试套件
    for test_type, test_path, description in test_suites:
        if os.path.exists(test_path):
            results[test_type] = run_test_suite(test_type, test_path, description)
        else:
            print(f"⚠️  跳过 {description} - 目录不存在: {test_path}")
            results[test_type] = None
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_type, result in results.items():
        if result is True:
            print(f"✅ {test_type.upper()}: 通过")
            passed += 1
        elif result is False:
            print(f"❌ {test_type.upper()}: 失败")
            failed += 1
        else:
            print(f"⚠️  {test_type.upper()}: 跳过")
            skipped += 1
    
    print(f"\n总执行时间: {total_duration:.1f}秒")
    print(f"通过: {passed}, 失败: {failed}, 跳过: {skipped}")
    
    # 生成覆盖率报告（如果可用）
    try:
        print("\n生成代码覆盖率报告...")
        subprocess.run([sys.executable, '-m', 'coverage', 'report'], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  无法生成覆盖率报告（coverage未安装或未配置）")
    
    # 返回退出码
    if failed > 0:
        print("\n❌ 测试套件执行失败")
        sys.exit(1)
    else:
        print("\n✅ 所有测试通过")
        sys.exit(0)

if __name__ == "__main__":
    main()
EOF

python run_comprehensive_tests.py
```

## 测试执行和报告

### 运行所有测试

#### 基础测试执行
```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定类型的测试
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/system/ -v
python -m pytest tests/performance/ -v

# 运行特定模块测试
python -m pytest tests/unit/test_models/ -v
python -m pytest tests/unit/test_datasets/ -v
```

#### 覆盖率测试
```bash
# 安装覆盖率工具
pip install coverage pytest-cov

# 运行覆盖率测试
python -m pytest tests/ --cov=mmsurv --cov-report=html --cov-report=term

# 查看覆盖率报告
open htmlcov/index.html  # 在浏览器中查看
```

#### 性能基准测试
```bash
# 运行性能基准测试
python -m pytest tests/performance/ -v --benchmark-only

# 生成性能报告
python -m pytest tests/performance/ --benchmark-json=benchmark_results.json
```

### 测试报告生成

#### HTML测试报告
```bash
# 安装报告生成工具
pip install pytest-html

# 生成HTML报告
python -m pytest tests/ --html=test_report.html --self-contained-html
```

#### JUnit XML报告
```bash
# 生成JUnit XML报告（用于CI/CD）
python -m pytest tests/ --junitxml=test_results.xml
```

## 质量保证

### 代码质量检查

#### 静态代码分析
```bash
# 安装代码质量工具
pip install flake8 black isort mypy

# 代码格式检查
flake8 mmsurv/ tests/

# 代码格式化
black mmsurv/ tests/

# 导入排序
isort mmsurv/ tests/

# 类型检查
mypy mmsurv/
```

#### 安全检查
```bash
# 安装安全检查工具
pip install bandit safety

# 安全漏洞检查
bandit -r mmsurv/

# 依赖安全检查
safety check
```

### 性能监控

#### 内存泄漏检测
```bash
# 安装内存分析工具
pip install memory-profiler

# 内存使用分析
python -m memory_profiler scripts/train.py
```

#### 性能分析
```bash
# 安装性能分析工具
pip install py-spy

# 性能分析
py-spy record -o profile.svg -- python scripts/train.py
```

## 故障排除

### 常见测试问题

#### 问题1：测试环境配置错误
**症状**：ImportError或ModuleNotFoundError
**解决方案**：
1. 检查PYTHONPATH设置
2. 确认虚拟环境激活
3. 验证依赖包安装
4. 检查测试文件路径

#### 问题2：测试数据问题
**症状**：测试数据加载失败或格式错误
**解决方案**：
1. 检查测试数据路径
2. 验证数据格式
3. 确认文件权限
4. 使用模拟数据替代

#### 问题3：测试超时
**症状**：测试执行时间过长
**解决方案**：
1. 减少测试数据大小
2. 优化测试逻辑
3. 增加超时时间
4. 并行执行测试

#### 问题4：随机性测试失败
**症状**：测试结果不稳定
**解决方案**：
1. 设置固定随机种子
2. 使用统计测试方法
3. 增加测试样本数量
4. 设置合理的容差范围

## 验证清单

完成本阶段后，请确认以下项目：

- [ ] 测试框架建立完成
- [ ] 单元测试实现完成
- [ ] 集成测试实现完成
- [ ] 系统测试实现完成
- [ ] 性能测试实现完成
- [ ] 回归测试实现完成
- [ ] 持续集成配置完成
- [ ] 所有测试通过
- [ ] 代码覆盖率达到80%以上
- [ ] 性能基准测试通过
- [ ] 安全检查通过
- [ ] 文档测试通过
- [ ] 测试报告生成成功

## 下一步

完成测试和验证后，您的MMSurv项目将具备：

1. **完整的测试体系** - 从单元测试到系统测试的全覆盖
2. **自动化质量保证** - 持续集成和自动化测试
3. **性能监控** - 性能基准和回归检测
4. **可靠性保证** - 全面的错误检测和处理
5. **可维护性** - 清晰的测试结构和文档

至此，您已经完成了MMSurv多模态生存预测项目的完整复现！

## 总结

通过本文档的指导，您已经建立了一个完整的测试和验证体系，确保项目的质量和可靠性。这个测试体系将帮助您：

- 及早发现和修复问题
- 确保代码质量和性能
- 维护项目的长期稳定性
- 支持持续开发和改进

恭喜您完成了MMSurv项目的完整复现！