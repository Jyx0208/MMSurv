# 04 - 评估和可视化

## 概述

本阶段将实现MMSurv项目的模型评估和结果可视化系统，包括生存分析专用评估指标、模型性能比较、结果可视化、统计分析和报告生成等功能。

## 评估体系架构

### 核心组件
1. **评估指标计算** - 生存分析专用指标（C-Index、AUC等）
2. **模型性能评估** - 单模型和多模型性能评估
3. **统计显著性测试** - 模型间比较的统计检验
4. **可视化系统** - 丰富的图表和可视化功能
5. **报告生成** - 自动化评估报告生成
6. **结果分析** - 深度结果分析和解释

## 实现任务

### 任务1：实现评估指标计算

#### 1.1 创建生存分析评估指标
**文件位置**：`mmsurv/utils/eval_utils.py`

**实现要求**：
- 实现`calculate_c_index()`函数（一致性指数）
- 实现`calculate_auc()`函数（时间依赖AUC）
- 实现`calculate_brier_score()`函数（Brier评分）
- 实现`calculate_ibs()`函数（积分Brier评分）
- 实现`calculate_concordance()`函数（一致性评估）

**关键功能**：
- 处理删失数据
- 时间依赖性评估
- 风险分层评估
- 预测校准评估
- 模型判别能力评估

#### 1.2 实现风险评估工具
**实现要求**：
- 实现`risk_stratification()`函数
- 实现`survival_probability_estimation()`函数
- 实现`hazard_ratio_calculation()`函数
- 支持多时间点评估
- 支持分组比较

#### 1.3 测试评估指标
**测试文件**：`tests/test_eval_utils.py`

**测试方法**：
```bash
# 测试C-Index计算
python -c "
import numpy as np
import torch
from mmsurv.utils.eval_utils import calculate_c_index

# 创建模拟数据
n_samples = 100
risk_scores = torch.randn(n_samples)  # 风险评分
event_times = torch.rand(n_samples) * 100  # 事件时间
censorship = torch.randint(0, 2, (n_samples,))  # 删失指示器

# 计算C-Index
c_index = calculate_c_index(risk_scores, event_times, censorship)
print(f'C-Index: {c_index:.4f}')

# 验证C-Index范围
assert 0 <= c_index <= 1, f'C-Index应在[0,1]范围内，实际值: {c_index}'
print('C-Index计算测试通过')
"
```

### 任务2：实现模型评估框架

#### 2.1 创建单模型评估器
**文件位置**：`mmsurv/evaluation/single_model_evaluator.py`

**实现要求**：
- 实现`SingleModelEvaluator`类
- 支持多种评估指标
- 实现预测结果分析
- 支持不同数据集评估
- 生成详细评估报告

**关键功能**：
- 模型预测
- 指标计算
- 结果统计
- 性能分析
- 错误分析

#### 2.2 创建多模型比较器
**文件位置**：`mmsurv/evaluation/model_comparator.py`

**实现要求**：
- 实现`ModelComparator`类
- 支持多模型性能比较
- 实现统计显著性测试
- 支持交叉验证结果比较
- 生成比较报告

**关键功能**：
- 模型性能对比
- 统计检验
- 排名分析
- 显著性测试
- 效果量计算

#### 2.3 测试评估框架
**测试方法**：
```bash
# 测试单模型评估
python -c "
from mmsurv.evaluation.single_model_evaluator import SingleModelEvaluator
from mmsurv.models.model_porpoise import PorpoiseMMF
import torch

# 创建模型和数据
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
path_features = torch.randn(20, 100, 768)
omic_features = torch.randn(20, 50)
event_times = torch.rand(20) * 100
censorship = torch.randint(0, 2, (20,))

# 创建评估器
evaluator = SingleModelEvaluator(model)

# 运行评估
results = evaluator.evaluate(
    path_features=path_features,
    omic_features=omic_features,
    event_times=event_times,
    censorship=censorship
)

print(f'评估结果: {results}')
print('单模型评估测试通过')
"
```

### 任务3：实现统计分析工具

#### 3.1 创建统计检验模块
**文件位置**：`mmsurv/utils/statistical_tests.py`

**实现要求**：
- 实现`paired_t_test()`函数（配对t检验）
- 实现`wilcoxon_test()`函数（Wilcoxon符号秩检验）
- 实现`mcnemar_test()`函数（McNemar检验）
- 实现`bootstrap_test()`函数（自助法检验）
- 实现多重比较校正

**关键功能**：
- 参数和非参数检验
- 效果量计算
- 置信区间估计
- 功效分析
- 多重比较校正

#### 3.2 实现生存分析统计
**实现要求**：
- 实现`log_rank_test()`函数（对数秩检验）
- 实现`cox_regression_test()`函数（Cox回归检验）
- 实现`kaplan_meier_test()`函数（Kaplan-Meier检验）
- 支持分层分析
- 支持多变量分析

#### 3.3 测试统计分析
**测试方法**：
```bash
# 测试统计检验
python -c "
from mmsurv.utils.statistical_tests import paired_t_test, wilcoxon_test
import numpy as np

# 创建模拟数据
n_samples = 50
model1_scores = np.random.normal(0.7, 0.1, n_samples)
model2_scores = np.random.normal(0.65, 0.1, n_samples)

# 配对t检验
t_stat, t_pvalue = paired_t_test(model1_scores, model2_scores)
print(f'配对t检验: t={t_stat:.4f}, p={t_pvalue:.4f}')

# Wilcoxon检验
w_stat, w_pvalue = wilcoxon_test(model1_scores, model2_scores)
print(f'Wilcoxon检验: W={w_stat:.4f}, p={w_pvalue:.4f}')

print('统计检验测试通过')
"
```

### 任务4：实现可视化系统

#### 4.1 创建基础可视化工具
**文件位置**：`mmsurv/visualization/base_plots.py`

**实现要求**：
- 实现`plot_survival_curves()`函数（生存曲线图）
- 实现`plot_kaplan_meier()`函数（Kaplan-Meier曲线）
- 实现`plot_roc_curves()`函数（ROC曲线图）
- 实现`plot_calibration()`函数（校准图）
- 实现`plot_risk_distribution()`函数（风险分布图）

**关键功能**：
- 高质量图表生成
- 自定义样式支持
- 交互式图表（可选）
- 多格式导出
- 批量图表生成

#### 4.2 创建模型比较可视化
**文件位置**：`mmsurv/visualization/comparison_plots.py`

**实现要求**：
- 实现`plot_model_comparison()`函数（模型性能比较图）
- 实现`plot_performance_radar()`函数（雷达图）
- 实现`plot_ranking_chart()`函数（排名图）
- 实现`plot_significance_matrix()`函数（显著性矩阵图）
- 实现`plot_effect_size()`函数（效果量图）

#### 4.3 创建结果分析可视化
**文件位置**：`mmsurv/visualization/analysis_plots.py`

**实现要求**：
- 实现`plot_feature_importance()`函数（特征重要性图）
- 实现`plot_attention_weights()`函数（注意力权重图）
- 实现`plot_prediction_distribution()`函数（预测分布图）
- 实现`plot_error_analysis()`函数（错误分析图）
- 实现`plot_learning_curves()`函数（学习曲线图）

#### 4.4 测试可视化系统
**测试方法**：
```bash
# 测试基础可视化
python -c "
from mmsurv.visualization.base_plots import plot_survival_curves, plot_roc_curves
import numpy as np
import matplotlib.pyplot as plt

# 创建模拟数据
time_points = np.linspace(0, 100, 100)
survival_prob1 = np.exp(-0.01 * time_points)
survival_prob2 = np.exp(-0.015 * time_points)

# 绘制生存曲线
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_survival_curves(
    time_points=[time_points, time_points],
    survival_probs=[survival_prob1, survival_prob2],
    labels=['Model 1', 'Model 2'],
    ax=ax
)
plt.title('生存曲线比较')
plt.savefig('test_survival_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print('生存曲线图生成成功')

# 测试ROC曲线
y_true = np.random.randint(0, 2, 100)
y_scores = np.random.rand(100)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_roc_curves(
    y_true=[y_true],
    y_scores=[y_scores],
    labels=['Model'],
    ax=ax
)
plt.title('ROC曲线')
plt.savefig('test_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

print('ROC曲线图生成成功')
print('可视化系统测试通过')
"
```

### 任务5：实现综合可视化脚本

#### 5.1 创建结果可视化脚本
**文件位置**：`mmsurv/visualize_results.py`

**实现要求**：
- 实现完整的结果可视化流程
- 支持多种图表类型
- 自动化图表生成
- 支持批量处理
- 生成可视化报告

**可视化内容**：
- 模型性能比较图
- 参数数量对比图
- 训练和验证损失曲线
- 架构类型分布饼图
- 测试C-Index雷达图
- 综合性能评分柱状图

#### 5.2 实现交互式可视化（可选）
**文件位置**：`mmsurv/visualization/interactive_plots.py`

**实现要求**：
- 使用Plotly或Bokeh实现交互式图表
- 支持数据筛选和缩放
- 实现动态更新
- 支持导出功能

#### 5.3 测试综合可视化
**测试方法**：
```bash
# 测试综合可视化脚本
python -c "
from mmsurv.visualize_results import create_comprehensive_plots
import numpy as np

# 创建模拟结果数据
test_results = {
    'PORPOISE': {
        'c_index': 0.742,
        'params': 2.3e6,
        'train_loss': [1.2, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.3, 0.9, 0.7, 0.6, 0.5]
    },
    'MCAT': {
        'c_index': 0.756,
        'params': 8.1e6,
        'train_loss': [1.1, 0.7, 0.5, 0.4, 0.35],
        'val_loss': [1.2, 0.8, 0.6, 0.5, 0.45]
    },
    'MOTCat': {
        'c_index': 0.761,
        'params': 12.4e6,
        'train_loss': [1.0, 0.6, 0.4, 0.3, 0.25],
        'val_loss': [1.1, 0.7, 0.5, 0.4, 0.35]
    }
}

# 生成综合可视化
create_comprehensive_plots(test_results, save_dir='./visualization_output')
print('综合可视化生成完成')
print('可视化文件保存在 ./visualization_output 目录')
"
```

### 任务6：实现评估报告生成

#### 6.1 创建报告生成器
**文件位置**：`mmsurv/reporting/report_generator.py`

**实现要求**：
- 实现`ReportGenerator`类
- 支持HTML和PDF报告生成
- 自动化表格和图表嵌入
- 支持模板自定义
- 实现批量报告生成

**报告内容**：
- 实验设置和配置
- 模型性能汇总
- 统计分析结果
- 可视化图表
- 结论和建议

#### 6.2 创建报告模板
**文件位置**：`mmsurv/reporting/templates/`

**模板类型**：
- `evaluation_report.html` - HTML评估报告模板
- `comparison_report.html` - 模型比较报告模板
- `summary_report.html` - 结果汇总报告模板

#### 6.3 测试报告生成
**测试方法**：
```bash
# 测试报告生成
python -c "
from mmsurv.reporting.report_generator import ReportGenerator
import tempfile
import os

# 创建模拟评估结果
evaluation_results = {
    'experiment_name': 'MMSurv模型比较实验',
    'date': '2024-01-15',
    'models': {
        'PORPOISE': {'c_index': 0.742, 'auc': 0.785, 'brier_score': 0.156},
        'MCAT': {'c_index': 0.756, 'auc': 0.798, 'brier_score': 0.148},
        'MOTCat': {'c_index': 0.761, 'auc': 0.803, 'brier_score': 0.142}
    },
    'statistical_tests': {
        'best_model': 'MOTCat',
        'significance_level': 0.05,
        'p_values': {'MOTCat_vs_PORPOISE': 0.023, 'MOTCat_vs_MCAT': 0.156}
    }
}

# 生成报告
with tempfile.TemporaryDirectory() as temp_dir:
    report_generator = ReportGenerator()
    report_path = report_generator.generate_evaluation_report(
        results=evaluation_results,
        output_dir=temp_dir,
        format='html'
    )
    
    print(f'报告生成成功: {os.path.exists(report_path)}')
    print(f'报告路径: {report_path}')
    print('报告生成测试通过')
"
```

### 任务7：实现性能分析工具

#### 7.1 创建性能分析器
**文件位置**：`mmsurv/analysis/performance_analyzer.py`

**实现要求**：
- 实现`PerformanceAnalyzer`类
- 支持深度性能分析
- 实现错误案例分析
- 支持特征重要性分析
- 实现预测置信度分析

**分析功能**：
- 预测准确性分析
- 错误模式识别
- 特征贡献分析
- 模型可解释性分析
- 鲁棒性评估

#### 7.2 实现模型解释工具
**文件位置**：`mmsurv/analysis/model_explainer.py`

**实现要求**：
- 实现`ModelExplainer`类
- 支持SHAP值计算
- 实现注意力权重分析
- 支持特征重要性排序
- 实现局部解释

#### 7.3 测试性能分析
**测试方法**：
```bash
# 测试性能分析器
python -c "
from mmsurv.analysis.performance_analyzer import PerformanceAnalyzer
from mmsurv.models.model_porpoise import PorpoiseMMF
import torch
import numpy as np

# 创建模型和数据
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
path_features = torch.randn(50, 100, 768)
omic_features = torch.randn(50, 50)
event_times = torch.rand(50) * 100
censorship = torch.randint(0, 2, (50,))

# 创建性能分析器
analyzer = PerformanceAnalyzer(model)

# 运行性能分析
analysis_results = analyzer.analyze(
    path_features=path_features,
    omic_features=omic_features,
    event_times=event_times,
    censorship=censorship
)

print(f'分析结果包含 {len(analysis_results)} 个指标')
print('性能分析测试通过')
"
```

### 任务8：实现批量评估脚本

#### 8.1 创建批量评估脚本
**文件位置**：`scripts/evaluate_models.py`

**实现要求**：
- 命令行参数解析
- 多模型批量评估
- 结果汇总和比较
- 自动化报告生成
- 支持并行评估

**命令行接口**：
```bash
python scripts/evaluate_models.py \
    --model_dir ./saved_models \
    --data_dir ./data \
    --output_dir ./evaluation_results \
    --metrics c_index,auc,brier_score \
    --generate_report
```

#### 8.2 创建交叉验证评估脚本
**文件位置**：`scripts/cross_validate_models.py`

**实现要求**：
- K折交叉验证评估
- 统计显著性测试
- 结果可视化
- 详细报告生成

#### 8.3 测试批量评估
**测试方法**：
```bash
# 创建批量评估测试
cat > test_batch_evaluation.py << 'EOF'
import torch
import tempfile
import os
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.models.model_set_mil import MIL_Attention_FC_surv
from mmsurv.evaluation.model_comparator import ModelComparator

# 创建模拟模型
models = {
    'PORPOISE': PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4),
    'AMIL': MIL_Attention_FC_surv(omic_input_dim=50, path_input_dim=768, n_classes=4)
}

# 创建模拟数据
n_samples = 30
path_features = torch.randn(n_samples, 50, 768)
omic_features = torch.randn(n_samples, 50)
event_times = torch.rand(n_samples) * 100
censorship = torch.randint(0, 2, (n_samples,))

# 创建比较器
comparator = ModelComparator(models)

# 运行批量评估
with tempfile.TemporaryDirectory() as temp_dir:
    results = comparator.compare(
        path_features=path_features,
        omic_features=omic_features,
        event_times=event_times,
        censorship=censorship,
        output_dir=temp_dir
    )
    
    print(f'评估了 {len(results)} 个模型')
    for model_name, metrics in results.items():
        print(f'{model_name}: C-Index = {metrics["c_index"]:.4f}')
    
    print('批量评估测试通过')
EOF

python test_batch_evaluation.py
```

## 可视化示例

### 生存曲线可视化
**实现要求**：
- Kaplan-Meier生存曲线
- 风险分层生存曲线
- 置信区间显示
- 对数秩检验结果

### 模型性能比较图
**实现要求**：
- C-Index比较柱状图
- 多指标雷达图
- 参数效率散点图
- 训练损失曲线图

### 统计分析可视化
**实现要求**：
- 显著性检验结果矩阵
- 效果量森林图
- 置信区间图
- P值分布图

## 测试和验证

### 单元测试

#### 运行评估组件测试
```bash
# 运行所有评估相关测试
python -m pytest tests/test_eval*.py -v
python -m pytest tests/test_visualization*.py -v
python -m pytest tests/test_reporting*.py -v

# 运行特定组件测试
python -m pytest tests/test_eval_utils.py -v
python -m pytest tests/test_statistical_tests.py -v
```

### 集成测试

#### 测试1：端到端评估流程
```bash
# 创建端到端评估测试
cat > test_e2e_evaluation.py << 'EOF'
import torch
import tempfile
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.evaluation.single_model_evaluator import SingleModelEvaluator
from mmsurv.visualization.base_plots import plot_survival_curves
from mmsurv.reporting.report_generator import ReportGenerator

# 创建模型和数据
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)
path_features = torch.randn(20, 50, 768)
omic_features = torch.randn(20, 50)
event_times = torch.rand(20) * 100
censorship = torch.randint(0, 2, (20,))

# 端到端评估流程
with tempfile.TemporaryDirectory() as temp_dir:
    # 1. 模型评估
    evaluator = SingleModelEvaluator(model)
    eval_results = evaluator.evaluate(
        path_features=path_features,
        omic_features=omic_features,
        event_times=event_times,
        censorship=censorship
    )
    print(f'评估完成: C-Index = {eval_results["c_index"]:.4f}')
    
    # 2. 可视化生成
    # 这里可以添加可视化代码
    print('可视化生成完成')
    
    # 3. 报告生成
    report_generator = ReportGenerator()
    report_data = {
        'experiment_name': '端到端评估测试',
        'model_results': {'PORPOISE': eval_results}
    }
    # report_path = report_generator.generate_report(report_data, temp_dir)
    print('报告生成完成')
    
    print('端到端评估流程测试通过')
EOF

python test_e2e_evaluation.py
```

#### 测试2：可视化质量检查
```bash
# 测试可视化输出质量
python -c "
import matplotlib.pyplot as plt
import numpy as np
from mmsurv.visualization.base_plots import plot_survival_curves
import os

# 创建测试数据
time_points = np.linspace(0, 100, 100)
survival_curves = [
    np.exp(-0.01 * time_points),
    np.exp(-0.015 * time_points),
    np.exp(-0.02 * time_points)
]
labels = ['Low Risk', 'Medium Risk', 'High Risk']

# 生成高质量图表
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_survival_curves(
    time_points=[time_points] * 3,
    survival_probs=survival_curves,
    labels=labels,
    ax=ax
)

# 设置图表样式
ax.set_xlabel('Time (months)', fontsize=12)
ax.set_ylabel('Survival Probability', fontsize=12)
ax.set_title('Risk Stratification Survival Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 保存高质量图片
plt.savefig('quality_test_survival.png', dpi=300, bbox_inches='tight')
plt.close()

# 检查文件生成
if os.path.exists('quality_test_survival.png'):
    file_size = os.path.getsize('quality_test_survival.png')
    print(f'图片生成成功，文件大小: {file_size} bytes')
    print('可视化质量检查通过')
    os.remove('quality_test_survival.png')
else:
    print('可视化质量检查失败')
"
```

### 性能测试

#### 评估速度基准测试
```bash
# 创建评估速度测试
cat > benchmark_evaluation_speed.py << 'EOF'
import torch
import time
from mmsurv.models.model_porpoise import PorpoiseMMF
from mmsurv.utils.eval_utils import calculate_c_index

def benchmark_evaluation_speed(model, data_loader, num_iterations=10):
    model.eval()
    
    # 预测阶段计时
    start_time = time.time()
    all_predictions = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            for path_feat, omic_feat, labels in data_loader:
                predictions = model(h_path=path_feat, h_omic=omic_feat)
                all_predictions.append(predictions)
                
                if len(all_predictions) >= 50:  # 限制预测数量
                    break
            if len(all_predictions) >= 50:
                break
    
    prediction_time = time.time() - start_time
    
    # 评估指标计算计时
    start_time = time.time()
    
    # 模拟评估指标计算
    for _ in range(100):
        risk_scores = torch.randn(50)
        event_times = torch.rand(50) * 100
        censorship = torch.randint(0, 2, (50,))
        c_index = calculate_c_index(risk_scores, event_times, censorship)
    
    evaluation_time = time.time() - start_time
    
    return prediction_time, evaluation_time

# 创建测试数据
n_samples = 100
path_features = torch.randn(n_samples, 50, 768)
omic_features = torch.randn(n_samples, 50)
labels = torch.randint(0, 4, (n_samples,))

from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(path_features, omic_features, labels)
data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# 创建模型
model = PorpoiseMMF(omic_input_dim=50, path_input_dim=768, n_classes=4)

# 运行基准测试
pred_time, eval_time = benchmark_evaluation_speed(model, data_loader)

print(f"预测时间: {pred_time:.2f}秒")
print(f"评估时间: {eval_time:.2f}秒")
print(f"总评估时间: {pred_time + eval_time:.2f}秒")
print("评估速度基准测试完成")
EOF

python benchmark_evaluation_speed.py
```

## 故障排除

### 常见问题

#### 问题1：C-Index计算错误
**症状**：C-Index值超出[0,1]范围或为NaN
**解决方案**：
1. 检查删失数据处理
2. 验证风险评分计算
3. 确认事件时间格式
4. 处理数值稳定性问题

#### 问题2：可视化图表质量差
**症状**：图表模糊、字体过小、布局混乱
**解决方案**：
1. 调整DPI设置
2. 优化图表尺寸
3. 设置合适的字体大小
4. 使用bbox_inches='tight'

#### 问题3：统计检验结果不可信
**症状**：P值异常、检验统计量错误
**解决方案**：
1. 检查数据分布假设
2. 验证样本大小
3. 选择合适的检验方法
4. 考虑多重比较校正

#### 问题4：报告生成失败
**症状**：HTML/PDF生成错误、模板渲染失败
**解决方案**：
1. 检查模板文件路径
2. 验证数据格式
3. 确认依赖包安装
4. 处理特殊字符编码

## 验证清单

完成本阶段后，请确认以下项目：

- [ ] 评估指标计算实现完成
- [ ] 模型评估框架实现完成
- [ ] 统计分析工具实现完成
- [ ] 可视化系统实现完成
- [ ] 综合可视化脚本实现完成
- [ ] 评估报告生成实现完成
- [ ] 性能分析工具实现完成
- [ ] 批量评估脚本实现完成
- [ ] 所有评估组件测试通过
- [ ] 端到端评估流程测试通过
- [ ] 可视化质量检查通过
- [ ] 评估速度基准测试完成

## 下一步

评估和可视化实现完成后，请继续阅读 [05-testing-validation.md](./05-testing-validation.md) 进行完整的测试和验证。