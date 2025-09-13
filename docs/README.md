# MMSurv 项目复现文档

本目录包含了完整复现 MMSurv 多模态生存预测项目的详细文档。

## 文档结构

按照以下顺序阅读和执行复现步骤：

1. **[00-overview.md](./00-overview.md)** - 项目概述和环境准备
2. **[01-data-preparation.md](./01-data-preparation.md)** - 数据准备和预处理
3. **[02-model-implementation.md](./02-model-implementation.md)** - 模型架构实现
4. **[03-training-pipeline.md](./03-training-pipeline.md)** - 训练流程实现
5. **[04-evaluation-visualization.md](./04-evaluation-visualization.md)** - 评估和可视化
6. **[05-testing-validation.md](./05-testing-validation.md)** - 完整测试和验证

## 使用说明

- 每个文档都包含详细的步骤说明和测试方法
- 请按照文档顺序逐步执行
- 每完成一个阶段，务必进行相应的测试验证
- 如遇到问题，请参考各文档中的故障排除部分

## 项目特点

- **多模态融合**：结合病理图像和基因组数据
- **生存预测**：离散时间生存分析
- **多种模型**：支持6种不同的深度学习架构
- **完整流程**：从数据预处理到模型评估的端到端解决方案