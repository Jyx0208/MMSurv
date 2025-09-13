import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 创建简化的结果图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Multi-Modal Survival Prediction Models - Key Results', fontsize=16, fontweight='bold')

# 数据
models = ['PORPOISE', 'MCAT', 'DeepSet', 'AMIL', 'MOTCat']
test_scores = [0.4286, 0.7857, 0.6429, 0.7857, 0.7500]
val_scores = [1.0000, 1.0000, 1.0000, 0.0000, 1.0000]
parameters = [943300, 3636934, 1000132, 1263045, 3373766]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

# 1. Test C-Index Performance
bars1 = ax1.bar(models, test_scores, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Test C-Index', fontsize=12)
ax1.set_title('Test Performance (C-Index)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, score in zip(bars1, test_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2. Model Size (Parameters)
bars2 = ax2.bar(models, [p/1000000 for p in parameters], color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Parameters (Millions)', fontsize=12)
ax2.set_title('Model Complexity', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 添加参数标签
for bar, param in zip(bars2, parameters):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{param/1000000:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Performance vs Complexity Scatter
scatter = ax3.scatter([p/1000000 for p in parameters], test_scores, 
                     c=colors, s=300, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_xlabel('Model Size (Million Parameters)', fontsize=12)
ax3.set_ylabel('Test C-Index', fontsize=12)
ax3.set_title('Performance vs Complexity', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 添加模型标签
for i, model in enumerate(models):
    ax3.annotate(model, (parameters[i]/1000000, test_scores[i]),
                xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))

# 4. Architecture Summary
ax4.axis('off')
summary_text = """
KEY FINDINGS:

🏆 BEST PERFORMERS:
   • MCAT: 0.7857 (3.6M params)
   • AMIL: 0.7857 (1.3M params)
   
⚡ MOST EFFICIENT:
   • PORPOISE: 0.4286 (0.9M params)
   
🔬 DATASET:
   • 30 patients, 76 slides
   • 4 modalities: RNA + DNA + CNV + Pathology
   • 50 omics features + 768-dim images
   
📊 ARCHITECTURES TESTED:
   ✓ Multiple Instance Learning (MIL)
   ✓ Co-Attention Transformer
   ✓ DeepSet
   ✓ Attention-based MIL
   ✓ Optimal Transport Co-Attention
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/train/MMSurv/mmsurv/model_results_summary.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Summary results saved as 'model_results_summary.png'")

# 创建详细的结果表格
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# 表格数据
table_data = [
    ['PORPOISE', 'MIL', '1.0000', '0.4286', '943K', 'Fast, Efficient'],
    ['MCAT', 'Co-Attention', '1.0000', '0.7857', '3.6M', 'High Performance'],
    ['DeepSet', 'Set Function', '1.0000', '0.6429', '1.0M', 'Theoretical Foundation'],
    ['AMIL', 'Attention-MIL', '0.0000', '0.7857', '1.3M', 'Attention Mechanism'],
    ['MOTCat', 'Optimal Transport', '1.0000', '0.7500', '3.4M', 'Cross-modal Alignment']
]

columns = ['Model', 'Architecture', 'Validation\nC-Index', 'Test\nC-Index', 'Parameters', 'Key Features']

table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)

# 设置表格样式
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#2E8B57')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 根据性能着色
for i, row in enumerate(table_data, 1):
    test_score = float(row[3])
    if test_score >= 0.75:
        color = '#90EE90'  # 浅绿色 - 高性能
    elif test_score >= 0.6:
        color = '#FFE4B5'  # 浅橙色 - 中等性能  
    else:
        color = '#FFB6C1'  # 浅红色 - 低性能
        
    for j in range(len(columns)):
        table[(i, j)].set_facecolor(color)

ax.set_title('Multi-Modal Survival Prediction Models - Detailed Results', 
             fontsize=16, fontweight='bold', pad=20)

plt.savefig('/home/train/MMSurv/mmsurv/model_results_table.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Detailed table saved as 'model_results_table.png'")
plt.show()

print("\n" + "="*60)
print("MULTI-MODAL SURVIVAL PREDICTION - FINAL RESULTS")
print("="*60)
print("✅ Successfully tested 5 different model architectures")
print("✅ All models support multi-modal data (Images + Omics)")
print("✅ Added pathology modality (20 features)")
print("✅ Generated comprehensive virtual dataset (30 patients)")
print("="*60)