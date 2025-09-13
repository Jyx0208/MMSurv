import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置样式
plt.style.use('default')
sns.set_palette("husl")

# 测试结果数据
results_data = {
    'Model': ['PORPOISE', 'MCAT', 'DeepSet', 'AMIL', 'MOTCat'],
    'Validation_C_Index': [1.0000, 1.0000, 1.0000, 0.0000, 1.0000],
    'Test_C_Index': [0.4286, 0.7857, 0.6429, 0.7857, 0.7500],
    'Parameters': [943300, 3636934, 1000132, 1263045, 3373766],
    'Training_Loss': [1.1574, 1.7405, 1.6588, 1.6260, 1.7236],
    'Validation_Loss': [1.3337, 1.3313, 0.9919, 1.3051, 1.3366],
    'Architecture_Type': ['MIL', 'Co-Attention', 'DeepSet', 'Attention-MIL', 'Optimal Transport']
}

df = pd.DataFrame(results_data)

# 创建综合结果图表
fig = plt.figure(figsize=(20, 15))

# 1. C-Index Performance Comparison (Top Left)
ax1 = plt.subplot(3, 3, 1)
x_pos = np.arange(len(df['Model']))
width = 0.35
colors_val = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
colors_test = ['#2980b9', '#c0392b', '#27ae60', '#d68910', '#8e44ad']

bars1 = ax1.bar(x_pos - width/2, df['Validation_C_Index'], width, 
                label='Validation C-Index', color=colors_val, alpha=0.7)
bars2 = ax1.bar(x_pos + width/2, df['Test_C_Index'], width,
                label='Test C-Index', color=colors_test, alpha=0.9)

ax1.set_xlabel('Models', fontsize=12)
ax1.set_ylabel('C-Index', fontsize=12)
ax1.set_title('C-Index Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df['Model'], rotation=45)
ax1.legend()
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    val1, val2 = df['Validation_C_Index'][i], df['Test_C_Index'][i]
    ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
             f'{val1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
             f'{val2:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Model Complexity (Top Center)
ax2 = plt.subplot(3, 3, 2)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
bars = ax2.bar(df['Model'], df['Parameters'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax2.set_xlabel('Models', fontsize=12)
ax2.set_ylabel('Number of Parameters', fontsize=12)
ax2.set_title('Model Complexity (Parameters)', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# 添加参数数量标签
for bar, val in zip(bars, df['Parameters']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80000,
             f'{val/1000000:.1f}M' if val >= 1000000 else f'{val/1000:.0f}K', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 3. Loss Comparison (Top Right)
ax3 = plt.subplot(3, 3, 3)
x_pos = np.arange(len(df['Model']))
bars1 = ax3.bar(x_pos - width/2, df['Training_Loss'], width,
                label='Training Loss', color='lightgreen', alpha=0.8, edgecolor='darkgreen')
bars2 = ax3.bar(x_pos + width/2, df['Validation_Loss'], width,
                label='Validation Loss', color='orange', alpha=0.8, edgecolor='darkorange')

ax3.set_xlabel('Models', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(df['Model'], rotation=45)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Test C-Index Horizontal Bar Chart (Middle Left)
ax4 = plt.subplot(3, 3, 4)
bars = ax4.barh(df['Model'], df['Test_C_Index'], color=colors, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Test C-Index', fontsize=12)
ax4.set_title('Test C-Index Performance', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# 添加数值标签
for bar, score in zip(bars, df['Test_C_Index']):
    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{score:.4f}', ha='left', va='center', fontsize=11, fontweight='bold')

# 5. Architecture Distribution (Middle Center)
ax5 = plt.subplot(3, 3, 5)
arch_counts = df['Architecture_Type'].value_counts()
wedges, texts, autotexts = ax5.pie(arch_counts.values, labels=arch_counts.index, autopct='%1.1f%%',
        colors=colors[:len(arch_counts)], startangle=90, explode=[0.05]*len(arch_counts))
ax5.set_title('Architecture Type Distribution', fontsize=14, fontweight='bold')

# 美化饼图文本
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 6. Performance vs Complexity Scatter (Middle Right)
ax6 = plt.subplot(3, 3, 6)
scatter = ax6.scatter(df['Parameters']/1000000, df['Test_C_Index'], 
                     c=colors, s=200, alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_xlabel('Model Size (Million Parameters)', fontsize=12)
ax6.set_ylabel('Test C-Index', fontsize=12)
ax6.set_title('Performance vs Model Complexity', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 添加模型名称标签
for i, model in enumerate(df['Model']):
    ax6.annotate(model, (df['Parameters'][i]/1000000, df['Test_C_Index'][i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

# 7. Radar Chart for All Metrics (Bottom Left)
ax7 = plt.subplot(3, 3, 7, projection='polar')
metrics = ['Test C-Index', 'Efficiency', 'Stability']
num_models = len(df['Model'])

# 归一化指标 (越高越好)
normalized_cindex = df['Test_C_Index']
normalized_efficiency = 1 - (df['Parameters'] / df['Parameters'].max())  # 参数越少越好
normalized_stability = 1 - (df['Validation_Loss'] / df['Validation_Loss'].max())  # 损失越低越好

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合

for i, model in enumerate(df['Model']):
    values = [normalized_cindex[i], normalized_efficiency[i], normalized_stability[i]]
    values += values[:1]  # 闭合
    
    ax7.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax7.fill(angles, values, alpha=0.1, color=colors[i])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(metrics)
ax7.set_ylim(0, 1)
ax7.set_title('Multi-Metric Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax7.grid(True)

# 8. Training Summary Table (Bottom Center & Right)
ax8 = plt.subplot(3, 3, (8, 9))
ax8.axis('tight')
ax8.axis('off')

# 创建详细表格数据
table_data = []
for i, row in df.iterrows():
    table_data.append([
        row['Model'],
        row['Architecture_Type'],
        f"{row['Validation_C_Index']:.4f}",
        f"{row['Test_C_Index']:.4f}",
        f"{row['Parameters']:,}",
        f"{row['Training_Loss']:.3f}",
        f"{row['Validation_Loss']:.3f}"
    ])

columns = ['Model', 'Architecture', 'Val C-Index', 'Test C-Index', 'Parameters', 'Train Loss', 'Val Loss']

# 创建表格
table = ax8.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 设置表格样式
for i in range(len(columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data) + 1):
    for j in range(len(columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        table[(i, j)].set_text_props(weight='normal')

ax8.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)

plt.suptitle('Multi-Modal Survival Prediction Models Comparison', 
             fontsize=18, fontweight='bold', y=0.95)

plt.tight_layout()
plt.savefig('/home/train/MMSurv/mmsurv/model_comparison_results.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Results visualization saved as 'model_comparison_results.png'")

# 打印详细结果
print("\n" + "="*80)
print("MULTI-MODAL SURVIVAL PREDICTION MODELS - TEST RESULTS")
print("="*80)

print(f"\n{'Model':<12} {'Architecture':<15} {'Val C-Index':<12} {'Test C-Index':<13} {'Parameters':<12} {'Train Loss':<11} {'Val Loss':<10}")
print("-" * 80)

for _, row in df.iterrows():
    print(f"{row['Model']:<12} {row['Architecture_Type']:<15} {row['Validation_C_Index']:<12.4f} "
          f"{row['Test_C_Index']:<13.4f} {row['Parameters']:<12,} {row['Training_Loss']:<11.3f} {row['Validation_Loss']:<10.3f}")

print("\n" + "="*80)
print("MODEL ANALYSIS SUMMARY")
print("="*80)

# 性能排名
test_ranking = df.nlargest(5, 'Test_C_Index')
print(f"\n🏆 TOP PERFORMERS (Test C-Index):")
for i, (_, row) in enumerate(test_ranking.iterrows(), 1):
    print(f"{i}. {row['Model']}: {row['Test_C_Index']:.4f}")

# 效率排名
efficiency_ranking = df.nsmallest(5, 'Parameters')
print(f"\n⚡ MOST EFFICIENT (Parameters):")
for i, (_, row) in enumerate(efficiency_ranking.iterrows(), 1):
    print(f"{i}. {row['Model']}: {row['Parameters']:,} parameters")

# 模型特征
print(f"\n📊 KEY FINDINGS:")
print(f"• Best Test Performance: MCAT & AMIL (0.7857)")
print(f"• Most Efficient: PORPOISE (943K parameters)")
print(f"• Most Complex: MCAT (3.6M parameters)")
print(f"• Lowest Validation Loss: DeepSet (0.9919)")
print(f"• All models except AMIL achieved perfect validation C-Index (1.0000)")

print(f"\n🔬 DATASET INFO:")
print(f"• Patients: 30")
print(f"• Slides: 76")
print(f"• Modalities: RNA(5) + DNA(10) + CNV(15) + Pathology(20)")
print(f"• Total Omics Features: 50")
print(f"• Image Features: 768-dim per patch")
print(f"• Survival Time Bins: 4 discrete intervals")

print("="*80)