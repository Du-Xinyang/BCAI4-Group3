import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=== 行为经济学实验数据分析 ===")
# 读取数据
df = pd.read_csv('exptal data 2025.csv')

print("数据基本信息:")
print(f"数据形状: {df.shape}")
print(f"用户数量: {df['User ID'].nunique()}")
print("\n前5行数据:")
print(df.head())

# 数据清洗
df_clean = df.copy()
df_clean = df_clean[df_clean['Score'] != -1]  # 移除无效数据

print(f"\n清洗后数据形状: {df_clean.shape}")

# 定义类别映射
def map_item_to_category(item_id):
    """将物品ID映射到类别"""
    if 1 <= item_id <= 5:
        return 'Category1'  # Colors
    elif 6 <= item_id <= 10:
        return 'Category2'  # Abstract Concepts
    elif 11 <= item_id <= 15:
        return 'Category3'  # Disciplines
    elif 16 <= item_id <= 20:
        return 'Category4'  # Places
    else:
        return 'Unknown'

# 分离喜好度评分和选择任务
likert_data = df_clean[df_clean['Question ID'] <= 20].copy()
choice_data = df_clean[df_clean['Question ID'] >= 21].copy()

# 为选择数据添加类别信息
choice_data['Category'] = choice_data['Answer'].apply(map_item_to_category)

print(f"喜好度数据: {likert_data.shape}")
print(f"选择任务数据: {choice_data.shape}")

# 类别统计信息
category_stats = {
    'Category': ['Category1', 'Category2', 'Category3', 'Category4'],
    'Mean': [2.5, 2.5, 2.0, 3.0],
    'Std': [1.13, 0.51, 0.0, 0.91],
    'Description': ['高风险中等回报', '低风险中等回报', '无风险低回报', '中等风险高回报']
}

stats_df = pd.DataFrame(category_stats)
print("=== 类别统计信息 ===")
print(stats_df)

# 选择频率分析
choice_freq = choice_data['Category'].value_counts().sort_index()
choice_percent = choice_freq / choice_freq.sum() * 100

print("\n=== 总体选择频率 ===")
for cat, freq, pct in zip(choice_freq.index, choice_freq.values, choice_percent.values):
    print(f"{cat}: {freq}次 ({pct:.1f}%)")


# Create visualization function
def plot_choice_analysis(choice_data):
    """Plot choice analysis charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Overall choice distribution
    choice_counts = choice_data['Category'].value_counts()
    axes[0, 0].pie(choice_counts.values, labels=choice_counts.index, autopct='%1.1f%%',
                   colors=['#f6dfaf', '#e3ccc2', '#8ca5c0', '#b9c8da'], textprops={'fontsize': 15})
    axes[0, 0].set_title('Overall Choice Distribution', fontsize=18, fontweight='bold')

    # 2. Choice frequency over time
    choice_data['Trial_Number'] = choice_data.groupby('User ID').cumcount() + 1
    choice_data['Block'] = np.where(choice_data['Trial_Number'] <= 20, 'First Half', 'Second Half')

    block_choice = pd.crosstab(choice_data['Block'], choice_data['Category'], normalize='index') * 100
    block_choice.plot(kind='bar', ax=axes[0, 1], color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    axes[0, 1].set_title('Choice Proportion Changes Between Halves', fontweight='bold')
    axes[0, 1].set_ylabel('Choice Proportion (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Individual choice pattern heatmap
    user_choice_matrix = pd.crosstab(choice_data['User ID'], choice_data['Category'], normalize='index')
    sns.heatmap(user_choice_matrix, ax=axes[1, 0], cmap='YlOrRd', cbar_kws={'label': 'Choice Proportion'})
    axes[1, 0].set_title('Individual Choice Pattern Heatmap', fontweight='bold')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('User ID')

    plt.tight_layout()
    plt.show()


# Execute visualization
plot_choice_analysis(choice_data)


def analyze_learning_effect(choice_data):
    """Analyze learning effect"""
    print("=== Learning Effect Analysis ===")

    # Block analysis
    choice_data['Trial_Block'] = (choice_data['Trial_Number'] - 1) // 10 + 1  # Every 10 trials as a block

    block_choice_pct = pd.crosstab(choice_data['Trial_Block'], choice_data['Category'],
                                   normalize='index') * 100
    colors = ['#c58b8a', '#244367', '#b3b5c6', '#f3c093']  # 618198
    # Plot learning curve
    plt.figure(figsize=(9, 5))
    for i, category in enumerate(['Category1', 'Category2', 'Category3', 'Category4']):
        plt.plot(block_choice_pct.index, block_choice_pct[category],
                 marker='o', label=category, linewidth=2,
                 color=colors[i])  # ,colors=['#e3ccc2','#8ca5c0','#b9c8da','#f6dfaf']

    plt.xlabel('Trial Block (10 trials per block)', fontsize=15)
    plt.ylabel('Choice Percentage (%)', fontsize=15)
    plt.title('Learning Curve of Choice Patterns', fontweight='bold', fontsize=18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Statistical test
    first_half = choice_data[choice_data['Block'] == 'First Half']
    second_half = choice_data[choice_data['Block'] == 'Second Half']

    print("\nChoice Percentage Changes Between First and Second Half:")
    first_counts = first_half['Category'].value_counts(normalize=True).sort_index()
    second_counts = second_half['Category'].value_counts(normalize=True).sort_index()

    change_df = pd.DataFrame({
        'First Half': first_counts * 100,
        'Second Half': second_counts * 100,
        'Change': (second_counts - first_counts) * 100
    })
    print(change_df.round(2))


analyze_learning_effect(choice_data)