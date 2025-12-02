import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# 1. 计算模型准确率 (Accuracy)

def calculate_heuristics_modular(df):
    # 避免修改原始数据
    df = df.copy()
    
    # 预处理：生成 CW 模型需要的 "上一轮得分" (prev_score)
    score_cols = ['score_cat1', 'score_cat2', 'score_cat3', 'score_cat4']
    for col in score_cols:
        df[f'prev_{col}'] = df.groupby('subject_id')[col].shift(1)
        
    results = []
    
    # 遍历每个被试进行计算
    for sub_id, sub_df in df.groupby('subject_id'):
        # --- A. 静态偏好模型 (Likeability) ---
        # 检查数据有效性 (User 4 这种无方差用户会被跳过)
        if not sub_df['valid_preference_data'].iloc[0]:
            acc_like = np.nan
        else:
            # 找出评分最高的选项
            ratings = sub_df[['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']].fillna(-1).values
            is_max = (ratings == ratings.max(axis=1)[:, None])
            # 计算命中率
            acc_like = is_max[np.arange(len(sub_df)), sub_df['category_idx'].values - 1].mean()
        
        # --- B. 动态模型 (从第 2 轮开始) ---
        dyn = sub_df[sub_df['trial'] > 1]
        
        if len(dyn) == 0:
            acc_wsls = acc_cw = acc_stick = acc_gf = np.nan
        else:
            # 预计算通用向量
            wins = dyn['prev_reward'] >= 3
            stay = dyn['category_idx'] == dyn['prev_choice']
            
            # 1. WSLS (赢留输变)
            acc_wsls = ((wins & stay) | (~wins & ~stay)).mean()
            
            # 2. Stickiness (粘性/重复)
            acc_stick = stay.mean()
            
            # 3. Gambler's Fallacy (赌徒谬误: 赢变输留)
            acc_gf = ((wins & ~stay) | (~wins & stay)).mean()
            
            # 4. Chasing Winner (CW: 追逐赢家)
            prev_scores = dyn[[f'prev_{c}' for c in score_cols]].values
            is_winner = (prev_scores == prev_scores.max(axis=1)[:, None])
            acc_cw = is_winner[np.arange(len(dyn)), dyn['category_idx'].values - 1].mean()

        results.append({
            'subject_id': sub_id, 
            'acc_like': acc_like, 'acc_wsls': acc_wsls, 
            'acc_cw': acc_cw, 'acc_stickiness': acc_stick, 'acc_gambler': acc_gf
        })
        
    return pd.DataFrame(results)

# ==========================================
# 2. 将 Accuracy 转换为 BIC
# ==========================================
def convert_acc_to_bic(acc_df, raw_df):
    # 1. 获取每个用户的 Trial 数量
    trial_counts = raw_df.groupby('subject_id')['trial'].count().reset_index(name='n_total')
    trial_counts['n_dyn'] = trial_counts['n_total'] - 1 # 动态模型少一轮
    
    # 合并到 ACC 表
    merged_df = acc_df.merge(trial_counts, on='subject_id', how='left')
    
    # 2. 设定参数
    log_epsilon = np.log(0.01) # 错误惩罚 (-4.605)
    
    # 3. 向量化计算函数
    # BIC = -2 * LogLikelihood = -2 * (N_miss * log(0.01))
    def calc_bic_vectorized(acc_series, n_series):
        n_miss = n_series * (1 - acc_series)
        return -2 * (n_miss * log_epsilon)

    # 4. 执行计算
    bic_df = pd.DataFrame({'subject_id': merged_df['subject_id']})
    
    # 计算 Likeability BIC (基于 n_total)
    if 'acc_like' in merged_df.columns:
        bic_df['bic_like'] = calc_bic_vectorized(merged_df['acc_like'], merged_df['n_total'])
    
    # 计算动态模型 BIC (基于 n_dyn)
    dyn_models = ['wsls', 'cw', 'stickiness', 'gambler']
    for m in dyn_models:
        col_acc = f'acc_{m}'
        col_bic = f'bic_{m}'
        if col_acc in merged_df.columns:
            bic_df[col_bic] = calc_bic_vectorized(merged_df[col_acc], merged_df['n_dyn'])
        
    return bic_df

df_master = pd.read_csv('D:/Development/Data/BCAI/PCA/02_Master_df.csv')


acc_df = calculate_heuristics_modular(df_master)

# Step 2: 转换为 BIC
bic_df = convert_acc_to_bic(acc_df, df_master)



def calculate_random_bic(df):
    """
    计算随机选择模型的 BIC 作为基准线 (Baseline)。
    假设: P(choice) = 0.25, k = 0
    公式: BIC = -2 * N * ln(0.25)
    """
    # 1. 获取每个被试的 Trial 总数
    # 注意：这里应该用 n_total (40)，因为随机策略不需要历史数据，适用于所有 Trial
    trial_counts = df.groupby('subject_id')['trial'].count().reset_index(name='n_total')
    
    # 2. 计算 BIC
    log_p = np.log(0.25) # 约等于 -1.386
    trial_counts['bic_random'] = -2 * trial_counts['n_total'] * log_p
    
    return trial_counts

# 加载数据
df_master = pd.read_csv('D:/Development/Data/BCAI/PCA/02_Master_df.csv')

# 计算
random_bic_df = calculate_random_bic(df_master)

print(">>> 随机策略 BIC 基准值 (Baseline):")
print(random_bic_df.head())
print(f"\n平均随机 BIC: {random_bic_df['bic_random'].mean():.2f}")

bic_df.to_csv("/Development/Data/BCAI/PCA/04.3_heuristic_results.csv", index=False)



    
results_df = pd.read_csv('D:/Development/Data/BCAI/PCA/03_heuristic_results.csv')

# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif' # 确保兼容性

# 准备绘图数据 (Long Format 适合 Boxplot)
plot_cols = ['acc_like', 'acc_wsls', 'acc_cw', 'acc_stickiness', 'acc_gambler']
df_melted = results_df.melt(id_vars='subject_id', value_vars=plot_cols, 
                            var_name='Model', value_name='Accuracy')

# 创建画布：2x2 的子图布局
fig, axes = plt.subplots(2, 2, figsize=(18, 12))


ax1 = axes[0, 0]
sns.boxplot(data=df_melted, x='Model', y='Accuracy', palette="Set3", ax=ax1, showfliers=False)
sns.swarmplot(data=df_melted, x='Model', y='Accuracy', color=".25", size=4, alpha=0.7, ax=ax1)
ax1.set_title('A. Distribution of Model Accuracies', fontsize=14, fontweight='bold')
ax1.axhline(0.25, ls='--', color='red', label='Random Chance (0.25)')
ax1.set_xlabel('')
ax1.legend(loc='upper right')

ax2 = axes[0, 1]
corr_matrix = results_df[plot_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax2)
ax2.set_title('B. Correlation Matrix of Strategies', fontsize=14, fontweight='bold')


ax3 = axes[1, 0]
# ci=68 等同于 Standard Error (SE)
sns.barplot(data=df_melted, x='Model', y='Accuracy', palette="viridis", errorbar=('ci', 68), capsize=.1, ax=ax3)
ax3.set_title('C. Mean Accuracy (Error Bar = SE)', fontsize=14, fontweight='bold')
ax3.axhline(0.25, ls='--', color='red')
ax3.set_ylim(0, 0.65) # 根据数据范围调整
ax3.set_xlabel('')


ax4 = axes[1, 1]
sns.scatterplot(data=results_df, x='acc_wsls', y='acc_like', s=120, color='purple', alpha=0.6, edgecolor='k', ax=ax4)

# 添加辅助线和标注
ax4.set_xlabel('WSLS Accuracy (Strategy-driven)', fontsize=12)
ax4.set_ylabel('Likeability Accuracy (Preference-driven)', fontsize=12)
ax4.set_title('D. Individual Strategy Profile', fontsize=14, fontweight='bold')

# 标注特征明显的被试 ID (例如准确率 > 0.6 的人)
for i in range(results_df.shape[0]):
    x_val = results_df.acc_wsls.iloc[i]
    y_val = results_df.acc_like.iloc[i]
    sub_id = results_df.subject_id.iloc[i]
    
    # 忽略 NaN (User 4) 和中间的普通点，只标注极端的
    if not np.isnan(y_val) and (y_val > 0.6 or x_val > 0.6):
        ax4.text(x_val + 0.01, y_val + 0.01, str(sub_id), fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show() # 或者 plt.savefig('heuristic_dashboard.png')
