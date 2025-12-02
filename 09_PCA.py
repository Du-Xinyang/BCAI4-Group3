import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os


# 在这里修改文件路径的前缀
base_dir = r'D:/Development/Data/BCAI/PCA'
rl_result_dir = os.path.join(base_dir, 'RL_results')
# =========================================

# 1. 读取数据
df_heuristics = pd.read_csv(os.path.join(base_dir, '03_heuristic_results.csv'))
df_std = pd.read_csv(os.path.join(rl_result_dir, '00_RL.csv'))
df_RL_PT = pd.read_csv(os.path.join(rl_result_dir, '01_RL_PT.csv'))
df_pt = pd.read_csv(os.path.join(rl_result_dir, '02_PT.csv'))
df_hybrid = pd.read_csv(os.path.join(rl_result_dir, '03_hybrid.csv'))

# 2. 构建特征矩阵 (Feature Matrix)
# 提取启发式模型准确率
master_features = df_heuristics[['subject_id', 'acc_like', 'acc_wsls', 'acc_cw', 'acc_gambler']].copy()

# 合并 Standard RL 参数
df_std_renamed = df_std[['subject_id', 'alpha', 'beta']].rename(columns={'alpha': 'alpha_std', 'beta': 'beta_std'})
master_features = master_features.merge(df_std_renamed, on='subject_id', how='left')

# 合并 PT 参数 (使用 lambda)
df_RL_PT_renamed = df_RL_PT[['subject_id', 'lambda']].rename(columns={'lambda': 'lambda_pt'})
master_features = master_features.merge(df_RL_PT_renamed, on='subject_id', how='left')

# 合并 PT 参数 (使用 lambda)
df_pt_renamed = df_pt[['subject_id', 'gamma','alpha']].rename(columns={'gamma': 'gamma_pt','alpha':'alpha_pt'})
master_features = master_features.merge(df_pt_renamed, on='subject_id', how='left')

# 合并 Hybrid 参数 (使用 omega)
df_hybrid_renamed = df_hybrid[['subject_id', 'omega']].rename(columns={'omega': 'omega_hybrid'})
master_features = master_features.merge(df_hybrid_renamed, on='subject_id', how='left')

# 3. 预处理
# 填充 NaN: 对于准确率，使用列均值填充 (表示平均水平)
master_features = master_features.fillna(master_features.mean())
#     'acc_like', 'acc_wsls', 'acc_cw', 
# 定义用于聚类的特征列
feature_cols = ['acc_like', 'acc_wsls', 'acc_cw', 
    'alpha_std', 'beta_std', 'gamma_pt','alpha_pt','omega_hybrid'
]
X = master_features[feature_cols]

# 标准化 (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 4. K-Means 聚类

### 设置分为几类
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
master_features['cluster'] = clusters

# 5. PCA 降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
master_features['pca_1'] = X_pca[:, 0]
master_features['pca_2'] = X_pca[:, 1]

# 绘制聚类图
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=master_features, x='pca_1', y='pca_2', 
    hue='cluster', palette='viridis', s=100, style='cluster'
)

# 标注几个点

plt.title(f'Subject Clustering (K={k}) based on Strategy Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('clustering_plot.png')
plt.show()

# 6. 打印聚类中心统计
cluster_summary = master_features.groupby('cluster')[feature_cols].mean()
print("Cluster Centers (Mean Values):")
print(cluster_summary)

# 保存带有聚类标签的结果
master_features.to_csv('clustered_subjects.csv', index=False)
print("\n结果已保存至 clustered_subjects.csv")