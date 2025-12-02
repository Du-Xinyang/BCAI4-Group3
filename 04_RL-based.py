
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import inf
import warnings
import os


# 忽略 log(0) 产生的 RuntimeWarning，在代码中已通过 max(p, 1e-12) 处理
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ==========================================
# 1. 核心辅助函数 & 鲁棒性逻辑 (源自 model_build_fit.py)
# ==========================================

def softmax(x, beta=1.0):
    """
    Softmax function with stability adjustment.
    """
    # 减去最大值防止指数爆炸
    ex = np.exp(beta * (x - np.max(x)))
    return ex / ex.sum()

def calculate_bic(nll, n_params, n_observations):
    """Calculate Bayesian Information Criterion (BIC)"""
    return 2 * nll + n_params * np.log(n_observations)

def prepare_arrays_robust(sub_df):
    """
    数据预处理：确保无NaN值，提取choice和reward。
    兼容 RL_model.py 的 category_idx (1-4) 或 0-3 索引。
    """
    # 确定列名
    choice_col = 'category_idx' if 'category_idx' in sub_df.columns else 'choice'
    reward_col = 'reward'
    
    # 过滤 NaN
    mask = ~sub_df[choice_col].isna() & ~sub_df[reward_col].isna()
    clean_df = sub_df[mask].copy()
    
    choices = clean_df[choice_col].astype(int).values
    rewards = clean_df[reward_col].astype(float).values
    
    # 确保 choices 是 1-4 格式 (用于内部逻辑统一)
    if choices.min() == 0:
        choices = choices + 1
        
    return choices, rewards

# ==========================================
# 2. RL 模型算法 (Rescorla-Wagner)
# ==========================================

def negloglik_rl_robust(params, choices, rewards, n_options=4, q0=2.5):
    """
    负对数似然函数：RL
    Params: [alpha, log_beta] (使用 log_beta 确保 beta > 0)
    """
    alpha, logbeta = params
    
    # 硬约束检查 (虽然 L-BFGS-B 有边界，双重保险)
    if not (0 <= alpha <= 1):
        return 1e9
        
    beta = np.exp(logbeta) # 还原 beta
    Q = np.ones(n_options) * q0
    nll = 0.0
    
    for c, r in zip(choices, rewards):
        # c is 1-4, map to 0-3
        c_idx = c - 1
        
        probs = softmax(Q, beta=beta)
        p = probs[c_idx]
        
        # 避免 log(0)
        p = max(p, 1e-12)
        nll -= np.log(p)
        
        # 更新 Q 值
        Q[c_idx] = Q[c_idx] + alpha * (r - Q[c_idx])
        
    return nll

def fit_robust_rl_model(df):
    """
    模块化函数：拟合 RL 模型
    算法源自 model_build_fit.py (使用 log-space beta 和特定边界)
    """
    print("正在拟合模型: Robust Standard RL (Source: model_build_fit.py)...")
    results = []
    
    # 遍历每个被试
    # 兼容 subject_id 或 user 列名
    user_col = 'subject_id' if 'subject_id' in df.columns else 'user'
    
    for sub_id, sub_df in df.groupby(user_col):
        choices, rewards = prepare_arrays_robust(sub_df)
        
        if len(choices) < 5: # 数据过少跳过
            continue

        # === 鲁棒性配置 (Strictly from model_build_fit.py) ===
        # 初始值: alpha=0.3, beta=1.0 (log(1.0)=0)
        x0 = np.array([0.3, np.log(1.0)])
        
        # 边界: alpha [0, 1], beta [1e-3, 1e3] (log space)
        bounds = [(0.0, 1.0), (np.log(1e-3), np.log(1e3))]
        
        # 优化
        res = minimize(
            lambda x: negloglik_rl_robust(x, choices, rewards), 
            x0, 
            bounds=bounds, 
            method='L-BFGS-B'
        )
        
        if res.success:
            alpha = res.x[0]
            beta = float(np.exp(res.x[1])) # 还原 beta
            nll = res.fun
        else:
            alpha, beta, nll = np.nan, np.nan, np.nan
            
        # 计算 BIC (k=2: alpha, beta)
        bic = calculate_bic(nll, 2, len(choices)) if not np.isnan(nll) else np.nan
        
        results.append({
            'subject_id': sub_id,
            'alpha': alpha,
            'beta': beta,
            'nll': nll,
            'bic': bic
        })
        
    return pd.DataFrame(results)

# ==========================================
# 3. PT 模型算法 (Static Prospect Theory)
# ==========================================

def compute_empirical_category_distributions(df):
    """
    计算全局经验分布 (用于 PT 模型的概率加权)
    """
    dists = {}
    # 确保有 reward 和 cat/category_idx 列
    cat_col = 'category_idx' if 'category_idx' in df.columns else 'cat'
    # 如果 category_idx 是 0-3，映射到 1-4
    if df[cat_col].min() == 0:
        df = df.copy()
        df[cat_col] = df[cat_col] + 1
    
    for cat in [1, 2, 3, 4]:
        vals = df[df[cat_col] == cat]['reward'].dropna().values
        if len(vals) == 0:
            dists[cat] = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        else:
            unique, counts = np.unique(vals, return_counts=True)
            probs = counts / counts.sum()
            dists[cat] = (unique, probs)
    return dists

def prelec_weight(p, gamma):
    """Prelec (1998) probability weighting function"""
    p = np.clip(p, 1e-12, 1.0)
    return np.exp(-(-np.log(p)) ** gamma)

def subjective_EV_for_category(cat, alpha_val, gamma_val, category_dists):
    """计算特定类别的由 PT 调整后的主观期望值"""
    outcomes, probs = category_dists[cat]
    
    # Value function: u(x) = x^alpha
    # 注意：这里假设 reward >= 0，如果存在负数需调整逻辑，源代码假设 x^alpha
    u = outcomes ** alpha_val
    
    # Probability weighting
    w = prelec_weight(probs, gamma_val)
    
    # Normalize weights (approximated)
    if w.sum() == 0:
        w = probs
    else:
        w = w / w.sum()
        
    return np.sum(w * u)

def negloglik_pt_robust(params, choices, category_dists):
    """
    负对数似然函数：Static PT
    Params: [log_alpha, log_gamma, log_beta]
    """
    log_alpha, log_gamma, log_beta = params
    
    alpha_val = np.exp(log_alpha)
    gamma_val = np.exp(log_gamma)
    beta = np.exp(log_beta)
    
    # 预计算每个类别的静态 sEV
    sEV = np.array([subjective_EV_for_category(cat, alpha_val, gamma_val, category_dists) 
                    for cat in [1, 2, 3, 4]])
    
    nll = 0.0
    for c in choices:
        c_idx = c - 1 # map 1-4 to 0-3
        probs = softmax(sEV, beta=beta)
        p = probs[c_idx]
        p = max(p, 1e-12)
        nll -= np.log(p)
        
    return nll

def fit_robust_pt_model(df):
    """
    模块化函数：拟合 Static PT 模型
    算法源自 model_build_fit.py
    """
    print("正在拟合模型: Robust Static PT (Source: model_build_fit.py)...")
    results = []
    
    # 1. 计算全局经验分布 (Pooled)
    category_dists = compute_empirical_category_distributions(df)
    
    user_col = 'subject_id' if 'subject_id' in df.columns else 'user'
    
    for sub_id, sub_df in df.groupby(user_col):
        choices, _ = prepare_arrays_robust(sub_df) # PT 不需要逐试次的 reward，需要全局分布
        
        if len(choices) < 5:
            continue
            
        # === 鲁棒性配置 (Strictly from model_build_fit.py) ===
        # Initial logs: alpha=0.9, gamma=0.9, beta=1.0
        x0 = np.array([np.log(0.9), np.log(0.9), np.log(1.0)])
        
        # Bounds (all in log space)
        # alpha: [0.01, 5.0], gamma: [0.01, 5.0], beta: [0.001, 1000]
        bounds = [
            (np.log(1e-2), np.log(5.0)), 
            (np.log(1e-2), np.log(5.0)), 
            (np.log(1e-3), np.log(1e3))
        ]
        
        res = minimize(
            lambda x: negloglik_pt_robust(x, choices, category_dists),
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if res.success:
            alpha = float(np.exp(res.x[0]))
            gamma = float(np.exp(res.x[1]))
            beta = float(np.exp(res.x[2]))
            nll = res.fun
        else:
            alpha, gamma, beta, nll = np.nan, np.nan, np.nan, np.nan
            
        # BIC (k=3: alpha, gamma, beta)
        bic = calculate_bic(nll, 3, len(choices)) if not np.isnan(nll) else np.nan
        
        results.append({
            'subject_id': sub_id,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'nll': nll,
            'bic': bic
        })
        
    return pd.DataFrame(results)

# 2. 模型: RL + Prospect Theory (Robust Version)
def fit_rl_pt(df):
    print("正在拟合模型 2/4: RL + Prospect Theory (Robust) ...")
    results = []
    GAMMA = 0.8
    REF_POINT = 2.5
    
    # NLL: Params = [alpha, log_beta, log_lambda]
    def get_nll_robust(params, choices, rewards):
        alpha, log_beta, log_lamb = params
        if not (0 <= alpha <= 1): return 1e9
        
        beta = np.exp(log_beta)
        lamb = np.exp(log_lamb)
        
        q_values = np.full(4, 2.5)
        nll = 0.0
        
        for c, r in zip(choices, rewards):
            c_idx = int(c) - 1
            # Softmax
            probs = softmax(q_values, beta=beta)
            nll -= np.log(max(probs[c_idx], 1e-12))
            
            # PT Utility
            utility = (r - REF_POINT)**GAMMA if r >= REF_POINT else -lamb * ((REF_POINT - r)**GAMMA)
            
            # Update
            q_values[c_idx] += alpha * (utility - q_values[c_idx])
        return nll

    for sub_id, sub_df in df.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue
        
        # Initial: alpha=0.5, beta=2.0, lambda=1.0
        x0 = [0.5, np.log(2.0), np.log(1.0)]
        # Bounds: alpha[0,1], beta[0.001, 1000], lambda[0.001, 100]
        bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (np.log(1e-3), np.log(1e1))]
        
        res = minimize(lambda x: get_nll_robust(x, choices, rewards), x0, bounds=bounds, method='L-BFGS-B')
        
        if res.success:
            results.append({
                'subject_id': sub_id,
                'alpha': res.x[0],
                'beta': np.exp(res.x[1]),
                'lambda': np.exp(res.x[2]),
                'nll': res.fun,
                'bic': calculate_bic(res.fun, 3, len(choices))
            })
    return pd.DataFrame(results)

# 3. 模型: Hybrid RL + Likeability (Robust Version)
def fit_hybrid_likeability(df):
    print("正在拟合模型 3/4: Hybrid RL + Likeability (Robust) ...")
    
    # 标准化 Likeability (保持原逻辑)
    df_std = df.copy()
    like_cols = ['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']
    for sub_id, group in df_std.groupby('subject_id'):
        vals = group[like_cols].values.flatten()
        std_val = np.nanstd(vals)
        if std_val == 0: std_val = 1
        df_std.loc[df_std['subject_id'] == sub_id, like_cols] = (group[like_cols] - np.nanmean(vals)) / std_val
    
    results = []
    
    # NLL: Params = [alpha, log_beta, omega] (omega 可正可负，不需要 log)
    def get_nll_robust(params, choices, rewards, like_matrix):
        alpha, log_beta, omega = params
        if not (0 <= alpha <= 1): return 1e9
        
        beta = np.exp(log_beta)
        q_values = np.full(4, 2.5)
        nll = 0.0
        
        for i, (c, r) in enumerate(zip(choices, rewards)):
            c_idx = int(c) - 1
            likes = like_matrix[i]
            
            # V = beta*Q + omega*L
            v_values = (beta * q_values) + (omega * likes)
            
            probs = softmax(v_values, beta=1.0) # beta 已融入 v_values
            nll -= np.log(max(probs[c_idx], 1e-12))
            
            q_values[c_idx] += alpha * (r - q_values[c_idx])
        return nll

    for sub_id, sub_df in df_std.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue
        
        like_matrix = sub_df[like_cols].values
        # 确保 like_matrix 长度与 cleaned choices 匹配
        if len(like_matrix) != len(choices):
            # 简单处理：重新对其索引 (假设 sub_df 已经是 clean 的，或者需要重新切片)
            mask = ~sub_df['category_idx'].isna() & ~sub_df['reward'].isna()
            like_matrix = sub_df.loc[mask, like_cols].values
            
        x0 = [0.5, np.log(2.0), 0.5]
        bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (-10, 10)]
        
        res = minimize(lambda x: get_nll_robust(x, choices, rewards, like_matrix), x0, bounds=bounds, method='L-BFGS-B')
        
        if res.success:
            results.append({
                'subject_id': sub_id,
                'alpha': res.x[0],
                'beta': np.exp(res.x[1]),
                'omega': res.x[2],
                'nll': res.fun,
                'bic': calculate_bic(res.fun, 3, len(choices))
            })
    return pd.DataFrame(results)



df_master = pd.read_csv('D:/Development/Data/BCAI/PCA/02_Master_df.csv')

df_standard_rl = fit_robust_rl_model(df_master)
df_rl_pt = fit_rl_pt(df_master)
df_pt = fit_robust_pt_model(df_master)
df_hybrid = fit_hybrid_likeability(df_master)


# 3. 保存结果
# ------------------------------------------------
output_dir = 'D:/Development/Data/BCAI/PCA/RL_results'
os.makedirs(output_dir, exist_ok=True)

df_standard_rl.to_csv(os.path.join(output_dir, '00_RL.csv'), index=False)
df_rl_pt.to_csv(os.path.join(output_dir, '01_RL_PT1.csv'), index=False)
df_hybrid.to_csv(os.path.join(output_dir, '03_hybrid.csv'), index=False)
df_pt.to_csv(os.path.join(output_dir, '03_PT2.csv'), index=False)