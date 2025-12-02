import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import inf
import warnings
import os


warnings.filterwarnings('ignore', category=RuntimeWarning)
# ==========================================
# 1. Core functions
# ==========================================

def softmax(x, beta=1.0):
    """
    Softmax function with stability adjustment.
    """
    # Subtract the maximum value to prevent index explosion
    ex = np.exp(beta * (x - np.max(x)))
    return ex / ex.sum()

def calculate_bic(nll, n_params, n_observations):
    """Calculate Bayesian Information Criterion (BIC)"""
    return 2 * nll + n_params * np.log(n_observations)

def prepare_arrays_robust(sub_df):
    """
    Data preprocessing: Ensure no NaN values, extract choices and rewards.
    Compatible with categorizy_idx (1-4) or 0-3 indexes of RL_madel. py.
    """
    # define col names
    choice_col = 'category_idx' if 'category_idx' in sub_df.columns else 'choice'
    reward_col = 'reward'

    # filter NaN
    mask = ~sub_df[choice_col].isna() & ~sub_df[reward_col].isna()
    clean_df = sub_df[mask].copy()

    choices = clean_df[choice_col].astype(int).values
    rewards = clean_df[reward_col].astype(float).values


    if choices.min() == 0:
        choices = choices + 1

    return choices, rewards

# ==========================================
# 2. RL MODEL algorithms (Rescorla-Wagner)
# ==========================================

def negloglik_rl_robust(params, choices, rewards, n_options=4, q0=2.5):
    """
    Negative log likelihood function:RL
    Params: [alpha, log_beta] (use log_beta to make sure beta > 0)
    """
    alpha, logbeta = params

    # Hard constraint check (although L-BFGS-B has boundaries, double insurance)
    if not (0 <= alpha <= 1):
        return 1e9

    beta = np.exp(logbeta)
    Q = np.ones(n_options) * q0
    nll = 0.0

    for c, r in zip(choices, rewards):
        # c is 1-4, map to 0-3
        c_idx = c - 1

        probs = softmax(Q, beta=beta)
        p = probs[c_idx]

        # avoid log(0)
        p = max(p, 1e-12)
        nll -= np.log(p)

        # updata Q value
        Q[c_idx] = Q[c_idx] + alpha * (r - Q[c_idx])

    return nll

def fit_robust_rl_model(df):
    """
    fit RL model

    """
    print("正在拟合模型: Robust Standard RL (Source: model_build_fit.py)...")
    results = []

    # Traverse each subject

    user_col = 'subject_id' if 'subject_id' in df.columns else 'user'

    for sub_id, sub_df in df.groupby(user_col):
        choices, rewards = prepare_arrays_robust(sub_df)

        if len(choices) < 5:
            continue


        x0 = np.array([0.3, np.log(1.0)])

        # bound: alpha [0, 1], beta [1e-3, 1e3] (log space)
        bounds = [(0.0, 1.0), (np.log(1e-3), np.log(1e3))]

        # optimize
        res = minimize(
            lambda x: negloglik_rl_robust(x, choices, rewards),
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if res.success:
            alpha = res.x[0]
            beta = float(np.exp(res.x[1]))
            nll = res.fun
        else:
            alpha, beta, nll = np.nan, np.nan, np.nan

        # BIC (k=2: alpha, beta)
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
# 3. PT Model (Static Prospect Theory)
# ==========================================

def compute_empirical_category_distributions(df):
    """
    Calculate the global empirical distribution (used for probability weighting in PT models)
    """
    dists = {}

    cat_col = 'category_idx' if 'category_idx' in df.columns else 'cat'

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
    """Calculate the subjective expected value adjusted by PT for a specific category"""
    outcomes, probs = category_dists[cat]

    # Value function: u(x) = x^alpha

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
    nll：Static PT
    Params: [log_alpha, log_gamma, log_beta]
    """
    log_alpha, log_gamma, log_beta = params

    alpha_val = np.exp(log_alpha)
    gamma_val = np.exp(log_gamma)
    beta = np.exp(log_beta)

    # sEV
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
    fit Static PT model

    """
    print("正在拟合模型: Robust Static PT (Source: model_build_fit.py)...")
    results = []

    # 1.  (Pooled)
    category_dists = compute_empirical_category_distributions(df)

    user_col = 'subject_id' if 'subject_id' in df.columns else 'user'

    for sub_id, sub_df in df.groupby(user_col):
        choices, _ = prepare_arrays_robust(sub_df) # PT

        if len(choices) < 5:
            continue


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

# 2. Model: RL + Prospect Theory (Robust Version)
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
            utility = (r - REF_POINT )**GAMMA if r >= REF_POINT else -lamb * ((REF_POINT - r )**GAMMA)

            # Update
            q_values[c_idx] += alpha * (utility - q_values[c_idx])
        return nll

    for sub_id, sub_df in df.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue

        # Initial: alpha=0.5, beta=2.0, lambda=1.0
        x0 = [0.5, np.log(2.0), np.log(1.0)]
        # Bounds: alpha[0,1], beta[0.001, 1000], lambda[0.001, 100]
        bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (np.log(1e-3), np.log(1e2))]

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

# 3. Model: Hybrid RL + Likeability (Robust Version)
def fit_hybrid_likeability(df):
    print("正在拟合模型 3/4: Hybrid RL + Likeability (Robust) ...")

    # normalize Likeability
    df_std = df.copy()
    like_cols = ['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']
    for sub_id, group in df_std.groupby('subject_id'):
        vals = group[like_cols].values.flatten()
        std_val = np.nanstd(vals)
        if std_val == 0: std_val = 1
        df_std.loc[df_std['subject_id'] == sub_id, like_cols] = (group[like_cols] - np.nanmean(vals)) / std_val

    results = []

    # NLL: Params = [alpha, log_beta, omega]
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

            probs = softmax(v_values, beta=1.0)
            nll -= np.log(max(probs[c_idx], 1e-12))

            q_values[c_idx] += alpha * (r - q_values[c_idx])
        return nll

    for sub_id, sub_df in df_std.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue

        like_matrix = sub_df[like_cols].values

        if len(like_matrix) != len(choices):

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

# 4.  RL with Perseveration (Robust Version)
def fit_perseveration_rl(df):
    print("正在拟合模型 4/4: RL with Perseveration (Robust) ...")
    results = []

    # NLL: Params = [alpha, log_beta, kappa]
    def get_nll_robust(params, choices, rewards):
        alpha, log_beta, kappa = params
        if not (0 <= alpha <= 1): return 1e9

        beta = np.exp(log_beta)
        q_values = np.full(4, 2.5)
        nll = 0.0
        prev_choice = -1

        for c, r in zip(choices, rewards):
            c_idx = int(c) - 1

            # V = beta*Q + kappa*I
            v_values = beta * q_values
            if prev_choice != -1:
                v_values[prev_choice] += kappa

            probs = softmax(v_values, beta=1.0)
            nll -= np.log(max(probs[c_idx], 1e-12))

            q_values[c_idx] += alpha * (r - q_values[c_idx])
            prev_choice = c_idx
        return nll

    for sub_id, sub_df in df.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue

        x0 = [0.5, np.log(2.0), 0.5]
        bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (-5, 5)]

        res = minimize(lambda x: get_nll_robust(x, choices, rewards), x0, bounds=bounds, method='L-BFGS-B')

        if res.success:
            results.append({
                'subject_id': sub_id,
                'alpha': res.x[0],
                'beta': np.exp(res.x[1]),
                'kappa': res.x[2],
                'nll': res.fun,
                'bic': calculate_bic(res.fun, 3, len(choices))
            })
    return pd.DataFrame(results)














############################################Check the significance of parameters (calculate SE and p values)
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# ==========================================
# Enhanced Helper Functions with SE/P-value calculation
# ==========================================
df_master = pd.read_csv('02_Master_df.csv')


def calculate_se_pvalues(res, nll_func, choices, rewards=None, **kwargs):
    """
    Calculate standard errors and p-values using Hessian matrix

    Parameters:
    -----------
    res : OptimizeResult
        Result from scipy.optimize.minimize
    nll_func : callable
        Negative log-likelihood function
    choices : array
        Choice data
    rewards : array, optional
        Reward data (for RL models)
    **kwargs : dict
        Additional arguments for nll_func (e.g., category_dists, like_matrix)

    Returns:
    --------
    se_dict : dict
        Dictionary of standard errors for each parameter
    pval_dict : dict
        Dictionary of p-values for each parameter
    """
    if not res.success:
        return {}, {}

    # Compute Hessian using finite differences
    params = res.x
    epsilon = 1e-5
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))

    # Prepare arguments for nll_func
    if rewards is not None:
        args = (choices, rewards)
    else:
        args = (choices,)

    # Add kwargs if present
    for key, value in kwargs.items():
        args = args + (value,)

    # Compute Hessian numerically
    for i in range(n_params):
        for j in range(n_params):
            params_pp = params.copy()
            params_pm = params.copy()
            params_mp = params.copy()
            params_mm = params.copy()

            params_pp[i] += epsilon
            params_pp[j] += epsilon
            params_pm[i] += epsilon
            params_pm[j] -= epsilon
            params_mp[i] -= epsilon
            params_mp[j] += epsilon
            params_mm[i] -= epsilon
            params_mm[j] -= epsilon

            hessian[i, j] = (
                                    nll_func(params_pp, *args) -
                                    nll_func(params_pm, *args) -
                                    nll_func(params_mp, *args) +
                                    nll_func(params_mm, *args)
                            ) / (4 * epsilon ** 2)

    # Calculate variance-covariance matrix (inverse of Hessian)
    try:
        # Add small ridge to diagonal for numerical stability
        hessian_stable = hessian + np.eye(n_params) * 1e-6
        var_cov = np.linalg.inv(hessian_stable)
        se = np.sqrt(np.diag(var_cov))

        # Calculate z-scores and p-values (two-tailed test)
        z_scores = params / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    except np.linalg.LinAlgError:
        # If Hessian is singular, return NaN
        se = np.full(n_params, np.nan)
        p_values = np.full(n_params, np.nan)

    return se, p_values


def format_significance(p_value):
    """Format p-value with significance stars"""
    if np.isnan(p_value):
        return ''
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '+'
    else:
        return ''


# ==========================================
# Modified RL Model with SE/P-values
# ==========================================

def fit_robust_rl_model_with_stats(df):
    """
    Fit RL model with standard errors and p-values
    """
    print("正在拟合模型: Robust Standard RL with Statistics...")
    results = []

    user_col = 'subject_id' if 'subject_id' in df.columns else 'user'

    for sub_id, sub_df in df.groupby(user_col):
        choices, rewards = prepare_arrays_robust(sub_df)

        if len(choices) < 5:
            continue

        x0 = np.array([0.3, np.log(1.0)])
        bounds = [(0.0, 1.0), (np.log(1e-3), np.log(1e3))]

        res = minimize(
            lambda x: negloglik_rl_robust(x, choices, rewards),
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )

        if res.success:
            # Calculate SE and p-values
            se, p_values = calculate_se_pvalues(
                res, negloglik_rl_robust, choices, rewards
            )

            # Transform parameters back
            alpha = res.x[0]
            beta = float(np.exp(res.x[1]))
            nll = res.fun

            # SE for transformed parameters
            # For beta: SE(beta) ≈ beta * SE(log_beta) (delta method)
            se_alpha = se[0] if not np.isnan(se[0]) else np.nan
            se_beta = beta * se[1] if not np.isnan(se[1]) else np.nan

            p_alpha = p_values[0]
            p_beta = p_values[1]

        else:
            alpha, beta, nll = np.nan, np.nan, np.nan
            se_alpha, se_beta = np.nan, np.nan
            p_alpha, p_beta = np.nan, np.nan

        bic = calculate_bic(nll, 2, len(choices)) if not np.isnan(nll) else np.nan

        results.append({
            'subject_id': sub_id,
            'alpha': alpha,
            'alpha_se': se_alpha,
            'alpha_p': p_alpha,
            'alpha_sig': format_significance(p_alpha),
            'beta': beta,
            'beta_se': se_beta,
            'beta_p': p_beta,
            'beta_sig': format_significance(p_beta),
            'nll': nll,
            'bic': bic,
            'n_trials': len(choices)
        })

    return pd.DataFrame(results)


# ==========================================
# Modified PT Model with SE/P-values
# ==========================================

def fit_robust_pt_model_with_stats(df):
    """
    Fit Static PT model with standard errors and p-values
    """
    print("正在拟合模型: Robust Static PT with Statistics...")
    results = []

    category_dists = compute_empirical_category_distributions(df)
    user_col = 'subject_id' if 'subject_id' in df.columns else 'user'

    for sub_id, sub_df in df.groupby(user_col):
        choices, _ = prepare_arrays_robust(sub_df)

        if len(choices) < 5:
            continue

        x0 = np.array([np.log(0.9), np.log(0.9), np.log(1.0)])
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
            # Calculate SE and p-values
            se, p_values = calculate_se_pvalues(
                res, negloglik_pt_robust, choices,
                category_dists=category_dists
            )

            # Transform parameters
            alpha = float(np.exp(res.x[0]))
            gamma = float(np.exp(res.x[1]))
            beta = float(np.exp(res.x[2]))
            nll = res.fun

            # SE for transformed parameters (delta method)
            se_alpha = alpha * se[0] if not np.isnan(se[0]) else np.nan
            se_gamma = gamma * se[1] if not np.isnan(se[1]) else np.nan
            se_beta = beta * se[2] if not np.isnan(se[2]) else np.nan

            p_alpha = p_values[0]
            p_gamma = p_values[1]
            p_beta = p_values[2]

        else:
            alpha, gamma, beta, nll = np.nan, np.nan, np.nan, np.nan
            se_alpha, se_gamma, se_beta = np.nan, np.nan, np.nan
            p_alpha, p_gamma, p_beta = np.nan, np.nan, np.nan

        bic = calculate_bic(nll, 3, len(choices)) if not np.isnan(nll) else np.nan

        results.append({
            'subject_id': sub_id,
            'alpha': alpha,
            'alpha_se': se_alpha,
            'alpha_p': p_alpha,
            'alpha_sig': format_significance(p_alpha),
            'gamma': gamma,
            'gamma_se': se_gamma,
            'gamma_p': p_gamma,
            'gamma_sig': format_significance(p_gamma),
            'beta': beta,
            'beta_se': se_beta,
            'beta_p': p_beta,
            'beta_sig': format_significance(p_beta),
            'nll': nll,
            'bic': bic,
            'n_trials': len(choices)
        })

    return pd.DataFrame(results)


# ==========================================
# Modified RL+PT Model with SE/P-values
# ==========================================

def fit_rl_pt_with_stats(df):
    """Fit RL + Prospect Theory with statistics"""
    print("正在拟合模型: RL + Prospect Theory with Statistics...")
    results = []
    GAMMA = 0.8
    REF_POINT = 2.5

    def get_nll_robust(params, choices, rewards):
        alpha, log_beta, log_lamb = params
        if not (0 <= alpha <= 1): return 1e9

        beta = np.exp(log_beta)
        lamb = np.exp(log_lamb)

        q_values = np.full(4, 2.5)
        nll = 0.0

        for c, r in zip(choices, rewards):
            c_idx = int(c) - 1
            probs = softmax(q_values, beta=beta)
            nll -= np.log(max(probs[c_idx], 1e-12))

            utility = (r - REF_POINT) ** GAMMA if r >= REF_POINT else -lamb * ((REF_POINT - r) ** GAMMA)
            q_values[c_idx] += alpha * (utility - q_values[c_idx])
        return nll

    for sub_id, sub_df in df.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue

        x0 = [0.5, np.log(2.0), np.log(1.0)]
        bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (np.log(1e-3), np.log(1e2))]

        res = minimize(lambda x: get_nll_robust(x, choices, rewards), x0, bounds=bounds, method='L-BFGS-B')

        if res.success:
            se, p_values = calculate_se_pvalues(res, get_nll_robust, choices, rewards)

            alpha = res.x[0]
            beta = np.exp(res.x[1])
            lamb = np.exp(res.x[2])

            se_alpha = se[0]
            se_beta = beta * se[1]
            se_lambda = lamb * se[2]

            results.append({
                'subject_id': sub_id,
                'alpha': alpha,
                'alpha_se': se_alpha,
                'alpha_p': p_values[0],
                'alpha_sig': format_significance(p_values[0]),
                'beta': beta,
                'beta_se': se_beta,
                'beta_p': p_values[1],
                'beta_sig': format_significance(p_values[1]),
                'lambda': lamb,
                'lambda_se': se_lambda,
                'lambda_p': p_values[2],
                'lambda_sig': format_significance(p_values[2]),
                'nll': res.fun,
                'bic': calculate_bic(res.fun, 3, len(choices)),
                'n_trials': len(choices)
            })
    return pd.DataFrame(results)


# ==========================================
# Modified Hybrid Model with SE/P-values
# ==========================================

def fit_hybrid_likeability_with_stats(df):
    """Fit Hybrid RL + Likeability with statistics"""
    print("正在拟合模型: Hybrid RL + Likeability with Statistics...")

    df_std = df.copy()
    like_cols = ['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']
    for sub_id, group in df_std.groupby('subject_id'):
        vals = group[like_cols].values.flatten()
        std_val = np.nanstd(vals)
        if std_val == 0: std_val = 1
        df_std.loc[df_std['subject_id'] == sub_id, like_cols] = (group[like_cols] - np.nanmean(vals)) / std_val

    results = []

    def get_nll_robust(params, choices, rewards, like_matrix):
        alpha, log_beta, omega = params
        if not (0 <= alpha <= 1): return 1e9

        beta = np.exp(log_beta)
        q_values = np.full(4, 2.5)
        nll = 0.0

        for i, (c, r) in enumerate(zip(choices, rewards)):
            c_idx = int(c) - 1
            likes = like_matrix[i]

            v_values = (beta * q_values) + (omega * likes)
            probs = softmax(v_values, beta=1.0)
            nll -= np.log(max(probs[c_idx], 1e-12))

            q_values[c_idx] += alpha * (r - q_values[c_idx])
        return nll

    for sub_id, sub_df in df_std.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue

        mask = ~sub_df['category_idx'].isna() & ~sub_df['reward'].isna()
        like_matrix = sub_df.loc[mask, like_cols].values

        x0 = [0.5, np.log(2.0), 0.5]
        bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (-10, 10)]

        res = minimize(lambda x: get_nll_robust(x, choices, rewards, like_matrix), x0, bounds=bounds, method='L-BFGS-B')

        if res.success:
            se, p_values = calculate_se_pvalues(res, get_nll_robust, choices, rewards, like_matrix=like_matrix)

            alpha = res.x[0]
            beta = np.exp(res.x[1])
            omega = res.x[2]

            se_alpha = se[0]
            se_beta = beta * se[1]
            se_omega = se[2]

            results.append({
                'subject_id': sub_id,
                'alpha': alpha,
                'alpha_se': se_alpha,
                'alpha_p': p_values[0],
                'alpha_sig': format_significance(p_values[0]),
                'beta': beta,
                'beta_se': se_beta,
                'beta_p': p_values[1],
                'beta_sig': format_significance(p_values[1]),
                'omega': omega,
                'omega_se': se_omega,
                'omega_p': p_values[2],
                'omega_sig': format_significance(p_values[2]),
                'nll': res.fun,
                'bic': calculate_bic(res.fun, 3, len(choices)),
                'n_trials': len(choices)
            })
    return pd.DataFrame(results)


# ==========================================
# Modified Perseveration Model with SE/P-values
# ==========================================

def fit_perseveration_rl_with_stats(df):
    """Fit RL with Perseveration with statistics"""
    print("正在拟合模型: RL with Perseveration with Statistics...")
    results = []

    def get_nll_robust(params, choices, rewards):
        alpha, log_beta, kappa = params
        if not (0 <= alpha <= 1): return 1e9

        beta = np.exp(log_beta)
        q_values = np.full(4, 2.5)
        nll = 0.0
        prev_choice = -1

        for c, r in zip(choices, rewards):
            c_idx = int(c) - 1

            v_values = beta * q_values
            if prev_choice != -1:
                v_values[prev_choice] += kappa

            probs = softmax(v_values, beta=1.0)
            nll -= np.log(max(probs[c_idx], 1e-12))

            q_values[c_idx] += alpha * (r - q_values[c_idx])
            prev_choice = c_idx
        return nll

    for sub_id, sub_df in df.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5: continue

        x0 = [0.5, np.log(2.0), 0.5]
        bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (-5, 5)]

        res = minimize(lambda x: get_nll_robust(x, choices, rewards), x0, bounds=bounds, method='L-BFGS-B')

        if res.success:
            se, p_values = calculate_se_pvalues(res, get_nll_robust, choices, rewards)

            alpha = res.x[0]
            beta = np.exp(res.x[1])
            kappa = res.x[2]

            se_alpha = se[0]
            se_beta = beta * se[1]
            se_kappa = se[2]

            results.append({
                'subject_id': sub_id,
                'alpha': alpha,
                'alpha_se': se_alpha,
                'alpha_p': p_values[0],
                'alpha_sig': format_significance(p_values[0]),
                'beta': beta,
                'beta_se': se_beta,
                'beta_p': p_values[1],
                'beta_sig': format_significance(p_values[1]),
                'kappa': kappa,
                'kappa_se': se_kappa,
                'kappa_p': p_values[2],
                'kappa_sig': format_significance(p_values[2]),
                'nll': res.fun,
                'bic': calculate_bic(res.fun, 3, len(choices)),
                'n_trials': len(choices)
            })
    return pd.DataFrame(results)


# ==========================================
# Main Execution with Statistics
# ==========================================

# Fit all models
df_standard_rl_stats = fit_robust_rl_model_with_stats(df_master)
df_pt_stats = fit_robust_pt_model_with_stats(df_master)
df_rl_pt_stats = fit_rl_pt_with_stats(df_master)
df_hybrid_stats = fit_hybrid_likeability_with_stats(df_master)
df_perseveration_stats = fit_perseveration_rl_with_stats(df_master)


# ==========================================
# Generate Summary Tables
# ==========================================

def create_summary_table(df, model_name, param_names):
    """Create a publication-ready summary table"""
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    # Calculate aggregate statistics
    summary_rows = []

    for param in param_names:
        mean_val = df[param].mean()
        se_col = f"{param}_se"

        # Aggregate SE using formula: SE_aggregate = sqrt(mean(SE^2))
        agg_se = np.sqrt((df[se_col] ** 2).mean())

        # Test if aggregate mean is different from 0
        z_score = mean_val / agg_se
        p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

        summary_rows.append({
            'Parameter': param,
            'Mean': mean_val,
            'SE': agg_se,
            'z': z_score,
            'P': p_value,
            'Sig': format_significance(p_value)
        })

    summary_df = pd.DataFrame(summary_rows)

    # Format for display
    print(summary_df.to_string(index=False))
    print(f"\nN subjects: {len(df)}")
    print(f"Mean BIC: {df['bic'].mean():.2f}")
    print(f"Mean NLL: {df['nll'].mean():.2f}")

    return summary_df


# Generate summary tables for each model
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

summary_rl = create_summary_table(df_standard_rl_stats, "Standard RL", ['alpha', 'beta'])
summary_pt = create_summary_table(df_pt_stats, "Static Prospect Theory", ['alpha', 'gamma', 'beta'])
summary_rl_pt = create_summary_table(df_rl_pt_stats, "RL + Prospect Theory", ['alpha', 'beta', 'lambda'])
summary_hybrid = create_summary_table(df_hybrid_stats, "Hybrid RL + Likeability", ['alpha', 'beta', 'omega'])
summary_perseveration = create_summary_table(df_perseveration_stats, "RL + Perseveration", ['alpha', 'beta', 'kappa'])

# ==========================================
# Export Results
# ==========================================

# Save detailed results
# df_standard_rl_stats.to_csv('model_results_rl_with_stats.csv', index=False)
# df_pt_stats.to_csv('model_results_pt_with_stats.csv', index=False)
# df_rl_pt_stats.to_csv('model_results_rl_pt_with_stats.csv', index=False)
# df_hybrid_stats.to_csv('model_results_hybrid_with_stats.csv', index=False)
# df_perseveration_stats.to_csv('model_results_perseveration_with_stats.csv', index=False)

# # Save summary tables
# with pd.ExcelWriter('model_summary_statistics.xlsx') as writer:
#     summary_rl.to_excel(writer, sheet_name='RL', index=False)
#     summary_pt.to_excel(writer, sheet_name='PT', index=False)
#     summary_rl_pt.to_excel(writer, sheet_name='RL_PT', index=False)
#     summary_hybrid.to_excel(writer, sheet_name='Hybrid', index=False)
#     summary_perseveration.to_excel(writer, sheet_name='Perseveration', index=False)

summary_rl.to_csv('statistics/summary_rl.csv', index=False)
summary_pt.to_csv('statistics/summary_pt.csv', index=False)
summary_rl_pt.to_csv('statistics/summary_rl_pt.csv', index=False)
summary_hybrid.to_csv('statistics/summary_hybrid.csv', index=False)
summary_perseveration.to_csv('statistics/summary_perseveration.csv', index=False)

print("\n✓ Results saved to CSV and Excel files")