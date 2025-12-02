#########################model check
"""
Comprehensive Model Validation Suite for Three Models
Includes: RL, PT, Hybrid Likeability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
import os

# ============================================================================
# SETUP & HELPER FUNCTIONS
# ============================================================================

# Create output directory
os.makedirs('validation_results', exist_ok=True)


def softmax(x, beta=1.0):
    """Softmax function with stability adjustment"""
    # 更稳定的softmax实现
    x = beta * (x - np.max(x))  # 减去最大值提高数值稳定性
    ex = np.exp(x)
    # 检查并处理NaN和无穷大
    ex = np.where(np.isfinite(ex), ex, 0.0)
    sum_ex = np.sum(ex)
    if sum_ex == 0:
        # 如果所有值都是负无穷或NaN，返回均匀分布
        return np.ones_like(x) / len(x)
    return ex / sum_ex


def calculate_bic(nll, n_params, n_observations):
    """Calculate Bayesian Information Criterion"""
    return 2 * nll + n_params * np.log(n_observations)


def prelec_weight(p, gamma):
    """Prelec probability weighting function"""
    p = np.clip(p, 1e-12, 1.0)
    return np.exp(-(-np.log(p)) ** gamma)


def prepare_arrays(sub_df):
    """Extract choices and rewards, ensuring 1-4 indexing"""
    choices = sub_df['category_idx'].values
    rewards = sub_df['reward'].values
    # Filter NaN
    mask = ~np.isnan(choices) & ~np.isnan(rewards)
    return choices[mask].astype(int), rewards[mask]


# Load data
print("Loading data...")
df_master = pd.read_csv('02_Master_df.csv')


# Compute global category distributions for PT model
def compute_empirical_category_distributions(df):
    dists = {}
    for cat in [1, 2, 3, 4]:
        vals = df[df['category_idx'] == cat]['reward'].dropna().values
        if len(vals) == 0:
            dists[cat] = (np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        else:
            unique, counts = np.unique(vals, return_counts=True)
            probs = counts / counts.sum()
            dists[cat] = (unique, probs)
    return dists


category_dists = compute_empirical_category_distributions(df_master)


# ============================================================================
# MODEL FITTING FUNCTIONS (from your code)
# ============================================================================

def negloglik_rl(params, choices, rewards, n_options=4, q0=2.5):
    """RL model negative log-likelihood"""
    alpha, logbeta = params
    if not (0 <= alpha <= 1):
        return 1e9
    beta = np.exp(logbeta)
    Q = np.ones(n_options) * q0
    nll = 0.0

    for c, r in zip(choices, rewards):
        c_idx = int(c) - 1
        probs = softmax(Q, beta=beta)
        p = max(probs[c_idx], 1e-12)
        nll -= np.log(p)
        Q[c_idx] = Q[c_idx] + alpha * (r - Q[c_idx])
    return nll


def fit_rl(choices, rewards):
    """Fit RL model"""
    x0 = [0.3, np.log(1.0)]
    bounds = [(0.0, 1.0), (np.log(1e-3), np.log(1e3))]
    res = minimize(lambda x: negloglik_rl(x, choices, rewards), x0, bounds=bounds, method='L-BFGS-B')

    if res.success:
        return {
            'alpha': res.x[0],
            'beta': float(np.exp(res.x[1])),
            'nll': res.fun
        }
    return {'alpha': np.nan, 'beta': np.nan, 'nll': np.nan}


def subjective_EV_for_category(cat, alpha_val, gamma_val):
    """Calculate subjective expected value for PT model"""
    outcomes, probs = category_dists[cat]
    u = outcomes ** alpha_val
    w = prelec_weight(probs, gamma_val)
    if w.sum() == 0:
        w = probs
    else:
        w = w / w.sum()
    return np.sum(w * u)


def negloglik_pt(params, choices, rewards):
    """PT model negative log-likelihood"""
    log_alpha, log_gamma, log_beta = params
    alpha_val = np.exp(log_alpha)
    gamma_val = np.exp(log_gamma)
    beta = np.exp(log_beta)

    sEV = np.array([subjective_EV_for_category(cat, alpha_val, gamma_val)
                    for cat in [1, 2, 3, 4]])
    nll = 0.0
    for c in choices:
        c_idx = int(c) - 1
        probs = softmax(sEV, beta=beta)
        p = max(probs[c_idx], 1e-12)
        nll -= np.log(p)
    return nll


def fit_pt(choices, rewards):
    """Fit PT model"""
    x0 = [np.log(0.9), np.log(0.9), np.log(1.0)]
    bounds = [(np.log(1e-2), np.log(5.0)), (np.log(1e-2), np.log(5.0)), (np.log(1e-3), np.log(1e3))]
    res = minimize(lambda x: negloglik_pt(x, choices, rewards), x0, bounds=bounds, method='L-BFGS-B')

    if res.success:
        return {
            'alpha': float(np.exp(res.x[0])),
            'gamma': float(np.exp(res.x[1])),
            'beta': float(np.exp(res.x[2])),
            'nll': res.fun
        }
    return {'alpha': np.nan, 'gamma': np.nan, 'beta': np.nan, 'nll': np.nan}


def negloglik_hybrid_like(params, choices, rewards, like_matrix):
    """Hybrid Likeability model negative log-likelihood"""
    alpha, log_beta, omega = params
    if not (0 <= alpha <= 1):
        return 1e9
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


def fit_hybrid_like(choices, rewards, like_matrix):
    """Fit Hybrid Likeability model"""
    x0 = [0.5, np.log(2.0), 0.5]
    bounds = [(0, 1), (np.log(1e-3), np.log(1e3)), (-10, 10)]
    res = minimize(lambda x: negloglik_hybrid_like(x, choices, rewards, like_matrix),
                   x0, bounds=bounds, method='L-BFGS-B')

    if res.success:
        return {
            'alpha': res.x[0],
            'beta': np.exp(res.x[1]),
            'omega': res.x[2],
            'nll': res.fun
        }
    return {'alpha': np.nan, 'beta': np.nan, 'omega': np.nan, 'nll': np.nan}


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def simulate_rl_data(alpha, beta, n_trials=40, n_options=4, q0=2.5):
    """Simulate data from RL model"""
    Q = np.ones(n_options) * q0
    choices = []
    rewards = []

    true_rewards = {1: (2.5, 1.13), 2: (2.5, 0.51), 3: (2.0, 0.0), 4: (3.0, 0.91)}

    for t in range(n_trials):
        probs = softmax(Q, beta=beta)
        choice = np.random.choice(n_options, p=probs) + 1
        mean_r, std_r = true_rewards[choice]
        reward = mean_r if std_r == 0 else np.clip(np.random.normal(mean_r, std_r), 1, 4)
        choices.append(choice)
        rewards.append(reward)
        Q[choice - 1] = Q[choice - 1] + alpha * (reward - Q[choice - 1])

    return np.array(choices), np.array(rewards)


def simulate_pt_data(alpha, gamma, beta, n_trials=40, n_options=4):
    """Simulate data from PT model"""
    sEV = np.zeros(n_options)
    for cat in range(1, n_options + 1):
        outcomes, probs = category_dists[cat]
        u = outcomes ** alpha
        w = prelec_weight(probs, gamma)
        w = w / w.sum()
        sEV[cat - 1] = np.sum(w * u)

    choices = []
    rewards = []
    true_rewards = {1: (2.5, 1.13), 2: (2.5, 0.51), 3: (2.0, 0.0), 4: (3.0, 0.91)}

    for t in range(n_trials):
        probs = softmax(sEV, beta=beta)
        choice = np.random.choice(n_options, p=probs) + 1
        mean_r, std_r = true_rewards[choice]
        reward = mean_r if std_r == 0 else np.clip(np.random.normal(mean_r, std_r), 1, 4)
        choices.append(choice)
        rewards.append(reward)

    return np.array(choices), np.array(rewards)


def simulate_hybrid_like_data(alpha, beta, omega, like_matrix, n_trials=40):
    """Simulate data from Hybrid Likeability model with improved numerical stability"""
    q_values = np.full(4, 2.5)
    choices = []
    rewards = []
    true_rewards = {1: (2.5, 1.13), 2: (2.5, 0.51), 3: (2.0, 0.0), 4: (3.0, 0.91)}

    for t in range(n_trials):
        # 确保like_matrix索引在范围内
        like_idx = t % len(like_matrix)
        likes = like_matrix[like_idx]

        # 计算价值函数，添加小常数避免数值问题
        v_values = (beta * q_values) + (omega * likes) + 1e-12

        # 使用改进的softmax
        probs = softmax(v_values, beta=1.0)

        # 确保概率有效
        if not np.all(np.isfinite(probs)) or np.any(probs < 0) or abs(np.sum(probs) - 1.0) > 1e-6:
            probs = np.ones(4) / 4  # 退回到均匀分布

        # 确保概率和为1
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / np.sum(probs)

        try:
            choice = np.random.choice(4, p=probs) + 1
        except ValueError as e:
            print(f"Probability error: {probs}, sum: {np.sum(probs)}")
            print(f"v_values: {v_values}, likes: {likes}")
            print(f"alpha: {alpha}, beta: {beta}, omega: {omega}")
            # 退回到均匀分布
            choice = np.random.choice(4) + 1

        mean_r, std_r = true_rewards[choice]
        reward = mean_r if std_r == 0 else np.clip(np.random.normal(mean_r, std_r), 1, 4)
        choices.append(choice)
        rewards.append(reward)
        q_values[choice - 1] += alpha * (reward - q_values[choice - 1])

    return np.array(choices), np.array(rewards)


# ============================================================================
# PART 1: PARAMETER RECOVERY
# ============================================================================

def parameter_recovery_all_models(n_simulations=50, n_trials=40):
    """Perform parameter recovery for three models"""
    print("\n" + "=" * 70)
    print("PARAMETER RECOVERY: THREE MODELS")
    print("=" * 70)

    np.random.seed(42)

    # --- RL Model ---
    print("\n[1/3] RL Model...")
    rl_results = []
    true_alphas_rl = np.random.uniform(0.05, 0.8, n_simulations)
    true_betas_rl = np.random.uniform(0.5, 5.0, n_simulations)

    for i in range(n_simulations):
        alpha_true = true_alphas_rl[i]
        beta_true = true_betas_rl[i]
        choices, rewards = simulate_rl_data(alpha_true, beta_true, n_trials)
        fit_res = fit_rl(choices, rewards)
        rl_results.append({
            'true_alpha': alpha_true, 'true_beta': beta_true,
            'recovered_alpha': fit_res['alpha'], 'recovered_beta': fit_res['beta']
        })

    rl_df = pd.DataFrame(rl_results)
    r_alpha_rl, _ = pearsonr(rl_df['true_alpha'], rl_df['recovered_alpha'])
    r_beta_rl, _ = pearsonr(rl_df['true_beta'], rl_df['recovered_beta'])
    print(f"  Alpha: r={r_alpha_rl:.3f}, Beta: r={r_beta_rl:.3f}")

    # --- PT Model ---
    print("\n[2/3] PT Model...")
    pt_results = []
    true_alphas_pt = np.random.uniform(0.5, 2.0, n_simulations)
    true_gammas_pt = np.random.uniform(0.3, 2.0, n_simulations)
    true_betas_pt = np.random.uniform(0.5, 5.0, n_simulations)

    for i in range(n_simulations):
        choices, rewards = simulate_pt_data(true_alphas_pt[i], true_gammas_pt[i],
                                            true_betas_pt[i], n_trials)
        fit_res = fit_pt(choices, rewards)
        pt_results.append({
            'true_alpha': true_alphas_pt[i], 'true_gamma': true_gammas_pt[i],
            'true_beta': true_betas_pt[i],
            'recovered_alpha': fit_res['alpha'], 'recovered_gamma': fit_res['gamma'],
            'recovered_beta': fit_res['beta']
        })

    pt_df = pd.DataFrame(pt_results)
    r_alpha_pt, _ = pearsonr(pt_df['true_alpha'], pt_df['recovered_alpha'])
    r_gamma_pt, _ = pearsonr(pt_df['true_gamma'], pt_df['recovered_gamma'])
    r_beta_pt, _ = pearsonr(pt_df['true_beta'], pt_df['recovered_beta'])
    print(f"  Alpha: r={r_alpha_pt:.3f}, Gamma: r={r_gamma_pt:.3f}, Beta: r={r_beta_pt:.3f}")

    # --- Hybrid Likeability Model ---
    print("\n[3/3] Hybrid Likeability Model...")
    like_results = []
    true_alphas_like = np.random.uniform(0.05, 0.8, n_simulations)
    true_betas_like = np.random.uniform(0.5, 5.0, n_simulations)
    true_omegas = np.random.uniform(-3, 3, n_simulations)

    # Use sample likeability data from real data
    sample_likes = df_master[df_master['subject_id'] == 1][['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']].values

    for i in range(n_simulations):
        choices, rewards = simulate_hybrid_like_data(
            true_alphas_like[i], true_betas_like[i], true_omegas[i], sample_likes, n_trials
        )
        fit_res = fit_hybrid_like(choices, rewards, sample_likes[:len(choices)])
        like_results.append({
            'true_alpha': true_alphas_like[i], 'true_beta': true_betas_like[i],
            'true_omega': true_omegas[i],
            'recovered_alpha': fit_res['alpha'], 'recovered_beta': fit_res['beta'],
            'recovered_omega': fit_res['omega']
        })

    like_df = pd.DataFrame(like_results)
    r_alpha_like, _ = pearsonr(like_df['true_alpha'], like_df['recovered_alpha'])
    r_beta_like, _ = pearsonr(like_df['true_beta'], like_df['recovered_beta'])
    r_omega, _ = pearsonr(like_df['true_omega'], like_df['recovered_omega'])
    print(f"  Alpha: r={r_alpha_like:.3f}, Beta: r={r_beta_like:.3f}, Omega: r={r_omega:.3f}")

    # 修正的子图布局 - 使用3x3网格
    fig = plt.figure(figsize=(18, 12))

    # RL Model (2个参数)
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(rl_df['true_alpha'], rl_df['recovered_alpha'], alpha=0.6)
    ax1.plot([0, 1], [0, 1], 'r--')
    ax1.set_title(f'RL: α (r={r_alpha_rl:.3f})')
    ax1.set_xlabel('True α');
    ax1.set_ylabel('Recovered α')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(rl_df['true_beta'], rl_df['recovered_beta'], alpha=0.6)
    max_b = max(rl_df['true_beta'].max(), rl_df['recovered_beta'].max())
    ax2.plot([0, max_b], [0, max_b], 'r--')
    ax2.set_title(f'RL: β (r={r_beta_rl:.3f})')
    ax2.set_xlabel('True β');
    ax2.set_ylabel('Recovered β')
    ax2.grid(True, alpha=0.3)

    # PT Model (3个参数)
    ax3 = plt.subplot(3, 3, 4)
    ax3.scatter(pt_df['true_alpha'], pt_df['recovered_alpha'], alpha=0.6)
    max_a = max(pt_df['true_alpha'].max(), pt_df['recovered_alpha'].max())
    ax3.plot([0, max_a], [0, max_a], 'r--')
    ax3.set_title(f'PT: α (r={r_alpha_pt:.3f})')
    ax3.set_xlabel('True α');
    ax3.set_ylabel('Recovered α')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 3, 5)
    ax4.scatter(pt_df['true_gamma'], pt_df['recovered_gamma'], alpha=0.6)
    max_g = max(pt_df['true_gamma'].max(), pt_df['recovered_gamma'].max())
    ax4.plot([0, max_g], [0, max_g], 'r--')
    ax4.set_title(f'PT: γ (r={r_gamma_pt:.3f})')
    ax4.set_xlabel('True γ');
    ax4.set_ylabel('Recovered γ')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 6)
    ax5.scatter(pt_df['true_beta'], pt_df['recovered_beta'], alpha=0.6)
    max_b = max(pt_df['true_beta'].max(), pt_df['recovered_beta'].max())
    ax5.plot([0, max_b], [0, max_b], 'r--')
    ax5.set_title(f'PT: β (r={r_beta_pt:.3f})')
    ax5.set_xlabel('True β');
    ax5.set_ylabel('Recovered β')
    ax5.grid(True, alpha=0.3)

    # Hybrid Likeability (3个参数)
    ax6 = plt.subplot(3, 3, 7)
    ax6.scatter(like_df['true_alpha'], like_df['recovered_alpha'], alpha=0.6)
    ax6.plot([0, 1], [0, 1], 'r--')
    ax6.set_title(f'Hybrid-L: α (r={r_alpha_like:.3f})')
    ax6.set_xlabel('True α');
    ax6.set_ylabel('Recovered α')
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 8)
    ax7.scatter(like_df['true_beta'], like_df['recovered_beta'], alpha=0.6)
    max_b = max(like_df['true_beta'].max(), like_df['recovered_beta'].max())
    ax7.plot([0, max_b], [0, max_b], 'r--')
    ax7.set_title(f'Hybrid-L: β (r={r_beta_like:.3f})')
    ax7.set_xlabel('True β');
    ax7.set_ylabel('Recovered β')
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 9)
    ax8.scatter(like_df['true_omega'], like_df['recovered_omega'], alpha=0.6)
    min_o = min(like_df['true_omega'].min(), like_df['recovered_omega'].min())
    max_o = max(like_df['true_omega'].max(), like_df['recovered_omega'].max())
    ax8.plot([min_o, max_o], [min_o, max_o], 'r--')
    ax8.set_title(f'Hybrid-L: ω (r={r_omega:.3f})')
    ax8.set_xlabel('True ω');
    ax8.set_ylabel('Recovered ω')
    ax8.grid(True, alpha=0.3)

    # 添加总标题
    plt.suptitle('Parameter Recovery for Three Models', fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.savefig('validation_results/parameter_recovery_three_models.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: validation_results/parameter_recovery_three_models.png")

    # Save results
    rl_df.to_csv('validation_results/param_recovery_rl.csv', index=False)
    pt_df.to_csv('validation_results/param_recovery_pt.csv', index=False)
    like_df.to_csv('validation_results/param_recovery_hybrid_like.csv', index=False)

    return rl_df, pt_df, like_df


# ============================================================================
# PART 2: MODEL RECOVERY
# ============================================================================

def model_recovery_all(n_simulations=20, n_trials=40):
    """Model recovery for three models"""
    print("\n" + "=" * 70)
    print("MODEL RECOVERY: THREE MODELS")
    print("=" * 70)

    np.random.seed(44)
    results = []
    sample_likes = df_master[df_master['subject_id'] == 1][['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']].values

    for sim in range(n_simulations):
        if (sim + 1) % 5 == 0:
            print(f"Simulation {sim + 1}/{n_simulations}")

        # --- Generate data from RL model ---
        alpha_rl = np.random.uniform(0.1, 0.6)
        beta_rl = np.random.uniform(0.5, 4.0)
        choices_rl, rewards_rl = simulate_rl_data(alpha_rl, beta_rl, n_trials)

        # Fit all three models to RL-generated data
        rl_fit = fit_rl(choices_rl, rewards_rl)
        pt_fit = fit_pt(choices_rl, rewards_rl)
        like_fit = fit_hybrid_like(choices_rl, rewards_rl, sample_likes[:len(choices_rl)])

        rl_bic = calculate_bic(rl_fit['nll'], 2, n_trials)
        pt_bic = calculate_bic(pt_fit['nll'], 3, n_trials)
        like_bic = calculate_bic(like_fit['nll'], 3, n_trials)

        winner = min([('RL', rl_bic), ('PT', pt_bic), ('Hybrid_Like', like_bic)],
                     key=lambda x: x[1])[0]

        results.append({
            'true_model': 'RL',
            'rl_bic': rl_bic, 'pt_bic': pt_bic, 'hybrid_like_bic': like_bic,
            'winner': winner
        })

        # --- Generate data from PT model ---
        alpha_pt = np.random.uniform(0.7, 1.5)
        gamma_pt = np.random.uniform(0.5, 1.5)
        beta_pt = np.random.uniform(0.5, 4.0)
        choices_pt, rewards_pt = simulate_pt_data(alpha_pt, gamma_pt, beta_pt, n_trials)

        rl_fit = fit_rl(choices_pt, rewards_pt)
        pt_fit = fit_pt(choices_pt, rewards_pt)
        like_fit = fit_hybrid_like(choices_pt, rewards_pt, sample_likes[:len(choices_pt)])

        rl_bic = calculate_bic(rl_fit['nll'], 2, n_trials)
        pt_bic = calculate_bic(pt_fit['nll'], 3, n_trials)
        like_bic = calculate_bic(like_fit['nll'], 3, n_trials)

        winner = min([('RL', rl_bic), ('PT', pt_bic), ('Hybrid_Like', like_bic)],
                     key=lambda x: x[1])[0]

        results.append({
            'true_model': 'PT',
            'rl_bic': rl_bic, 'pt_bic': pt_bic, 'hybrid_like_bic': like_bic,
            'winner': winner
        })

        # --- Generate data from Hybrid Likeability model ---
        alpha_like = np.random.uniform(0.1, 0.6)
        beta_like = np.random.uniform(0.5, 4.0)
        omega_like = np.random.uniform(-2, 2)
        choices_like, rewards_like = simulate_hybrid_like_data(alpha_like, beta_like, omega_like, sample_likes,
                                                               n_trials)

        rl_fit = fit_rl(choices_like, rewards_like)
        pt_fit = fit_pt(choices_like, rewards_like)
        like_fit = fit_hybrid_like(choices_like, rewards_like, sample_likes[:len(choices_like)])

        rl_bic = calculate_bic(rl_fit['nll'], 2, n_trials)
        pt_bic = calculate_bic(pt_fit['nll'], 3, n_trials)
        like_bic = calculate_bic(like_fit['nll'], 3, n_trials)

        winner = min([('RL', rl_bic), ('PT', pt_bic), ('Hybrid_Like', like_bic)],
                     key=lambda x: x[1])[0]

        results.append({
            'true_model': 'Hybrid_Like',
            'rl_bic': rl_bic, 'pt_bic': pt_bic, 'hybrid_like_bic': like_bic,
            'winner': winner
        })

    results_df = pd.DataFrame(results)

    # Create confusion matrix
    confusion = pd.crosstab(
        results_df['true_model'],
        results_df['winner'],
        normalize='index'
    ) * 100

    print("\nConfusion Matrix (% correct):")
    print(confusion)

    # Calculate overall accuracy
    accuracy = (results_df['true_model'] == results_df['winner']).mean()
    print(f"\nOverall Model Recovery Accuracy: {accuracy * 100:.1f}%")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='.1f', cmap='Blues',
                cbar_kws={'label': '% Recovery'}, vmin=0, vmax=100)
    plt.title(f'Model Recovery Confusion Matrix (3 Models)\n(Overall Accuracy: {accuracy * 100:.1f}%)', fontsize=14)
    plt.xlabel('Recovered Model', fontsize=12)
    plt.ylabel('True Model', fontsize=12)
    plt.tight_layout()
    plt.savefig('validation_results/model_recovery_confusion_3models.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved: validation_results/model_recovery_confusion_3models.png")

    results_df.to_csv('validation_results/model_recovery_results_3models.csv', index=False)

    return results_df, confusion


# ============================================================================
# PART 3: POSTERIOR PREDICTIVE CHECKS
# ============================================================================

def posterior_predictive_checks_all(df, n_simulations=100):
    """Perform posterior predictive checks for three models"""
    print("\n" + "=" * 70)
    print("POSTERIOR PREDICTIVE CHECKS: THREE MODELS")
    print("=" * 70)

    # Get unique subjects with valid preference data
    valid_subjects = df[df['valid_preference_data'] == True]['subject_id'].unique()

    observed_stats = []
    rl_sim_stats = []
    pt_sim_stats = []
    like_sim_stats = []

    for subj_id in valid_subjects:
        print(f"\nProcessing subject {subj_id}...")

        subj_df = df[df['subject_id'] == subj_id]
        choices_obs, rewards_obs = prepare_arrays(subj_df)

        if len(choices_obs) < 10:
            continue

        # Get likeability data
        like_cols = ['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']
        like_matrix = subj_df[like_cols].values
        like_matrix = like_matrix[~np.isnan(choices_obs) & ~np.isnan(rewards_obs)]

        # Fit all models to observed data
        rl_fit = fit_rl(choices_obs, rewards_obs)
        pt_fit = fit_pt(choices_obs, rewards_obs)
        like_fit = fit_hybrid_like(choices_obs, rewards_obs, like_matrix)

        # Observed statistics
        obs_cat_freq = np.array([np.mean(choices_obs == i) for i in [1, 2, 3, 4]])
        obs_mean_reward = np.mean(rewards_obs)
        obs_std_reward = np.std(rewards_obs)
        obs_switches = np.sum(np.diff(choices_obs) != 0) / (len(choices_obs) - 1)

        observed_stats.append({
            'subject_id': subj_id,
            'cat1_freq': obs_cat_freq[0], 'cat2_freq': obs_cat_freq[1],
            'cat3_freq': obs_cat_freq[2], 'cat4_freq': obs_cat_freq[3],
            'mean_reward': obs_mean_reward,
            'std_reward': obs_std_reward,
            'switch_rate': obs_switches
        })

        # Simulate from RL model
        rl_stats_list = {'cat_freq': [], 'mean_reward': [], 'std_reward': [], 'switch_rate': []}
        for sim in range(n_simulations):
            c_sim, r_sim = simulate_rl_data(rl_fit['alpha'], rl_fit['beta'], len(choices_obs))
            rl_stats_list['cat_freq'].append([np.mean(c_sim == i) for i in [1, 2, 3, 4]])
            rl_stats_list['mean_reward'].append(np.mean(r_sim))
            rl_stats_list['std_reward'].append(np.std(r_sim))
            rl_stats_list['switch_rate'].append(np.sum(np.diff(c_sim) != 0) / (len(c_sim) - 1))

        rl_cat_freq = np.mean(rl_stats_list['cat_freq'], axis=0)
        rl_sim_stats.append({
            'subject_id': subj_id,
            'cat1_freq': rl_cat_freq[0], 'cat2_freq': rl_cat_freq[1],
            'cat3_freq': rl_cat_freq[2], 'cat4_freq': rl_cat_freq[3],
            'mean_reward': np.mean(rl_stats_list['mean_reward']),
            'std_reward': np.mean(rl_stats_list['std_reward']),
            'switch_rate': np.mean(rl_stats_list['switch_rate'])
        })

        # Simulate from PT model
        pt_stats_list = {'cat_freq': [], 'mean_reward': [], 'std_reward': [], 'switch_rate': []}
        for sim in range(n_simulations):
            c_sim, r_sim = simulate_pt_data(pt_fit['alpha'], pt_fit['gamma'], pt_fit['beta'], len(choices_obs))
            pt_stats_list['cat_freq'].append([np.mean(c_sim == i) for i in [1, 2, 3, 4]])
            pt_stats_list['mean_reward'].append(np.mean(r_sim))
            pt_stats_list['std_reward'].append(np.std(r_sim))
            pt_stats_list['switch_rate'].append(np.sum(np.diff(c_sim) != 0) / (len(c_sim) - 1))

        pt_cat_freq = np.mean(pt_stats_list['cat_freq'], axis=0)
        pt_sim_stats.append({
            'subject_id': subj_id,
            'cat1_freq': pt_cat_freq[0], 'cat2_freq': pt_cat_freq[1],
            'cat3_freq': pt_cat_freq[2], 'cat4_freq': pt_cat_freq[3],
            'mean_reward': np.mean(pt_stats_list['mean_reward']),
            'std_reward': np.mean(pt_stats_list['std_reward']),
            'switch_rate': np.mean(pt_stats_list['switch_rate'])
        })

        # Simulate from Hybrid Likeability model
        like_stats_list = {'cat_freq': [], 'mean_reward': [], 'std_reward': [], 'switch_rate': []}
        for sim in range(n_simulations):
            c_sim, r_sim = simulate_hybrid_like_data(like_fit['alpha'], like_fit['beta'],
                                                     like_fit['omega'], like_matrix, len(choices_obs))
            like_stats_list['cat_freq'].append([np.mean(c_sim == i) for i in [1, 2, 3, 4]])
            like_stats_list['mean_reward'].append(np.mean(r_sim))
            like_stats_list['std_reward'].append(np.std(r_sim))
            like_stats_list['switch_rate'].append(np.sum(np.diff(c_sim) != 0) / (len(c_sim) - 1))

        like_cat_freq = np.mean(like_stats_list['cat_freq'], axis=0)
        like_sim_stats.append({
            'subject_id': subj_id,
            'cat1_freq': like_cat_freq[0], 'cat2_freq': like_cat_freq[1],
            'cat3_freq': like_cat_freq[2], 'cat4_freq': like_cat_freq[3],
            'mean_reward': np.mean(like_stats_list['mean_reward']),
            'std_reward': np.mean(like_stats_list['std_reward']),
            'switch_rate': np.mean(like_stats_list['switch_rate'])
        })

    # Convert to DataFrames
    obs_df = pd.DataFrame(observed_stats)
    rl_df = pd.DataFrame(rl_sim_stats)
    pt_df = pd.DataFrame(pt_sim_stats)
    like_df = pd.DataFrame(like_sim_stats)

    # Plot comparisons
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    stats_to_plot = [
        ('cat1_freq', 'Category 1 Frequency'),
        ('cat4_freq', 'Category 4 Frequency'),
        ('mean_reward', 'Mean Reward'),
        ('std_reward', 'Reward Std Dev'),
        ('switch_rate', 'Switch Rate')
    ]

    for idx, (stat, label) in enumerate(stats_to_plot):
        if idx >= 6:
            break
        ax = axes[idx // 3, idx % 3]

        ax.scatter(obs_df[stat], rl_df[stat], alpha=0.5, label='RL', s=30, color='blue')
        ax.scatter(obs_df[stat], pt_df[stat], alpha=0.5, label='PT', s=30, color='red')
        ax.scatter(obs_df[stat], like_df[stat], alpha=0.5, label='Hybrid-L', s=30, color='green')

        min_val = min(obs_df[stat].min(), rl_df[stat].min(), pt_df[stat].min(),
                      like_df[stat].min())
        max_val = max(obs_df[stat].max(), rl_df[stat].max(), pt_df[stat].max(),
                      like_df[stat].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect match')

        ax.set_xlabel(f'Observed {label}', fontsize=10)
        ax.set_ylabel(f'Simulated {label}', fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    if len(stats_to_plot) < 6:
        fig.delaxes(axes[1, 2])

    plt.suptitle('Posterior Predictive Check: Three Models', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('validation_results/ppc_three_models.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nSaved: validation_results/ppc_three_models.png")

    # Calculate correlations
    print("\n" + "=" * 70)
    print("Correlation between Observed and Simulated Statistics")
    print("=" * 70)

    for stat in ['cat1_freq', 'cat4_freq', 'mean_reward', 'std_reward', 'switch_rate']:
        rl_corr, _ = pearsonr(obs_df[stat], rl_df[stat])
        pt_corr, _ = pearsonr(obs_df[stat], pt_df[stat])
        like_corr, _ = pearsonr(obs_df[stat], like_df[stat])

        print(f"{stat:15s}: RL r={rl_corr:.3f}, PT r={pt_corr:.3f}, Hybrid-L r={like_corr:.3f}")

    # Save results
    obs_df.to_csv('validation_results/ppc_observed_stats.csv', index=False)
    rl_df.to_csv('validation_results/ppc_rl_simulated.csv', index=False)
    pt_df.to_csv('validation_results/ppc_pt_simulated.csv', index=False)
    like_df.to_csv('validation_results/ppc_hybrid_like_simulated.csv', index=False)

    return obs_df, rl_df, pt_df, like_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODEL VALIDATION SUITE FOR THREE MODELS")
    print("=" * 70)

    # 1. Parameter Recovery
    print("\n[1/3] Parameter Recovery...")
    rl_pr, pt_pr, like_pr = parameter_recovery_all_models(n_simulations=50, n_trials=40)

    # 2. Model Recovery
    print("\n[2/3] Model Recovery...")
    mr_results, mr_confusion = model_recovery_all(n_simulations=20, n_trials=40)

    # 3. Posterior Predictive Checks
    print("\n[3/3] Posterior Predictive Checks...")
    obs_stats, rl_stats, pt_stats, like_stats = posterior_predictive_checks_all(df_master, n_simulations=100)

    # Generate summary report
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY REPORT")
    print("=" * 70)

    print("\n1. Parameter Recovery:")
    print(f"   RL Model - Alpha: r={pearsonr(rl_pr['true_alpha'], rl_pr['recovered_alpha'])[0]:.3f}, " +
          f"Beta: r={pearsonr(rl_pr['true_beta'], rl_pr['recovered_beta'])[0]:.3f}")
    print(f"   PT Model - Alpha: r={pearsonr(pt_pr['true_alpha'], pt_pr['recovered_alpha'])[0]:.3f}, " +
          f"Gamma: r={pearsonr(pt_pr['true_gamma'], pt_pr['recovered_gamma'])[0]:.3f}, " +
          f"Beta: r={pearsonr(pt_pr['true_beta'], pt_pr['recovered_beta'])[0]:.3f}")
    print(f"   Hybrid-L Model - Alpha: r={pearsonr(like_pr['true_alpha'], like_pr['recovered_alpha'])[0]:.3f}, " +
          f"Beta: r={pearsonr(like_pr['true_beta'], like_pr['recovered_beta'])[0]:.3f}, " +
          f"Omega: r={pearsonr(like_pr['true_omega'], like_pr['recovered_omega'])[0]:.3f}")

    print("\n2. Model Recovery:")
    print(f"   Overall Accuracy: {(mr_results['true_model'] == mr_results['winner']).mean() * 100:.1f}%")
    for model in ['RL', 'PT', 'Hybrid_Like']:
        recovery = (mr_results[mr_results['true_model'] == model]['winner'] == model).mean() * 100
        print(f"   {model} Recovery: {recovery:.1f}%")

    print("\n3. Posterior Predictive Checks:")
    print(f"   Participants analyzed: {len(obs_stats)}")
    print(f"   Category 4 frequency - RL: r={pearsonr(obs_stats['cat4_freq'], rl_stats['cat4_freq'])[0]:.3f}, " +
          f"PT: r={pearsonr(obs_stats['cat4_freq'], pt_stats['cat4_freq'])[0]:.3f}, " +
          f"Hybrid-L: r={pearsonr(obs_stats['cat4_freq'], like_stats['cat4_freq'])[0]:.3f}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE!")
    print("=" * 70)
    print("\nAll validation results saved in validation_results/ directory")