############################################################Calculate Random BIC Values for Three Models by Simulation


import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ==========================================
# HELPER FUNCTIONS (from your original code)
# ==========================================

def softmax(x, beta=1.0):
    """Softmax function with stability adjustment."""
    ex = np.exp(beta * (x - np.max(x)))
    return ex / ex.sum()


def calculate_bic(nll, n_params, n_observations):
    """Calculate Bayesian Information Criterion (BIC)"""
    return 2 * nll + n_params * np.log(n_observations)


def prepare_arrays_robust(sub_df):
    """Data preprocessing: ensure no NaN values, extract choice and reward."""
    choice_col = 'category_idx' if 'category_idx' in sub_df.columns else 'choice'
    reward_col = 'reward'

    mask = ~sub_df[choice_col].isna() & ~sub_df[reward_col].isna()
    clean_df = sub_df[mask].copy()

    choices = clean_df[choice_col].astype(int).values
    rewards = clean_df[reward_col].astype(float).values

    if choices.min() == 0:
        choices = choices + 1

    return choices, rewards


# ==========================================
# RANDOM CHOICE SIMULATION
# ==========================================

def simulate_random_choices(df_original, seed=42):
    """
    Simulate a single random responder with 40 trials (Q21-Q60).
    Creates random choices for each question, ignoring subject structure.

    Parameters:
    -----------
    df_original : DataFrame
        Original experimental data (used only to get reward structure)
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    df_random : DataFrame
        Simulated data with 40 random choices (one virtual subject)
    """
    np.random.seed(seed)

    # Get unique question information from original data
    # Assuming questions 21-60 (40 questions total)
    questions = range(21, 61)

    # Create a mapping of question_id to category rewards
    # Using the first occurrence of each question to get reward structure
    question_rewards = {}
    for qid in questions:
        q_data = df_original[df_original['question_id'] == qid]
        if len(q_data) > 0:
            # Get rewards for each category (1-4) for this question
            rewards_by_cat = {}
            for cat in [1, 2, 3, 4]:
                cat_data = q_data[q_data['category_idx'] == cat]
                if len(cat_data) > 0:
                    rewards_by_cat[cat] = cat_data['reward'].iloc[0]
            question_rewards[qid] = rewards_by_cat

    # Generate 40 random choices (one per question)
    random_choices = np.random.randint(1, 5, size=40)

    # Create dataframe for single random subject
    data = []
    for i, qid in enumerate(questions):
        choice = random_choices[i]
        # Get reward for chosen category
        reward = question_rewards.get(qid, {}).get(choice, 2.5)  # default 2.5 if missing

        # Get likeability data from original (use first subject's data as placeholder)
        sample_row = df_original[df_original['question_id'] == qid].iloc[0]

        data.append({
            'subject_id': 999,  # Virtual random subject ID
            'trial': i + 1,
            'question_id': qid,
            'category_idx': choice,
            'reward': reward,
            'valid_preference_data': True,
            'prev_reward': 0 if i == 0 else data[i - 1]['reward'],
            'prev_choice': 0 if i == 0 else data[i - 1]['category_idx'],
            'like_cat1': sample_row['like_cat1'],
            'like_cat2': sample_row['like_cat2'],
            'like_cat3': sample_row['like_cat3'],
            'like_cat4': sample_row['like_cat4'],
            'score_cat1': sample_row['score_cat1'],
            'score_cat2': sample_row['score_cat2'],
            'score_cat3': sample_row['score_cat3'],
            'score_cat4': sample_row['score_cat4']
        })

    df_random = pd.DataFrame(data)

    print(f"✓ Simulated 40 random choices (Q21-Q60) for 1 virtual subject")
    print(f"✓ Choice distribution: {np.bincount(random_choices)[1:]}")

    return df_random


# ==========================================
# MODEL FITTING FUNCTIONS (unchanged structure)
# ==========================================

# 1. Standard RL Model
def negloglik_rl_robust(params, choices, rewards, n_options=4, q0=2.5):
    alpha, logbeta = params
    if not (0 <= alpha <= 1):
        return 1e9

    beta = np.exp(logbeta)
    Q = np.ones(n_options) * q0
    nll = 0.0

    for c, r in zip(choices, rewards):
        c_idx = c - 1
        probs = softmax(Q, beta=beta)
        p = probs[c_idx]
        p = max(p, 1e-12)
        nll -= np.log(p)
        Q[c_idx] = Q[c_idx] + alpha * (r - Q[c_idx])

    return nll


def fit_robust_rl_model(df):
    print("Fitting: Standard RL Model (Random Data)...")
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
            x0, bounds=bounds, method='L-BFGS-B'
        )

        if res.success:
            alpha = res.x[0]
            beta = float(np.exp(res.x[1]))
            nll = res.fun
        else:
            alpha, beta, nll = np.nan, np.nan, np.nan

        bic = calculate_bic(nll, 2, len(choices)) if not np.isnan(nll) else np.nan

        results.append({
            'subject_id': sub_id,
            'alpha': alpha,
            'beta': beta,
            'nll': nll,
            'bic': bic
        })

    return pd.DataFrame(results)


# 2. RL + Prospect Theory
def fit_rl_pt(df):
    print("Fitting: RL + Prospect Theory (Random Data)...")
    results = []
    GAMMA = 0.8
    REF_POINT = 2.5

    def get_nll_robust(params, choices, rewards):
        alpha, log_beta, log_lamb = params
        if not (0 <= alpha <= 1):
            return 1e9

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
        if len(choices) < 5:
            continue

        x0 = [0.5, np.log(2.0), np.log(1.0)]
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


# 3. Static PT Model
def compute_empirical_category_distributions(df):
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
    p = np.clip(p, 1e-12, 1.0)
    return np.exp(-(-np.log(p)) ** gamma)


def subjective_EV_for_category(cat, alpha_val, gamma_val, category_dists):
    outcomes, probs = category_dists[cat]
    u = outcomes ** alpha_val
    w = prelec_weight(probs, gamma_val)
    if w.sum() == 0:
        w = probs
    else:
        w = w / w.sum()
    return np.sum(w * u)


def negloglik_pt_robust(params, choices, category_dists):
    log_alpha, log_gamma, log_beta = params
    alpha_val = np.exp(log_alpha)
    gamma_val = np.exp(log_gamma)
    beta = np.exp(log_beta)

    sEV = np.array([subjective_EV_for_category(cat, alpha_val, gamma_val, category_dists)
                    for cat in [1, 2, 3, 4]])

    nll = 0.0
    for c in choices:
        c_idx = c - 1
        probs = softmax(sEV, beta=beta)
        p = probs[c_idx]
        p = max(p, 1e-12)
        nll -= np.log(p)
    return nll


def fit_robust_pt_model(df):
    print("Fitting: Static PT Model (Random Data)...")
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
            x0, bounds=bounds, method='L-BFGS-B'
        )

        if res.success:
            alpha = float(np.exp(res.x[0]))
            gamma = float(np.exp(res.x[1]))
            beta = float(np.exp(res.x[2]))
            nll = res.fun
        else:
            alpha, gamma, beta, nll = np.nan, np.nan, np.nan, np.nan

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


# 4. Hybrid RL + Likeability
def fit_hybrid_likeability(df):
    print("Fitting: Hybrid RL + Likeability (Random Data)...")

    df_std = df.copy()
    like_cols = ['like_cat1', 'like_cat2', 'like_cat3', 'like_cat4']
    for sub_id, group in df_std.groupby('subject_id'):
        vals = group[like_cols].values.flatten()
        std_val = np.nanstd(vals)
        if std_val == 0:
            std_val = 1
        df_std.loc[df_std['subject_id'] == sub_id, like_cols] = (group[like_cols] - np.nanmean(vals)) / std_val

    results = []

    def get_nll_robust(params, choices, rewards, like_matrix):
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

    for sub_id, sub_df in df_std.groupby('subject_id'):
        choices, rewards = prepare_arrays_robust(sub_df)
        if len(choices) < 5:
            continue

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


# 5. RL with Perseveration
def fit_perseveration_rl(df):
    print("Fitting: RL with Perseveration (Random Data)...")
    results = []

    def get_nll_robust(params, choices, rewards):
        alpha, log_beta, kappa = params
        if not (0 <= alpha <= 1):
            return 1e9

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
        if len(choices) < 5:
            continue

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


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    # Load your original data
    df_master = pd.read_csv('02_Master_df.txt')

    print("=" * 60)
    print("RANDOM CHOICE SIMULATION & MODEL FITTING")
    print("=" * 60)
    print(f"Loaded {len(df_master)} rows, {df_master['subject_id'].nunique()} subjects\n")

    # Step 1: Simulate random choices (40 trials, Q21-Q60)
    df_random = simulate_random_choices(df_master, seed=42)

    # Step 2: Fit all models to random data
    print("\n" + "=" * 60)
    print("FITTING MODELS TO RANDOM DATA")
    print("=" * 60 + "\n")

    df_random_rl = fit_robust_rl_model(df_random)
    df_random_rl_pt = fit_rl_pt(df_random)
    df_random_pt = fit_robust_pt_model(df_random)
    df_random_hybrid = fit_hybrid_likeability(df_random)
    df_random_perseveration = fit_perseveration_rl(df_random)

    # Step 3: Save results to separate files
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    df_random_rl.to_csv('random_bic_standard_rl.csv', index=False)
    print("✓ Saved: random_bic_standard_rl.csv")

    df_random_rl_pt.to_csv('random_bic_rl_pt.csv', index=False)
    print("✓ Saved: random_bic_rl_pt.csv")

    df_random_pt.to_csv('random_bic_static_pt.csv', index=False)
    print("✓ Saved: random_bic_static_pt.csv")

    df_random_hybrid.to_csv('random_bic_hybrid_likeability.csv', index=False)
    print("✓ Saved: random_bic_hybrid_likeability.csv")

    df_random_perseveration.to_csv('random_bic_perseveration_rl.csv', index=False)
    print("✓ Saved: random_bic_perseveration_rl.csv")

    # Step 4: Display summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: Random BIC Values")
    print("=" * 60)

    print(f"\nStandard RL:")
    print(f"  Mean BIC: {df_random_rl['bic'].mean():.2f} ± {df_random_rl['bic'].std():.2f}")

    print(f"\nRL + PT:")
    print(f"  Mean BIC: {df_random_rl_pt['bic'].mean():.2f} ± {df_random_rl_pt['bic'].std():.2f}")

    print(f"\nStatic PT:")
    print(f"  Mean BIC: {df_random_pt['bic'].mean():.2f} ± {df_random_pt['bic'].std():.2f}")

    print(f"\nHybrid RL + Likeability:")
    print(f"  Mean BIC: {df_random_hybrid['bic'].mean():.2f} ± {df_random_hybrid['bic'].std():.2f}")

    print(f"\nRL + Perseveration:")
    print(f"  Mean BIC: {df_random_perseveration['bic'].mean():.2f} ± {df_random_perseveration['bic'].std():.2f}")

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)