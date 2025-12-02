import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data():
    raw_df = pd.read_csv('D:/Development/Data/BCAI/W8/data_2025_cleaned.csv')
    
    qstat = pd.read_csv('D:/Development/Data/BCAI/PCA/01_Qstat.csv')
    
    item_likes = pd.read_csv('D:/Development/Data/BCAI/W8/01_item_like.csv')


    df = raw_df.dropna(subset=['answer', 'score']).copy()
    
    cat_map = {'Colours': 1, 'Abstract concepts': 2, 'Subjects': 3, 'Places': 4}
    df['category_idx'] = df['category'].map(cat_map)
    
    df = df.rename(columns={'user_id': 'subject_id', 'score': 'reward'})

    trials = df[(df['question_id'] >= 21) & (df['question_id'] <= 60)].copy()
    trials = trials.sort_values(['subject_id', 'question_id'])
    trials['trial'] = trials.groupby('subject_id').cumcount() + 1

    q_items = qstat.pivot(index='question_id', columns='category_id', values='item_id')
    q_items.columns = [f'item_id_cat{c}' for c in q_items.columns]

    q_scores = qstat.pivot(index='question_id', columns='category_id', values='available_score')
    q_scores.columns = [f'score_cat{c}' for c in q_scores.columns]

    trials = trials.merge(q_items, on='question_id', how='left')
    trials = trials.merge(q_scores, on='question_id', how='left')

    item_likes['like'] = pd.to_numeric(item_likes['like'], errors='coerce')
    user_item_map = item_likes.set_index(['user_id', 'item_id'])['like'].to_dict()

    def get_ratings(row):
        uid = row['subject_id']
        ratings = {}
        for c in range(1, 5):
            iid = row[f'item_id_cat{c}']
            r = user_item_map.get((uid, iid), np.nan)
            ratings[f'like_cat{c}'] = r
        return pd.Series(ratings)

    rating_cols = trials.apply(get_ratings, axis=1)
    trials = pd.concat([trials, rating_cols], axis=1)

    trials['prev_reward'] = trials.groupby('subject_id')['reward'].shift(1)
    trials['prev_choice'] = trials.groupby('subject_id')['category_idx'].shift(1)
    
    trials['prev_reward'] = trials['prev_reward'].fillna(0)
    trials['prev_choice'] = trials['prev_choice'].fillna(0) # 0 表示无选择

    user_rating_std = item_likes.groupby('user_id')['like'].std()
    low_var_users = user_rating_std[user_rating_std == 0].index.tolist()
    trials['valid_preference_data'] = ~trials['subject_id'].isin(low_var_users)

    cols_to_keep = [
        'subject_id', 'trial', 'question_id', 'category_idx', 'reward', 
        'valid_preference_data', 'prev_reward', 'prev_choice'
    ] 
    cols_to_keep += [f'like_cat{c}' for c in range(1, 5)]
    cols_to_keep += [f'score_cat{c}' for c in range(1, 5)]
    
    data_final = trials[cols_to_keep].copy()
    
    return data_final, low_var_users

df_analysis, excluded_users = preprocess_data()

print(df_analysis.head())
print(df_analysis.tail())