import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from tqdm import tqdm
from joblib import Parallel, delayed


def converter(x):
    if x == '<1H OCEAN':
        return 0
    if x == 'INLAND':
        return 1
    if x == 'NEAR OCEAN':
        return 2
    if x == 'NEAR BAY':
        return 3
    else:
        return 4


def get_seed_list():
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


def make_prediction(x_train, x_test, y_train, y_test, modeling_seed):
    clf = RandomForestRegressor(n_estimators=25, max_depth=5, random_state=modeling_seed)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    return explained_variance_score(y_test, y_predict)

def make_prediction_ols(x_train, x_test, y_train, y_test):
    """
    Deterministic ordinary least squares regression.
    """
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return explained_variance_score(y_test, y_pred)


def process_seed_pair(folding_seed, modeling_seed, df):
    x = df.iloc[:, :-3].copy()
    x['ocean_proximity'] = df['ocean_proximity']
    x['median_income'] = df['median_income']
    y = df['median_house_value']

    from sklearn.model_selection import KFold
    skf = KFold(n_splits=5, random_state=folding_seed, shuffle=True)
    scores = []
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scores.append(make_prediction(x_train, x_test, y_train, y_test, modeling_seed))
    r2_score = float(np.mean(scores))

    return {'Folding_Seed': folding_seed, 'Modeling_Seed': modeling_seed, 'R2': r2_score}

def process_seed_ols(folding_seed, df):
    x = df.iloc[:, :-3].copy()
    x['ocean_proximity'] = df['ocean_proximity']
    x['median_income'] = df['median_income']
    y = df['median_house_value']

    from sklearn.model_selection import KFold
    skf = KFold(n_splits=5, random_state=folding_seed, shuffle=True)
    scores = []
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scores.append(make_prediction_ols(x_train, x_test, y_train, y_test))
    r2_score = float(np.mean(scores))

    return {'Folding_Seed': folding_seed, 'R2': r2_score}

def mean_r2_for_seed_ols(seed, df, n_fold=5):
    """
    Mean explained variance across KFold splits for OLS using seed only for fold shuffling.
    """
    from sklearn.model_selection import KFold
    skf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    x = df.iloc[:, :-3].copy()
    x['ocean_proximity'] = df['ocean_proximity']
    x['median_income'] = df['median_income']
    y = df['median_house_value']
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = LinearRegression()
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        scores.append(explained_variance_score(y_test, preds))
    return float(np.mean(scores))


def single_seed_r2(seed, df):
    """
    Compute mean explained variance (R2 proxy) over 5 folds using the same
    seed for both folding and model initialization.
    """
    x = df.iloc[:, :-3].copy()
    x['ocean_proximity'] = df['ocean_proximity']
    x['median_income'] = df['median_income']
    y = df['median_house_value']
    from sklearn.model_selection import KFold
    skf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scores.append(make_prediction(x_train, x_test, y_train, y_test, modeling_seed=seed))
    return float(np.mean(scores))


if __name__ == "__main__":
    seed_limit = 1000
    housing_path = os.path.join(os.getcwd(), '..', 'data', 'housing')
    df = pd.read_csv(os.path.join(housing_path, 'raw', 'housing.csv'),
                     converters={'ocean_proximity': converter})
    df = df.dropna()

    seed_list = get_seed_list()[:seed_limit]

    # Random forest: folding x modeling seeds (existing behavior)
    results = Parallel(n_jobs=10)(
        delayed(process_seed_pair)(folding_seed, modeling_seed, df)
        for folding_seed in tqdm(seed_list, desc="RF folding seeds")
        for modeling_seed in seed_list
    )

    # Ordinary least squares: deterministic model, vary folding seed only
    ols_results = Parallel(n_jobs=10)(
        delayed(process_seed_ols)(folding_seed, df)
        for folding_seed in tqdm(seed_list, desc="OLS folding seeds")
    )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    ols_df = pd.DataFrame(ols_results)

    # Save the results
    results_df.to_csv(os.path.join(housing_path, 'results', 'housing_outputs_rf.csv'), index=False)
    # Keep legacy filename for RF results if needed
    results_df.to_csv(os.path.join(housing_path, 'results', 'r2.csv'), index=False)
    ols_df.to_csv(os.path.join(housing_path, 'results', 'housing_outputs_ols.csv'), index=False)

    # Compute single-seed metrics and write to ../data/housing
    specific_seeds = [42, 123]
    acc_lines = []
    for seed in specific_seeds:
        r2 = single_seed_r2(seed, df)
        r2_ols = mean_r2_for_seed_ols(seed, df)
        acc_lines.append(f"seed={seed}, rf_R2={r2:.4f}, ols_R2={r2_ols:.4f}")
    housing_dir = os.path.join(os.getcwd(), '..', 'data', 'housing')
    os.makedirs(housing_dir, exist_ok=True)
    acc_path = os.path.join(housing_dir, 'accuracy_seeds.txt')
    with open(acc_path, 'w', encoding='utf-8') as f:
        for line in acc_lines:
            f.write(line + "\n")
