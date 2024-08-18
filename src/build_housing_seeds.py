import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
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


def process_seed_pair(folding_seed, modeling_seed, df):
    x = df.iloc[:, :-3].copy()
    x['ocean_proximity'] = df['ocean_proximity']
    x['median_income'] = df['median_income']
    y = df['median_house_value']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=folding_seed)

    r2_score = make_prediction(x_train, x_test, y_train, y_test, modeling_seed)

    return {'Folding_Seed': folding_seed, 'Modeling_Seed': modeling_seed, 'R2': r2_score}


if __name__ == "__main__":
    seed_limit = 1000
    housing_path = os.path.join(os.getcwd(), '..', 'data', 'housing')
    df = pd.read_csv(os.path.join(housing_path, 'raw', 'housing.csv'),
                     converters={'ocean_proximity': converter})
    df = df.dropna()

    seed_list = get_seed_list()[:seed_limit]

    # Parallel processing
    results = Parallel(n_jobs=10)(
        delayed(process_seed_pair)(folding_seed, modeling_seed, df)
        for folding_seed in tqdm(seed_list)
        for modeling_seed in seed_list
    )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save the results
    results_df.to_csv(os.path.join(housing_path, 'results', 'r2.csv'), index=False)
