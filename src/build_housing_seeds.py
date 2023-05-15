# Thanks to nnandan15 on Kaggle for the inspiration for this
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from tqdm import tqdm


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


def make_prediction(x_train, x_test, y_train, y_test):
    clf = RandomForestRegressor(n_estimators=25,
                                max_depth=5)
    clf.fit(x_train, y_train)
    RandomForestRegressor(ccp_alpha=0.0,
                          criterion='mse',
                          max_depth=5,
                          max_features='auto',
                          n_estimators=25,
                          n_jobs=None,
                          oob_score=False,
                          random_state=state,
                          verbose=0)
    y_predict = clf.predict(x_test)
    return explained_variance_score(y_test, y_predict)


if __name__ == "__main__":
    seed_limit = 100000
    housing_path = os.path.join(os.getcwd(), '..', 'data', 'housing')
    df = pd.read_csv(os.path.join(housing_path, 'raw', 'housing.csv'),
                     converters={'ocean_proximity':converter})
    df = df.dropna()
    seed_list = get_seed_list()[:seed_limit]
    x = df.iloc[:, :-3]
    r2 = []
    for state in tqdm(seed_list):
        x['ocean_proximity']=df[['ocean_proximity']]
        x['median_income']=df['median_income']
        y = df['median_house_value']
        x_train, x_test, y_train, y_test=train_test_split(x,
                                                          y,
                                                          test_size=0.5,
                                                          random_state=state)
        r2.append(make_prediction(x_train, x_test, y_train, y_test))
    r2 = pd.DataFrame(r2, columns=['r2'])
    r2.to_csv(os.path.join(housing_path, 'results', 'r2.csv'), index=False)