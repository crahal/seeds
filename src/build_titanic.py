import pandas as pd
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


def wrangle_titanic(train_df):
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    train_df['Title'] = train_df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
    train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')
    train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
    train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    train_df['Title'] = train_df['Title'].map(title_mapping)
    train_df['Title'] = train_df['Title'].fillna(0)
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    guess_ages = np.zeros((2,3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = train_df[(train_df['Sex'] == i) & \
                                (train_df['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5 ) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            train_df.loc[(train_df.Age.isnull()) & (train_df.Sex == i) &
                         (train_df.Pclass == j+1), 'Age'] = guess_ages[i,j]
    train_df['Age'] = train_df['Age'].astype(int)
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    train_df.loc[train_df['Age'] <= 16, 'Age'] = 0
    train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1
    train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2
    train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3
    train_df.loc[train_df['Age'] > 64, 'Age'] = 5
    train_df = train_df.drop(['AgeBand'], axis=1)
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    train_df['IsAlone'] = 0
    train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    train_df['Age*Class'] = train_df.Age * train_df.Pclass
    freq_port = train_df.Embarked.dropna().mode()[0]
    train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
    train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    train_df.loc[train_df['Fare'] <= 7.91, 'Fare'] = 0
    train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1
    train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare']   = 2
    train_df.loc[train_df['Fare'] > 31, 'Fare'] = 3
    train_df['Fare'] = train_df['Fare'].dropna().astype(int)
    train_df = train_df.drop(['FareBand'], axis=1)
    if 'Survived' in train_df.columns:
        X_train = train_df.drop("Survived", axis=1)
        Y_train = train_df["Survived"]
    else:
        X_train = train_df
        Y_train = None
    return X_train, Y_train

def get_scores(Y_test, y_prob, y_train):

    def calc_r2(truth, pred, ybar_train):
        """Replicate FFC eval metric"""
        pred_err_sq = (truth - pred) ** 2
        sum_pred_err_sqr = pred_err_sq.sum()
        dev_sqr = (truth - ybar_train) ** 2
        sum_dev_sqr = dev_sqr.sum()
        r2 = 1 - (sum_pred_err_sqr / sum_dev_sqr)
        return r2

    def ll(x, p):
        """x is the truth, p is the guess"""
        z = (np.log(p) * x) + (np.log(1 - p) * (1 - x))
        return np.exp(np.sum(z) / len(z))

    def get_w(a, guess=0.5, bounds=[(0.001, 0.999)]):
        """argmin calc for 'w'"""
        res = minimize(minimize_me, guess, args=a,
                       options={'ftol': 0, 'gtol': 1e-09},
                       method='L-BFGS-B', bounds=bounds)
        return res['x'][0]

    def minimize_me(p, a):
        """ function to be minimized"""
        # abs(p*log(p)+(1-p)*log(1-p)-log(a))
        return abs((p * np.log(p)) + ((1 - p) * np.log(1 - p)) - np.log(a))

    def get_ew(w0, w1):
        """calculate the e(w) metric from w0 and w1"""
        return (w1 - w0) / w0

    y_prob = [x + 0.0001 if x == 0 else x for x in y_prob]
    y_prob = np.array([x - 0.001 if x == 1 else x for x in y_prob])
    score_list = []
    score_list.append(calc_r2(Y_test, y_prob, len(y_prob)*[np.mean(y_train)]))
    score_list.append(get_ew(get_w(ll(Y_test, np.mean(y_train))), get_w(ll(Y_test, y_prob))))
    return score_list


def get_predictions(X_train, Y_train, X_test, Y_test, model_seed):
    random_forest = RandomForestClassifier(random_state=model_seed)
    random_forest.fit(X_train, Y_train)
    Y_rf_pred_class = random_forest.predict(X_test)
    Y_rf_pred_proba = random_forest.predict_proba(X_test)[:,1]
    rf_scores = get_scores(Y_test, Y_rf_pred_proba, Y_train)
    score_holder = pd.DataFrame(list(zip(rf_scores)),
        columns=['RF'],
        index=['R2', 'IMV']
        )
    return score_holder


def process_seed(folding_seed, X, y, n_fold, seed_list):
    skf = KFold(n_splits=n_fold, random_state=folding_seed, shuffle=True)
    results = []
    for model_seed in seed_list:    
        score_holder = None
        counter = 0
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            score_temp = get_predictions(X_train, y_train, X_test, y_test, model_seed)
            if counter == 0:
                score_holder = score_temp
            else:
                score_holder += score_temp
            counter += 1
        score_holder = (score_holder / n_fold).round(decimals=4)
        results.append({
            'Folding_Seed': folding_seed,
            'Modeling_Seed': model_seed,
            'R2': list(score_holder['RF'])[0],
            'IMV': list(score_holder['RF'])[1]
        })    
    return results


def get_seed_list():
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


def main(data_path, table_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    X, y = wrangle_titanic(train_df)
    n_fold = 10
    seed_limit = 1000
    seed_list = get_seed_list()[0:seed_limit]
    results = Parallel(n_jobs=10)(delayed(process_seed)(folding_seed, X, y, n_fold, seed_list) for folding_seed in tqdm(seed_list))
    flattened_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened_results)
    df.to_csv(os.path.join(table_path, 'titanic_outputs.csv'), index=False)

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), '..', 'data', 'titanic', 'raw')
    table_path = os.path.join(os.getcwd(), '..', 'data', 'titanic', 'results')
    main(data_path, table_path)