import os
import csv
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_seed_list():
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


def make_kfold():

    np.random.seed(42)
    alpha = 200
    beta = 300
    n = 1000
    seed_limit = 1000
    seed_list = get_seed_list()[0:seed_limit]
    x = np.random.randn(n)
    epsilon = np.random.randn(n)
    y = alpha + beta * x + epsilon
    range_list = []
    median_list = []
    var_list = []
    min_list = []
    max_list = []
    k_list = []
    for k_folds in tqdm(range(2, n + 1, 1)):
        mean_list = []
        for seed in seed_list:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            mse_list = []
            for train_index, test_index in kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                x_train = x_train.reshape(-1, 1)
                x_test = x_test.reshape(-1, 1)
                model = LinearRegression()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)
            mean_list.append(np.mean(mse_list))
        median_list.append(np.median(mean_list))
        min_list.append(np.min(mean_list))
        max_list.append(np.max(mean_list))
        k_list.append(k_folds)
        range_list.append(abs(np.max(mean_list) - np.min(mean_list)))
        var_list.append(np.var(mean_list))

    combined_lists = zip(k_list, min_list, max_list, range_list, var_list, median_list)
    csv_file_path = f'output_{n}_{seed_limit}.csv'
    with open(os.path.join(os.getcwd(),
                           '..',
                           'data',
                           'kfold_leavep_equivalence',
                           csv_file_path),
              'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['k', 'min', 'max', 'range', 'var', 'median'])
        writer.writerows(combined_lists)


def make_rnorm():

    def get_seed_list():
        seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
        with open(seed_list_path) as f:
            return [int(line.rstrip('\n')) for line in f]

    def generate_random_samples(seed, num_samples):
        np.random.seed(seed)
        return np.random.normal(size=num_samples)

    def update_min_max_values(random_samples, min_val, max_val, min_list, max_list):
        mean_val = np.mean(random_samples)
        if mean_val < min_val:
            min_val = mean_val
            min_list = random_samples
        if mean_val > max_val:
            max_val = mean_val
            max_list = random_samples
        return min_val, max_val, min_list, max_list

    start_val = 10
    end_val = 1000000
    num_increments = 100
    round_numbers = np.geomspace(start_val, end_val, num=num_increments, dtype=int)
    nested_results = {}
    for num_samples in tqdm(round_numbers):
        min_val = np.inf
        max_val = -np.inf
        seed_list = get_seed_list()
        min_list = []
        max_list = []
        for seed in seed_list[0:100000]:
            random_samples = generate_random_samples(seed, num_samples)
            min_val, max_val, min_list, max_list = update_min_max_values(
                random_samples, min_val, max_val, min_list, max_list
            )
        results_dict = {
            "min_val": min_val,
            "max_val": max_val,
        }
        nested_results[int(num_samples)] = results_dict

    filename = f'rnom_samples_dontmatter_seeds{len(seed_list)}_results.json'
    output_file_path = os.path.join(os.getcwd(), '..', 'data', 'rnom', 'doesnt_matter', filename)
    with open(output_file_path, 'w') as json_file:
        json.dump(nested_results, json_file)

def main():
    make_kfold()
    make_rnorm()

if __name__ == "__main__":
    main()