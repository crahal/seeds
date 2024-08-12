import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def stochastic_solow_growth_model(T=100, seed=0):
    np.random.seed(seed)
    # Parameters with random fluctuations
    s = 0.3 + np.random.normal(0, 0.05)  # Savings rate with noise
    alpha = 0.3  # Capital share (assuming it's constant)
    delta = 0.05 + np.random.normal(0, 0.01)  # Depreciation rate with noise
    n = 0.02  # Population growth rate (assuming it's constant)
    A = 1.0 + np.random.normal(0, 0.1)  # Total Factor Productivity with noise
    K0 = 10  # Initial capital stock
    L0 = 100  # Initial labor

    time = np.arange(T)
    K = np.zeros(T)
    L = np.zeros(T)
    Y = np.zeros(T)

    K[0] = K0
    L[0] = L0

    for t in range(1, T):
        K[t] = K[t - 1] + s * Y[t - 1] - delta * K[t - 1]
        L[t] = L[t - 1] * (1 + n)
        Y[t] = A * K[t] ** alpha * L[t] ** (1 - alpha)

    return time, K, L, Y, s, delta, A


def write_combined_stochastic_solow_growth(seeds, filename='solow_growth_results.csv'):
    results = []
    for seed in tqdm(seeds):
        time, K, L, Y, s, delta, A = stochastic_solow_growth_model(seed=seed)
        for t in range(len(time)):
            results.append({
                'Seed': seed,
                'Time': time[t],
                'Capital Stock': K[t],
                'Labor': L[t],
                'Output': Y[t],
                'Savings Rate': s,
                'Depreciation Rate': delta,
                'TFP': A
            })
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(os.getcwd(), '..', 'data', 'solow', filename), index=False)
    print(f"Data written to {filename}")


def get_seed_list():
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


def main():
    seeds = get_seed_list()
    write_combined_stochastic_solow_growth(seeds)


if __name__ == '__main__':
    main()