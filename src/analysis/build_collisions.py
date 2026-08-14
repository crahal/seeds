import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_seed_list():
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        return [int(line.rstrip('\n')) for line in f]


def main():
    num_iterations = 5  # length(seeds)
    n = 1000000
    df = pd.DataFrame(index=range(n), columns=range(num_iterations))
    seed_list = get_seed_list()[:num_iterations]

    for i in tqdm(range(num_iterations)):
        np.random.seed(seed_list[i])
        U = np.random.random(n).astype(np.float32)
        df[i] = pd.Series(U).duplicated().cumsum()
    df.to_csv(os.path.join(os.getcwd(),
                           '..',
                           'data',
                           'output_list_Python.csv'),
              index=False)


if __name__ == '__main__':
    main()