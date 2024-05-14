import os
import json
import numpy as np
from tqdm import tqdm

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


def main():

    for num_samples in [100, 1000, 10000, 100000, 1000000]:
        min_val = np.inf
        max_val = -np.inf
        seed_list = get_seed_list()
        min_list = []
        max_list = []

        for seed in tqdm(seed_list):
            random_samples = generate_random_samples(seed, num_samples)
            min_val, max_val, min_list, max_list = update_min_max_values(
                random_samples, min_val, max_val, min_list, max_list
            )

        results_dict = {
            "min_val": min_val,
            "max_val": max_val,
            "min_list": min_list.tolist(),  # Convert to list for JSON serialization
            "max_list": max_list.tolist(),  # Convert to list for JSON serialization
        }

        # Save the dictionary to a JSON file
        filename = f'rnom_samples{num_samples}_seeds{len(seed_list)}_results.json'
        output_file_path = os.path.join(os.getcwd(), '..', 'data', 'rnom', filename)
        with open(output_file_path, 'w') as json_file:
            json.dump(results_dict, json_file)


if __name__ == "__main__":
    main()
