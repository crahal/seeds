import os
import csv
import secrets
from tqdm import tqdm


def generate_seeds():
    """
    Generates a random integer between 0 and 2147483647 (inclusive).
    """
    while True:
        n = int.from_bytes(secrets.token_bytes(4), 'big')
        if n < 2147483647:
            return n


if __name__ == "__main__":
    number_seeds = 100000
    dups = 1
    while dups >0:
        seed_list = [generate_seeds() for _ in tqdm(range(number_seeds))]
        dups = len(seed_list) - len(set(seed_list))
        # Avoiding the birthday paradox?
        if dups == 0:
            seed_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
            with open(seed_path, 'w') as file:
                for seed in seed_list:
                    file.write("%i\n" % seed)
        else:
            print(f'Wuhoh, we have {dups} birthdays in the room!')

    number_seeds = 200
    n_cols = 1000000
    seed_list_of_lists = []
    for _ in tqdm(range(n_cols)):
        dups = 1
        while dups > 0:
            seed_list = [generate_seeds() for _ in range(number_seeds)]
            dups = len(seed_list) - len(set(seed_list))
            if dups == 0:
                seed_list_of_lists.append(seed_list)
    seed_list_of_lists = [list(i) for i in zip(*seed_list_of_lists)]
    seed_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list_of_lists.txt')
    with open(seed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(seed_list_of_lists)
