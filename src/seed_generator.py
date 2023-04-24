import os
import secrets


def generate_seeds():
    return int.from_bytes(secrets.token_bytes(32), 'big')


if __name__ == "__main__":
    seed_list = []
    number_seeds = 10000
    seed_list = [generate_seeds() for _ in range(number_seeds)]

    # Avoiding the birthday paradox?
    if len(seed_list) == len(set(seed_list)):
        seed_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
        with open(seed_path, 'w') as file:
            for seed in seed_list:
                file.write("%i\n" % seed)
    else:
        print('Wuhoh, we have two birthdays in the room!')