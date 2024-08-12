import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, TimeoutError
from functools import partial


def initialize_grid(grid_size, empty_prob):
    agent_prob = (1 - empty_prob) / 2
    grid = np.random.choice([0, 1, -1], size=(grid_size, grid_size), p=[agent_prob, agent_prob, empty_prob])
    return grid

def is_happy(grid, x, y, threshold):
    type_at_cell = grid[x, y]
    if type_at_cell == -1:  # Empty cell
        return True

    neighbors = [(i, j) for i in range(max(0, x - 1), min(grid.shape[0], x + 2))
                 for j in range(max(0, y - 1), min(grid.shape[1], y + 2))
                 if (i, j) != (x, y)]

    same_type_count = sum(grid[i, j] == type_at_cell for i, j in neighbors)
    total_agents = sum(grid[i, j] != -1 for i, j in neighbors)

    if total_agents == 0:
        return True

    satisfaction = same_type_count / total_agents
    return satisfaction >= threshold

def get_empty_cells(grid):
    return [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == -1]


def get_unhappy_cells(grid, threshold):
    return [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])
            if grid[i, j] != -1 and not is_happy(grid, i, j, threshold)]


def move_agents(grid, threshold):
    unhappy_cells = get_unhappy_cells(grid, threshold)
    empty_cells = get_empty_cells(grid)

    if not unhappy_cells or not empty_cells:
        return False  # Convergence
    random.shuffle(unhappy_cells)
    random.shuffle(empty_cells)
    for (x, y) in unhappy_cells:
        if not empty_cells:
            break
        new_x, new_y = empty_cells.pop()
        grid[new_x, new_y], grid[x, y] = grid[x, y], -1

    return True


def schelling_model(seed, grid_size, empty_prob, threshold):
    random.seed(seed)
    np.random.seed(seed)
    grid = initialize_grid(grid_size, empty_prob)
    steps = 0
    happy_counts = []

    while True:
        happy_count = sum(is_happy(grid, i, j, threshold) for i in range(grid_size) for j in range(grid_size))
        happy_counts.append(happy_count)
        if not move_agents(grid, threshold):
            break
        steps += 1
    return seed, steps, happy_counts


def worker(seed, grid_size, empty_prob, threshold):
    try:
        return schelling_model(seed, grid_size, empty_prob, threshold)
    except Exception as e:
        print(f"Error processing seed {seed}: {e}")
        return seed, None, []


if __name__ == "__main__":
    GRID_SIZE = 200
    EMPTY_PROB = 0.3  # Probability of an empty cell
    SAME_THRESHOLD = 0.5  # Threshold for satisfaction
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        seed_list = [int(line.rstrip('\n')) for line in f][0:10000]
    all_results = []
    num_processes = min(cpu_count(), 8)  # Adjust the number as needed
    with Pool(num_processes) as pool:
        worker_partial = partial(worker, grid_size=GRID_SIZE, empty_prob=EMPTY_PROB, threshold=SAME_THRESHOLD)
        results = []
        for seed in tqdm(seed_list, total=len(seed_list), desc="Processing seeds"):
            try:
                result = pool.apply_async(worker_partial, args=(seed,))
                results.append(result.get(timeout=180))  # Timeout after 60 seconds
            except TimeoutError:
                print(f"Timeout processing seed {seed}")
                results.append((seed, None, []))
        for seed, steps, happy_counts in results:
            if steps is not None:
                for step, happy_count in enumerate(happy_counts):
                    all_results.append({
                        'Seed': seed,
                        'Step': step,
                        'Happy Count': happy_count
                    })
                all_results.append({
                    'Seed': seed,
                    'Step': 'Convergence',
                    'Happy Count': steps
                })
    df = pd.DataFrame(all_results)
    summary = df[df['Step'] == 'Convergence'].groupby('Seed').agg({'Happy Count': 'mean'}).reset_index()
    summary.columns = ['Seed', 'Total Steps to Converge']
    df.to_csv(os.path.join(os.getcwd(), '..', 'data', 'schelling', 'schelling_df.csv'))
    summary.to_csv(os.path.join(os.getcwd(), '..', 'data', 'schelling', 'schelling_summary.csv'))
