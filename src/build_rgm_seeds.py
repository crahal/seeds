import networkx as nx
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    seed_limit = 10000
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        seed_list = [int(line.rstrip('\n')) for line in f][0:seed_limit]
    pr_list = []
    nodes = 1000
    for pr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        avg_degree_dists = []
        for seed in tqdm(seed_list):
            np.random.seed(seed)
            G = nx.fast_gnp_random_graph(nodes, pr)
            degree_dist = dict(G.degree())
            avg_degree = sum(degree_dist.values()) / nodes
            avg_degree_dists.append(avg_degree)
        pr_list.append(avg_degree_dists)
    pr_list = [list(i) for i in zip(*pr_list)]
    rgm_path = os.path.join(os.getcwd(), '..', 'data', 'rgms', 'rgms.csv')
    with open(rgm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(pr_list)