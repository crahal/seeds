import numpy as np
import random
import multiprocessing as mp
from tqdm import tqdm
import os
import csv


def sample_cosine():
    rr=2.
    while rr > 1.:
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        v1 = 2*u1-1
        rr = v1*v1+u2*u2
    cc=(v1*v1-u2*u2)/rr
    return cc


class Buffon_needle_problem:
    """Acknowledgement: THanks to jpf on Stack Overflow
       for the inspiration for this!
       (Response to question 31291174)
       """
    def __init__(self, x, y, n, m):
        self.x = float(x)
        self.y = float(y)
        self.r = []
        self.z = []
        self.n = n
        self.m = m
        self.p = self.x/self.y

    def samples(self):
        for _ in range(self.n):
            self.r.append(random.uniform(0, self.y))
            C=sample_cosine()
            self.z.append(C*self.x/2.)
        return [self.r, self.z]

    def simulation(self):
        for j in range(self.m):
            self.r=[]
            self.z=[]
            self.samples()
            hits = 0
            for i in range(self.n):
                if self.r[i]+self.z[i] >= self.y or self.r[i]-self.z[i] < 0.:
                    hits += 1
                else:
                    continue
            est = self.p*float(self.n)/float(hits)
        return est


def outter_wrapper(n_throws):
    seed_limit = 5000
    seed_results = []
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        seed_list = [int(line.rstrip('\n')) for line in f][0:seed_limit]
    for seed in seed_list:
        random.seed(seed)
        seed_results.append(Buffon_needle_problem(1,2, n_throws, 1).simulation())
    return [n_throws,
            np.min(seed_results),
            np.percentile(seed_results, [75 ,25])[1],
            np.median(seed_results),
            np.percentile(seed_results, [75 ,25])[0],
            np.max(seed_results)]


if __name__ == "__main__":
    start_range = 1000
    finish_range = 50000
    step = 10
    throw_range = range(start_range, finish_range, step)
    n_cores = 4
    pool = mp.Pool(n_cores)
    print(f'Up and running with {n_cores} cores!')
    all_results = list(tqdm(pool.imap(outter_wrapper, throw_range),
                            total= int(finish_range - start_range) / step))
    with open(os.path.join(os.getcwd(), '..', 'data','needles', 'results',
                           f'throw{start_range}_{finish_range}_5000seeds.csv'), "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(all_results)
