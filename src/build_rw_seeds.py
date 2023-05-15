import numpy as np
import csv
import os
import yfinance as yf
from tqdm import tqdm
from pandas_datareader import data as pdr


if __name__ == "__main__":
    rw_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks.csv')
    yf.pdr_override()
    btc_data = pdr.get_data_yahoo('BTC-USD', start="2021-05-08", end="2023-05-15")
    size = 366
    start = btc_data['Close'].iloc[-1]
    half_sigma = btc_data['Close'].pct_change().std()/2
    change = btc_data['Close'].pct_change().mean()
    seed_limit = 10000
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        seed_list = [int(line.rstrip('\n')) for line in f][0:seed_limit]
    array_list = []
    for seed in tqdm(seed_list):
        rw = []
        rw.append(start)
        for days in range(0, size):
            daily_change = rw[days]*np.random.normal(change, half_sigma)
            rw.append(rw[days] + daily_change)
        array_list.append(rw)
    array_list = [list(i) for i in zip(*array_list)]
    seed_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks.csv')
    with open(seed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(array_list)