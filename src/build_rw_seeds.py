import os
import csv
import numpy as np
import yfinance as yf
from tqdm import tqdm
from pandas_datareader import data as pdr


if __name__ == "__main__":
    rw_path = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks.csv')
    yf.pdr_override()
    btc_data = pdr.get_data_yahoo('BTC-USD', start="2021-07-01", end="2024-06-30")
    usuk_data = pdr.get_data_yahoo('USDGBP=X', start="2021-07-01", end="2024-06-30")
    nasdaq_data = pdr.get_data_yahoo('^IXIC', start="2021-07-01", end="2024-06-30")
    nvidia_data = pdr.get_data_yahoo('NVDA', start="2021-07-01", end="2024-06-30")
    size = 366
    start_btc = btc_data['Close'].iloc[-1]
    start_usuk = usuk_data['Close'].iloc[-1]
    start_nasdaq = nasdaq_data['Close'].iloc[-1]
    start_nvidia = nvidia_data['Close'].iloc[-1]
    half_sigma_btc = btc_data['Close'].pct_change().std()/2
    change_btc = btc_data['Close'].pct_change().mean()
    half_sigma_usuk = usuk_data['Close'].pct_change().std()/2
    change_usuk = usuk_data['Close'].pct_change().mean()
    half_sigma_nasdaq = nasdaq_data['Close'].pct_change().std()/2
    change_nasdaq = nasdaq_data['Close'].pct_change().mean()
    half_sigma_nvidia = nvidia_data['Close'].pct_change().std()/2
    change_nvidia = nvidia_data['Close'].pct_change().mean()
    seed_limit = 100000
    seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
    with open(seed_list_path) as f:
        seed_list = [int(line.rstrip('\n')) for line in f][0:seed_limit]
    array_list_btc = []
    array_list_usuk = []
    array_list_nasdaq = []
    array_list_nvidia = []
    for seed in tqdm(seed_list):
        rw_btc = []
        rw_usuk = []
        rw_nasdaq = []
        rw_nvidia = []
        rw_btc.append(start_btc)
        rw_usuk.append(start_usuk)
        rw_nasdaq.append(start_nasdaq)
        rw_nvidia.append(start_nvidia)
        for days in range(0, size):
            daily_change_btc = rw_btc[days]*np.random.normal(change_btc,
                                                             half_sigma_btc)
            rw_btc.append(rw_btc[days] + daily_change_btc)

            daily_change_usuk = rw_usuk[days]*np.random.normal(change_usuk,
                                                               half_sigma_usuk)
            rw_usuk.append(rw_usuk[days] + daily_change_usuk)
            daily_change_nasdaq = rw_nasdaq[days]*np.random.normal(change_nasdaq,
                                                                   half_sigma_nasdaq)
            rw_nasdaq.append(rw_nasdaq[days] + daily_change_nasdaq)
            daily_change_nvidia = rw_nvidia[days]*np.random.normal(change_nvidia,
                                                                   half_sigma_nvidia)
            rw_nvidia.append(rw_nvidia[days] + daily_change_nvidia)
        array_list_btc.append(rw_btc)
        array_list_usuk.append(rw_usuk)
        array_list_nasdaq.append(rw_nasdaq)
        array_list_nvidia.append(rw_nvidia)
    array_list_btc = [list(i) for i in zip(*array_list_btc)]
    array_list_usuk = [list(i) for i in zip(*array_list_usuk)]
    array_list_nasdaq = [list(i) for i in zip(*array_list_nasdaq)]
    array_list_nvidia = [list(i) for i in zip(*array_list_nvidia)]
    out_path_btc = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_btc.csv')
    out_path_usuk = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_usuk.csv')
    out_path_nasdaq = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_nasdaq.csv')
    out_path_nvidia = os.path.join(os.getcwd(), '..', 'data', 'random_walk', 'random_walks_nvidia.csv')

    with open(out_path_btc, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(array_list_btc)
    with open(out_path_usuk, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(array_list_usuk)

    with open(out_path_nasdaq, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(array_list_nasdaq)

    with open(out_path_nvidia, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(array_list_nvidia)