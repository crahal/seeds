from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import warnings
import csv
warnings.filterwarnings("ignore", category=DeprecationWarning)


def save_results(straps):
    straps = [list(i) for i in zip(*straps)]
    out_path = os.path.join(os.getcwd(), '..', 'data', 'ehrlich',
                            'results', 'ehrlich_bootstraps.csv')
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(straps)


def get_data():
    uscrime_path = os.path.join(os.getcwd(), '..',
                                'data', 'ehrlich',
                                'raw', 'uscrime.txt')
    if os.path.exists(uscrime_path):
        df = pd.read_csv(uscrime_path, sep='\t')
    else:
        df = pd.read_csv("http://www.statsci.org/data/general/uscrime.txt", sep='\t')
        df.to_csv(uscrime_path, sep='\t')
    df['lncrime'] = np.log(df['Crime'])
    df['lnprob'] = np.log(df['Prob'])
    df['lntime'] = np.log(df['Time'])
    df['lnwealth'] = np.log(df['Wealth'])
    df['lnineq'] = np.log(df['Ineq'])
    df['lnnw'] = np.log(df['NW'])
    df['sqrtpop'] = np.sqrt(df['Pop'])
    return df


def print_full_model():
    lin_reg = LinearRegression()
    df = get_data()
    model_vars = ['lnprob', 'lntime', 'lnwealth', 'lnineq', 'lnnw']
    full_model = lin_reg.fit(df[model_vars], df['lncrime'], sample_weight=df['sqrtpop'])
    print(full_model.coef_)


def outter_wrapper(new_seed_list):
    coefs = []
    lin_reg = LinearRegression()
    model_vars = ['lnprob', 'lntime', 'lnwealth', 'lnineq', 'lnnw']
    df = get_data()
    for i in new_seed_list:
        sample = resample(df, replace=True, n_samples=len(df),
                          random_state=i)
        lin_reg.fit(sample[model_vars], sample['lncrime'],
                    sample_weight=sample['sqrtpop'])
        coefs.append(lin_reg.coef_.ravel()[0])
    return coefs


if __name__ == "__main__":
  seed_list_path = os.path.join(os.getcwd(), '..', 'assets',
                                'seed_list_of_lists.txt')
  seed_lists = pd.read_csv(seed_list_path, header=None)
  print_full_model()
  n_cores = 6
  pool = mp.Pool(n_cores)
  df = get_data()
  model_vars = ['lnprob', 'lntime', 'lnwealth', 'lnineq', 'lnnw']
  straps = []
  list_of_lists = seed_lists.T.values.tolist()
  all_results = list(tqdm(pool.imap(outter_wrapper, list_of_lists),
                          total=len(list_of_lists)))
  save_results(all_results)
