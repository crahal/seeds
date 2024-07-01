import os
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def main():
    # Parameters
    N = 1000  # Total population
    I0 = 10  # Initial number of infected individuals
    R0 = 0  # Initial number of recovered individuals
    beta = 0.3  # Infection rate
    gamma = 0.1  # Recovery rate
    T = 100  # Total time

    # Initial number of susceptible individuals
    S0 = N - I0 - R0

    # Time, Susceptible, Infected, Recovered arrays
    time = [0]
    S = [S0]
    I = [I0]
    R = [R0]


    def get_seed_list():
        seed_list_path = os.path.join(os.getcwd(), '..', 'assets', 'seed_list.txt')
        with open(seed_list_path) as f:
            return [int(line.rstrip('\n')) for line in f]


    seed_list = get_seed_list()

    # Gillespie Algorithm Simulation
    df_S = pd.DataFrame()
    df_I = pd.DataFrame()
    df_R = pd.DataFrame()
    df_time = pd.DataFrame()
    for seed in tqdm(seed_list[0:10000]):
        np.random.seed(seed)
        time = [0]
        S = [S0]
        I = [I0]
        R = [R0]
        while time[-1] < T and I[-1] > 0:
            S_current = S[-1]
            I_current = I[-1]
            R_current = R[-1]

            # Rates of events
            infection_rate = beta * S_current * I_current / N
            recovery_rate = gamma * I_current

            # Total rate
            total_rate = infection_rate + recovery_rate

            if total_rate == 0:
                break

            # Time until next event
            tau = np.random.exponential(1 / total_rate)
            time.append(time[-1] + tau)

            # Determine which event occurs
            if np.random.rand() < infection_rate / total_rate:
                # Infection event
                S.append(S_current - 1)
                I.append(I_current + 1)
                R.append(R_current)
            else:
                # Recovery event
                S.append(S_current)
                I.append(I_current - 1)
                R.append(R_current + 1)
        time_series = pd.Series(time)
        S = pd.Series(S)
        I = pd.Series(I)
        R = pd.Series(R)

        df_S = pd.concat([df_S, S])
        df_I = pd.concat([df_I, I])
        df_R = pd.concat([df_R, R])
        df_time = pd.concat([df_time, time_series])

    df_S = df_S.set_index(df_time[0])
    df_I = df_I.set_index(df_time[0])
    df_R = df_R.set_index(df_time[0])

    df_final = pd.DataFrame(columns=['Susceptible_min',
                                     'Susceptible_max',
                                     'Susceptible_med',
                                     'Infected_min',
                                     'Infected_max',
                                     'Infected_med',
                                     'Recovered_min',
                                     'Recovered_max',
                                     'Recovered_med']
                            )
    for timer in tqdm(np.arange(df_S.index.min(), df_S.index.max(), 0.1)):
        temp = df_S[df_S.index.astype(float).round(1) == timer]
        df_final.loc[timer, 'Susceptible_min'] = temp.min().iloc[0]
        df_final.loc[timer, 'Susceptible_max'] = temp.max().iloc[0]
        df_final.loc[timer, 'Susceptible_med'] = temp.median().iloc[0]

        temp = df_I[df_I.astype(float).index.round(1) == timer]
        df_final.loc[timer, 'Infected_min'] = temp.min().iloc[0]
        df_final.loc[timer, 'Infected_max'] = temp.max().iloc[0]
        df_final.loc[timer, 'Infected_med'] = temp.median().iloc[0]

        temp = df_R[df_R.index.astype(float).round(1) == timer]
        df_final.loc[timer, 'Recovered_min'] = temp.min().iloc[0]
        df_final.loc[timer, 'Recovered_max'] = temp.max().iloc[0]
        df_final.loc[timer, 'Recovered_med'] = temp.median().iloc[0]

    df_final.to_csv(os.path.join('..', 'data', 'sir', 'sir_seeds_1dp.csv'), sep=',')

if __name__ == '__main__':
    main()