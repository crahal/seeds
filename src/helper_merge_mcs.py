from tqdm import tqdm
import os
import glob
import pandas as pd


def main():
    csv_path = os.path.join(os.getcwd(), '..', 'data', 'mcs', 'results', 'csv_files')
    merged_path = os.path.join(os.getcwd(), '..', 'data', 'mcs', 'results', 'merged_files')
    files = [file_path for file_path in os.listdir(csv_path) if file_path.endswith('.csv')]
    df = None
    for fname in tqdm(files):
        path = os.path.join(csv_path, fname)
        if df is None:
            df = pd.read_csv(path, index_col=None, usecols=['x'])
        else:
            temp_df = pd.read_csv(path, index_col=None, usecols=['x'])
            temp_df = temp_df.rename({'x': fname.split('.')[0]}, axis=1)
            df = pd.merge(df, temp_df, left_index=True, right_index=True)
    df = df.round(4)
    df.to_csv(os.path.join(merged_path, 'merged_csvs.csv'), index=False)


if __name__ == "__main__":
    main()
