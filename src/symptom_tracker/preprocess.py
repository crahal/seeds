import numpy as np
import pandas as pd

def simple_preprocess(df):
    df = df[df['YEAR_OF_BIRTH'] < 2004].copy()
    df.loc[:, 'age'] = 2020 - df['YEAR_OF_BIRTH']
    df.loc[:, 'y'] = df['Simple_Result']
    df.loc[:, 'female'] = 1-df['GENDER']
    df.loc[:, 'asmosia'] = np.where(df['LOSS_OF_SMELL'] == True, 1, 0)
    df.loc[:, 'cough'] = np.where(df['PERSISTENT_COUGH'] == True, 1, 0)
    df.loc[:, 'fatigue'] = np.where(df['FATIGUE'] == "severe", 1, 0)
    df.loc[:, 'skipped_meals'] = np.where(df["SKIPPED_MEALS"] == True, 1, 0)
    df.loc[:, 'shortness_of_breath'] = np.where(df["SHORTNESS_OF_BREATH"] == "severe", 1, 0)
    df.loc[:, 'fever'] = np.where(df["FEVER"] == True, 1, 0)
    df.loc[:, 'delirium'] = np.where(df["DELIRIUM"] == True, 1, 0)
    df.loc[:, 'date'] = pd.to_datetime(df['UPDATED_AT.x'].str.split('-').str[0:3].str.join('-'))
    df.loc[:, 'week'] = df['date'].dt.isocalendar().week
    df.loc[:, 'week'] = df['week'] - df['week'].min()
    df.loc[:, 'diarrhoea'] = np.where(df["DIARRHOEA"] == True, 1, 0)
    df.loc[:, 'hoarse'] = np.where(df["HOARSE_VOICE"] == True, 1, 0)
    df.loc[:, 'abdominal_pain'] = np.where(df["ABDOMINAL_PAIN"] == True, 1, 0)
    df.loc[:, 'chest_pain'] = np.where(df["CHEST_PAIN"] == True, 1, 0)
    df = df[(df['HEIGHT_CM'] >= 110) & (df['HEIGHT_CM'] <=220)]
    df = df[(df['WEIGHT_KG'] >= 40) & (df['WEIGHT_KG'] <=200)]
    df.loc[:, 'bmi'] = df['WEIGHT_KG']/(np.square(df['HEIGHT_CM']/100))
    df = df[(df['bmi'] >= 14) & (df['bmi'] <=45)]
    
    return df
