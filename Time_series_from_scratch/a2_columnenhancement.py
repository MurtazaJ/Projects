#%%
from datetime import datetime
import pandas as pd
import numpy as np

def date_split(df_train, df_test):
    if type(df_train.date) == datetime:
        pass
    else: 
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_test['date'] = pd.to_datetime(df_test['date'])
        df_train['day'] = df_train['date'].map(lambda x: x.day)
        df_train['month'] = df_train['date'].map(lambda x: x.month)
        df_train['year'] = df_train['date'].map(lambda x: x.year)
        df_train.drop(['date'], axis = 1, inplace=True)
        df_test['day'] = df_test['date'].map(lambda x: x.day)
        df_test['month'] = df_test['date'].map(lambda x: x.month)
        df_test['year'] = df_test['date'].map(lambda x: x.year)
        df_test.drop(['date'], axis = 1, inplace=True)
    return df_train, df_test



