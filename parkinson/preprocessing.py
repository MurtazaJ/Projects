#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from utils import permutation_importance, mi_score, plot_mi

#%%
def create_df():
    df= pd.read_csv("data/parkinson_disease.csv")
    print(df.head())
    print(df.name.nunique() / df.shape[0])
    print(df.name.nunique())
    print(df.shape[0])
    print(df.shape)
    print(df.isnull().values.sum())    
    df.nunique()
    return df


def train_and_test(df):
    x = df.drop(['status','name' ], axis=1)
    y = df.status
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


