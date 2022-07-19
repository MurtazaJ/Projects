import os
#Main Libs
import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
import pickle
warnings.filterwarnings("ignore")

#Visualization Libs
import matplotlib.pyplot as plt
import seaborn           as sns
import plotly.express    as px
import plotly.offline    as pyoff
import plotly.graph_objs as go


# Sklearn Libs
from xgboost                   import XGBRegressor
from sklearn.svm               import SVR
from sklearn.tree              import DecisionTreeRegressor
from sklearn.neighbors         import KNeighborsRegressor
from sklearn.ensemble          import RandomForestRegressor
from sklearn.linear_model      import LinearRegression
from sklearn.datasets          import make_regression
from sklearn. linear_model     import Lasso
from sklearn                   import metrics

#Misc Libs
from sklearn.model_selection   import KFold, cross_val_score, train_test_split
from sklearn.cluster           import KMeans
from sklearn.model_selection   import GridSearchCV

# df = pd.read_excel (r'data/data.xlsx')

def rfm_analysis(df):
    df['order_date'] = pd.to_datetime(df['date_registered']) + pd.to_timedelta(df['day_after_registration'], unit='D')
    df_10_days = df[(df.day_after_registration < 10)].reset_index(drop =True)
    cust_ids = df_10_days.player_identifier.unique() 
    df_data_10_days = df[df.player_identifier.isin(cust_ids)]
    tx_less_2yrs = df_data_10_days[(df_data_10_days.order_date <= '2019-11-30')]
    tx_less_2yrs_last_purchase = df.groupby(['player_identifier']).order_date.max().reset_index()
    tx_less_2yrs_last_purchase.columns = ['player_identifier', 'MaxPurchaseDate']

    # Calculate Recency
    tx_less_2yrs_last_purchase['Recency'] = (tx_less_2yrs_last_purchase['MaxPurchaseDate'].max()  \
                                          - (tx_less_2yrs_last_purchase['MaxPurchaseDate'])) \
                                             /pd.to_timedelta(1,'day')
    kmeans_rec = KMeans(n_clusters=4)
    kmeans_rec.fit(tx_less_2yrs_last_purchase[['Recency']])
    tx_less_2yrs_last_purchase['RecencyCluster'] = kmeans_rec.predict(tx_less_2yrs_last_purchase[['Recency']])
    pickle.dump(kmeans_rec, open('pickled/recency_cluster.pkl', 'wb'))

    # Calculate Frequency
    tx_frequency = tx_less_2yrs.groupby('player_identifier').order_date.count().reset_index()
    tx_frequency.columns = ['player_identifier','Frequency'] 
    kmeans_freq = KMeans(n_clusters=4)
    kmeans_freq.fit(tx_frequency[['Frequency']])
    tx_frequency['FrequencyCluster'] = kmeans_freq.predict(tx_frequency[['Frequency']])
    pickle.dump(kmeans_freq, open('pickled/frequency_cluster.pkl', 'wb'))

    tx_Revenue = tx_less_2yrs.groupby('player_identifier')['prata_spent'].sum().reset_index()
    tx_Revenue = tx_Revenue.rename(columns = {'prata_spent':'prata_spent_total'})
    kmeans_rev = KMeans(n_clusters=4)
    kmeans_rev.fit(tx_Revenue[['prata_spent_total']])
    tx_Revenue['RevenueCluster'] = kmeans_rev.predict(tx_Revenue[['prata_spent_total']])
    pickle.dump(kmeans_rev, open('pickled/revenue_cluster.pkl', 'wb'))
    
    # Overall Data
    overall_data = pd.merge(tx_frequency, tx_Revenue, on='player_identifier', how='left')
    overall_data = pd.merge(overall_data, tx_less_2yrs_last_purchase,  on='player_identifier', how='left')
    overall_data['Overall_Score'] = overall_data['Recency'] + overall_data['Frequency'] + overall_data['prata_spent_total']
    overall_data
    
    kmeans_all = KMeans(n_clusters=4)
    kmeans_all.fit(overall_data[['Overall_Score']])
    overall_data['overall_cluster'] = kmeans_all.predict(overall_data[['Overall_Score']])
    pickle.dump(kmeans_all, open('pickled/all_cluster.pkl', 'wb'))

    over_data1 = overall_data.copy()
    over_data1 = over_data1.drop(['MaxPurchaseDate'], axis = 1)
    # print(over_data1.head())
    #Prediction of Model
    X = over_data1.drop(['prata_spent_total','player_identifier'], axis = 1)
    y = over_data1['prata_spent_total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44)
    print(X_train.head())
    print(y_train.head())

    model_lr = Lasso(alpha=0.005)  # Lasso Regression
    model_lr.fit(X_train,y_train)
    y_pred = model_lr.predict(X_test)
    pickle.dump(model_lr, open('pickled/trained_lasso_model.pkl', 'wb'))
    # print(metrics.mean_squared_error(y_test, y_pred))

    model_kn = KNeighborsRegressor(n_neighbors=4)
    model_kn.fit(X_train,y_train)
    y_pred1 = model_kn.predict(X_test)
    pickle.dump(model_kn, open('pickled/trained_kn_model.pkl', 'wb'))
    return tx_less_2yrs_last_purchase, print(metrics.mean_squared_error(y_test, y_pred))\
          , print(metrics.mean_squared_error(y_test, y_pred1))



def model_test(df):
    df['order_date'] = pd.to_datetime(df['date_registered']) + pd.to_timedelta(df['day_after_registration'], unit='D')

    tx_less_2yrs_last_purchase = df.groupby(['player_identifier']).order_date.max().reset_index()
    tx_less_2yrs_last_purchase.columns = ['player_identifier', 'MaxPurchaseDate']
    
    # Calculate Recency
    tx_less_2yrs_last_purchase['Recency'] = ((tx_less_2yrs_last_purchase['MaxPurchaseDate'].max() + pd.offsets.DateOffset(years=2)) \
                                          - (tx_less_2yrs_last_purchase['MaxPurchaseDate'])) \
                                              / pd.to_timedelta(1,'day')
    kmeans_rec = pickle.load(open('pickled/recency_cluster.pkl', 'rb'))
    tx_less_2yrs_last_purchase['RecencyCluster'] = kmeans_rec.predict(tx_less_2yrs_last_purchase[['Recency']])
    # print(tx_less_2yrs_last_purchase.head())



    # Calculate Frequency
    tx_frequency = df.groupby('player_identifier').order_date.count().reset_index()
    tx_frequency.columns = ['player_identifier','Frequency']
    kmeans_freq = pickle.load(open('pickled/frequency_cluster.pkl', 'rb'))
    tx_frequency['FrequencyCluster'] = kmeans_freq.predict(tx_frequency[['Frequency']])


    # Calculate Revenue
    tx_Revenue = df.groupby('player_identifier')['prata_spent'].sum().reset_index()
    tx_Revenue = tx_Revenue.rename(columns = {'prata_spent':'prata_spent_total'})
    kmeans_rev = pickle.load(open('pickled/revenue_cluster.pkl', 'rb'))
    tx_Revenue['RevenueCluster'] = kmeans_rev.predict(tx_Revenue[['prata_spent_total']])


    # Calculate Overall Score    
    overall_data = pd.merge(tx_frequency, tx_Revenue, on='player_identifier', how='left')
    overall_data = pd.merge(tx_less_2yrs_last_purchase, overall_data,  on='player_identifier', how='left')
    overall_data['Overall_Score'] = overall_data['Recency'] + overall_data['Frequency'] + overall_data['prata_spent_total']  
    kmeans_all = pickle.load(open('pickled/all_cluster.pkl', 'rb'))
    overall_data['Overall_Cluster'] = kmeans_all.predict(overall_data[['Overall_Score']])

    over_data1 = overall_data.copy()
    over_data1 = over_data1.drop(['MaxPurchaseDate'], axis = 1)
    # print(over_data1)

    X = over_data1.drop(['prata_spent_total','player_identifier'], axis = 1)
    y = over_data1['prata_spent_total']

    model_lr = pickle.load(open('pickled/trained_lasso_model.pkl', 'rb'))
    y_pred = model_lr.predict(X)

    model_kn = pickle.load(open('pickled/trained_kn_model.pkl', 'rb'))
    y_pred1 = model_kn.predict(X)
    print('Lasso model: Predicted prata spent', y_pred)
    print('kn Neighbours model: Predicted prata spent', y_pred1)
    return over_data1


# print(rfm_analysis(df))
# print(model_test(df))


