#%%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split,GridSearchCV

import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
warnings.filterwarnings('ignore')

#%%
pth_train = 'usedcar_price_prediction/data/train-data.csv'
pth_test = 'usedcar_price_prediction/data/test-data.csv'


def preprocessing(pth):
    df = pd.read_csv(pth)

    # dropping unnecessary column from both train and test dataset
    df = df.drop('Unnamed: 0',axis=1)
    
    '''print(df.isnull().sum())
    print(df.shape)
    print(df.info())'''

    #Dropping columns with 0.70% null values from the dataset
    df = df.drop('New_Price',axis=1)
    # print(df.columns)
    # print(df.shape)

    #Understand the unique counts of each column
    # unique = []
    # for x in df:
    #     y = len(df[x].unique())
    #     unique.append(y)
    #     unique_cnt = dict(zip(list(df),unique))
    # print(unique_cnt)

    #Create a new feature Comapny
    company = [i.split()[0] for i in df['Name']]
    df.insert(0, 'Company', company)
    # print(df['Company'].value_counts())
    # Removing useless characters from mileage, Engine and Power
    df['Mileage'] = pd.to_numeric(df['Mileage'].str.lower().str.get(0), errors ='coerce')
    df['Engine'] = pd.to_numeric(df['Engine'].str.lower().str.split().str.get(0), errors='coerce')
    df['Power'] = pd.to_numeric(df['Power'].str.lower().str.split().str.get(0), errors='coerce')

    #Data Visualisation
    # Q1. Which company has sold maximum cars?
    # plt.figure(figsize=(6,6))
    # plt.xlabel('Company Name')
    # plt.ylabel('No of cars sold')
    # plt.title('cars sold per company')
    # df['Company'].value_counts(). plot(kind = 'bar')
    # plt.show()

    # Q2. After Driving how many kms do people like to sell
    # company_kms_driven = df.groupby('Company').Kilometers_Driven.mean()
    # company_kms_driven.plot(kind = 'bar')
    # plt.xlabel('Company name')
    # plt.ylabel('kms driven')
    # plt.title("Average Kilometeres vs Company")
    # plt.show()

    #Converting names of card with frequency less than 10 into others to avoide over modelling
    df_name_cars = df.Name.value_counts() 
    name_stats_less_than_10 =df_name_cars[df_name_cars<=10]
    # name_stats_less_than_10
    df.Name = df.Name.apply(lambda x: 'other' if x in name_stats_less_than_10 else x)
    # len(df.Name.unique())


    ##Removing Outliers
    
    #For Mileage creating a mean of mileage less than 5
    # print(df.Mileage[df.Mileage<5].unique())
    #we will replace the mileage <z5 with their mean 
    df.Mileage = [i if i>5 else df.Mileage.mean() for i in df.Mileage ]
    # print(df.Mileage[df.Mileage<5].unique())
    
    # Removing outliers in kilometers
    '''print(df['Kilometers_Driven'].min())
    print(df['Kilometers_Driven'].max())'''
    
    # We see we have an outlier with a kilometer of 1650000 whcih sounds practivally impossible
    df[df['Kilometers_Driven']>1000000]  
    df['Kilometers_Driven'] = [i if i <1000000 else df['Kilometers_Driven'].mean() for i in df.Kilometers_Driven]
   
    # Fille null values in Power with its mean 
    df['Power'] = df['Power'].fillna(df['Power'].mean())
    df['Engine'] = df['Engine'].fillna(df['Power'].mean())

    # converting year into age
    df['Year'] = 2021-df['Year']

    # encoding categorical features
    categorical_ = ['Company', 'Name', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']
    df = pd.get_dummies(df, columns= categorical_)

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    df = clean_dataset(df)
    return df
#%%
