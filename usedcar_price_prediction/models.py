#%%
from preprocessing import preprocessing, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import pickle

pth_train = 'data/train-data.csv'
pth_test = 'data/test-data.csv'
df = preprocessing(pth_train)
X = df.drop(['Price'], axis = 1)
y = df.Price

scaler = MinMaxScaler()
# Passing through MinMaxScaler 
X[['Kilometers_Driven', 'Mileage']] = scaler.fit_transform(X[['Kilometers_Driven', 'Mileage']])     
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Finding the best model
dicts = {}
models = {'Lasso': Lasso(),
          'Ridge': Ridge() ,
          'KNeighbors Regression':KNeighborsRegressor(n_neighbors=2) 
         }
for k , v in models.items():
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    crss = cross_val_score(v, x_train, y_train, cv=cv)
    dicts[k] = crss.mean()
print(dicts)



# Finding the best model with parameters search using GridSearchCV
def find_best_model_using_gridsearchcv(X,y):
    models1 = {
            'KNeighborsRegressor' : {
                                    'model': KNeighborsRegressor(),
                                    'params': {
                                              'n_neighbors': [2,3,5,10]
                                                }
                                    },
            'lasso': {
                    'model': Lasso(),
                    'params': {
                                'alpha': [1,2],
                                'selection': ['random', 'cyclic']
                              }
                     },
            'decision_tree': {
                              'model': DecisionTreeRegressor(),
                              'params': {
                                        'criterion' : ['mse','friedman_mse'],
                                        'splitter': ['best','random']
                                        }
                            }
            }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for model_name, values in models1.items():
        gs =  GridSearchCV(values['model'], values['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': model_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


df_gs = find_best_model_using_gridsearchcv(x_train, y_train)


# Best model is KNeighborsRegressor
regr = KNeighborsRegressor(n_neighbors=3)
regr.fit(x_train, y_train)


# Checking Score
print("mean squared error: %.2f" % np.mean((regr.predict(x_test) - y_test) ** 2))
print("variance score: %.2f" % regr.score(x_test, y_test) )


#Saving file to Pickle
with open('used_car_price_model.pkl', 'wb') as f:
    pickle.dump(regr, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
## Predict model
def predict_price(year, kilometers_driven, mileage, 
                  engine, power, seats, company, name, 
                  location, fuel_type, transmission, owner_type):
    company_index = np.where(X.columns==company)[0]
    name_index = np.where(X.columns==name)[0]
    location_index = np.where(X.columns==location)[0]
    fuel_type_index = np.where(X.columns==fuel_type)[0]
    transmission_index = np.where(X.columns==transmission)[0]
    owner_type_index = np.where(X.columns==owner_type)[0]
    x2 = np.zeros(len(X.columns))
    x2[0] = year
    x2[1] = kilometers_driven
    x2[2] = mileage
    x2[3] = engine
    x2[4] = power
    x2[5] = seats
    if company_index >= 0:
        x2[company_index] = 1
    if name_index >= 0:
        x2[name_index] = 1
    if location_index >= 0:
        x2[location_index] = 1
    if fuel_type_index >= 0:
        x2[fuel_type_index] = 1
    if transmission_index >= 0:
        x2[transmission_index] = 1
    if owner_type_index >= 0:
        x2[owner_type_index] = 1

    result = regr.predict([x2])
    return result[0]

