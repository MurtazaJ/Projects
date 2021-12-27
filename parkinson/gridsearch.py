from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from app import create_df, train_and_test
from models import best_model, get_score, grid_cv
import pandas as pd
import csv
df = create_df()
X_train, X_test, y_train, y_test = train_and_test(df)

params = {
        'n_estimators': [100,200,300], 
        'criterion': ['gini','entropy'],
        "max_depth" : [8 , 12, 15, 20, 25, 30], 
        'class_weight' : [None, 'balanced'],
        'bootstrap' : [True, False],
        'max_features' : ["auto", "sqrt", "log2"]
        }
pipline_params = {f"model__{key}" : value for key, value in params.items()}
clf, best_params, results = grid_cv(X_train, y_train, RandomForestClassifier(), pipline_params, scaler = MinMaxScaler())

file = open('best_params.csv', 'w')
writer = csv.writer(file)
for key, value in best_params.items():
        writer.writerow([key,value])
file.close()


