from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from preprocessing import create_df, train_and_test
from models import best_model, get_score, grid_cv
df = create_df()
x,y,X_train, X_test, y_train, y_test = train_and_test(df)

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
print(results)
