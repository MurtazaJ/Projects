import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
xgboost.set_config.verbose = False

def best_model(x,y):
    models = [LogisticRegression(), SVC(), GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), XGBRFClassifier(eval_metric='mlogloss',use_label_encoder =False), LGBMClassifier()]
    model_names = ['logisitc reg', 'svm', 'naive bayes', 'knn', 'decision tree', 'random forest', 'xgboost', 'lgbm']
    scalers = [None, StandardScaler(), RobustScaler(), MinMaxScaler()]
    scaler_names = ['none', 'std', 'robust', 'min-max']
    scores = [[] for _ in range(4)]
    iterr = 0
    for model in models:
        for index_scaler, scaler in enumerate(scalers):
            iterr += 1
            print(f"iterations: ==> {iterr} / {len(models) * len(scalers)} ...")
            if scaler:
                model = Pipeline(steps = [('scaler', scaler), ('model', model)])
                skf = StratifiedKFold(5, shuffle = True)
                score = cross_val_score(model, cv = skf, X=x, y = y, scoring ='f1').mean()
                scores[index_scaler].append(score)
    return pd.DataFrame(scores, index = scaler_names, columns = model_names).T

def get_score(xt, yt, xtest, ytest, model, scaler=None):
    if scaler:
        model = Pipeline(steps = [('scaler', scaler), ('model', model)])
    
    model.fit(xt,yt)
    pred = model.predict(xtest)
    print('Report'.center(70, '='))
    print()
    print(f"Training score ==> {model.score(xt, yt)}")
    print(f"Training score ==> {model.score(xt, yt)}")
    print()
    print(classification_report(ytest,pred))
    print()
    sns.heatmap(confusion_matrix(ytest, pred), fmt ='.1f', annot=True)

def grid_cv(xt, yt, model, params, scaler = None):
    if scaler:
        model = Pipeline(steps = [('scaler', scaler), ('model', model)])
    skf = StratifiedKFold(5, shuffle = True)
    clf = GridSearchCV(model, param_grid = params, cv = skf, return_train_score = True)
    clf.fit(xt, yt)
    res = pd.DataFrame(clf.cv_results_).sort_values('mean_test_score', ascending = False)
    return clf.best_estimator_, clf.best_params_, res[['mean_train_score', 'mean_test_score', 'params']]


def plot_cv(result):
    sns.lineplot(x=result.reset_index().index, y = result.mean_train_score)
    sns.lineplot(x=result.reset_index().index, y = result.mean_test_score)
    plt.legend(['train_score', 'test_score'])
    plt.title('F1_Score')