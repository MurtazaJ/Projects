# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns
import time
from IPython.display import clear_output
import numpy    as np
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

# Modeling and Prediction 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression 
from sklearn.ensemble      import RandomForestClassifier 
from sklearn.tree          import DecisionTreeClassifier 
from sklearn.neighbors     import KNeighborsClassifier 
from sklearn.svm           import SVC
from sklearn.metrics       import confusion_matrix
from sklearn.metrics       import accuracy_score
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier
# from models_pred import models

os.getcwd()
# %%
df= pd.read_csv('heart.csv')
print(df)
print(df.shape)
print(df.isna().sum())

# Remove Duplicate rows
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape)
print(duplicate_rows)
df = df.drop_duplicates() #drops duplicated rows
duplicate_rows = df[df.duplicated()]
print("Number of duplicate rows :: ", duplicate_rows.shape)


# Plot the heat_map
sns.set(font_scale = 0.9)
plt.figure(figsize= (15,6))
sns.heatmap(df.corr(),annot= True)
plt.show()


# Plot comparisoon graphs
sns.distplot(df['age'], kde=False).set(title='Age distibuted in the dataset')
sns.set(font_scale = 2)
for x in df.columns:
    sns.FacetGrid(df,col='output',height=7).map(plt.hist, x)
    plt.show()


# Splitting the data
X = df.drop(['output'], axis = 1)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101, stratify= y)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# Model on which the testing will be done
tree_classifiers = {
  "Decision Tree": DecisionTreeClassifier(),
  "Extra Trees":   ExtraTreesClassifier(n_estimators=100),
  "Random Forest": RandomForestClassifier(n_estimators=100),
  "AdaBoost":      AdaBoostClassifier(n_estimators=100),
  "Skl GBM":       GradientBoostingClassifier(n_estimators=100),
  "Skl HistGBM":   HistGradientBoostingClassifier(max_iter=100),
  "XGBoost":       XGBClassifier(n_estimators=100),
  "LightGBM":      LGBMClassifier(n_estimators=100),
  "CatBoost":      CatBoostClassifier(n_estimators=100),
  'KNN neighbors': KNeighborsClassifier(n_neighbors = 3),
  'Logistic Regression' : LogisticRegression(),
  'SVM' :SVC(kernel='linear', C=1, random_state=101)
} 

#  Find the best model
skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

for model_name, model in tree_classifiers.items():

    start_time = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    total_time = time.time() - start_time
            
    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_test, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_test, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)

results_ord = results.sort_values(by=['Accuracy'], ascending=False, ignore_index=True)
results_ord.index += 1 
print(results_ord)

print('So Random forest is the best Classifier, so we choose it')

 
import pickle
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)



