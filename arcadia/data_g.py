#%%
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
import os

#%%
print(os.getcwd())

#%%

col = ["X", "Y", "Z", "Pressure", "GripAngle", "Timestamp", "Test ID", "Class"]

control_file = glob.glob("control/*")
df_c = pd.DataFrame()
for files in control_file:
    data = pd.read_csv(files, delimiter=";", names=col)
    data["Class"] = 0
    data['Patient_id'] = files.split('_')[1].strip('.txt')
    df_c= pd.concat([df_c, data])
    
control_file1 = glob.glob("parkinson/*")
df_p = pd.DataFrame()
for files in control_file1:
    # print(files)
    data1 = pd.read_csv(files, delimiter=";", names=col)
    data1["Class"] = 1
    data1['Patient_id'] = files.split('_')[1].strip('.txt')
    df_p= pd.concat([df_p, data1])
    

# %%
#printing shapes


print(df_c.shape)
print(df_p.shape)



# %%

#concatanating dataframe
df = pd.concat([df_c, df_p])
# print(df)


# %%
# df['Patient_id'].unique()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Patient_Id'] = le.fit_transform(df['Patient_id'])
df['Patient_Id'].unique()
df.drop(['Test ID', 'Patient_id', 'Timestamp'], axis= 1, inplace= True)
print(df)



# %%
def pairing(df, seq_length=400):
    a = []
    b = []
    n = df['Patient_Id'].unique()
    for x in range(len(n)):
        m= df.loc[df['Patient_Id']==n[x], :]
        # print(m)
        for i in range(0, (m.shape[0] - (seq_length+1)), seq_length+1):
            seq = np.zeros((seq_length, m.shape[1]))
            for j in range(seq_length):
                seq[j] = m.values[i+j]
            # seq= np.mean(seq, axis=0)   
            # print(seq.shape)
            a.append(seq.flatten())
            b.append(m['Class'][i + seq_length] )

    return np.array(a), np.array(b)

X,y = pairing(df)
print(X.shape)
print(y.shape)


# import pickle as pk
# with open(saved_model, 'wb') as file:
#     pk.dump((X,y), file)

#%%
# Train test and split
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, random_state=0 , stratify = y, test_size= 0.2)


#%%
# Training on machine learning modules

from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier
from sklearn.pipeline      import Pipeline
from time import time
from sklearn               import metrics

tree_classifiers = {
"Decision Tree": DecisionTreeClassifier(),
"Extra Trees":   ExtraTreesClassifier(n_estimators=100),
"Random Forest": RandomForestClassifier(n_estimators=100),
"AdaBoost":      AdaBoostClassifier(n_estimators=100),
"Skl GBM":       GradientBoostingClassifier(n_estimators=100),
"Skl HistGBM":   HistGradientBoostingClassifier(max_iter=100),
# "XGBoost":       XGBClassifier(n_estimators=100, use_label_encoder=False),
"LightGBM":      LGBMClassifier(n_estimators=100),
"CatBoost":      CatBoostClassifier(n_estimators=100, verbose=False)
}

results = pd.DataFrame({'Model': [], 'Accuracy': [],  'Time': []})

for model_name, model in tree_classifiers.items():
        
    pipe = Pipeline([ ('classifier', model)])

    start_time = time()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    total_time = time() - start_time

    results = results.append({
                            "Model":    model_name,
                            "Accuracy": metrics.accuracy_score(y_test, y_pred)*100,
                            "Time":     total_time
                            },
                            ignore_index=True)       

print(results)

