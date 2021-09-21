import pandas                as pd
import numpy                 as np
import matplotlib.pyplot     as plt
import seaborn               as sns
import a2_columnenhancement  as ce
import a3_rowenhancement     as rw
import a4_finding_model      as fm
import pickle                as pk 
from sklearn.metrics         import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
np.random.seed(0)



df_test = pd.read_csv('DailyDelhiClimateTest.csv')
df_train = pd.read_csv('DailyDelhiClimateTrain.csv')
df_train_c = pd.read_csv('DailyDelhiClimateTrain.csv')
df_test_c = pd.read_csv('DailyDelhiClimateTest.csv')


print('Original Shape')
print(df_test.shape)
print(df_train.shape)


##Column Enhancement
df_train, df_test = ce.date_split(df_train, df_test)
print('column enhanced Shape')
print(df_train.shape)
print(df_test.shape)



##Either do row enhancement or row pairing
#Row pairing
# x, y= rw.pairing(df_train)
# print (x.shape, y.shape)
# print(x)


# Row enhancement
x= rw.data_enhancement(df_train,40)
print('row enhanced shape on df_train')
print(x.shape)


# Train test Split
X = x.drop(['meantemp'], axis=1)
y = x['meantemp']
print('X.shape: ', X.shape)
print('y.shape: ', y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=0, test_size=.25)
print('X_train.shape,y_train.shape, X_test.shape, y_test.shape')
print(X_train.shape,y_train.shape, X_test.shape, y_test.shape)


#Finding the Best model and saving it in the best model
Results = fm.model_analysis(x, X_train, y_train, X_test,  y_test)

#loading the model from Pickle
with open('model_pickle', 'rb') as f:
    mp = pk.load(f)

#Testing df_test
x_test = df_test.drop(['meantemp'], axis=1)
y_test = df_test['meantemp']

# Using pickle model to predict
y_pred = mp.predict(x_test)
new_df = pd.DataFrame(y_pred)
print(new_df)
mse_best_model = mean_squared_error(y_test, y_pred)*100
print(mse_best_model)


# Plottoing figure with test values and predicted values
fig, ax = plt.subplots(figsize=(15,8))
chart=sns.lineplot(x=df_train_c['date'], y='meantemp', data = df_train_c)
chart.set_title('Delhi Climate')
chart = sns.lineplot(x=df_test_c['date'], y = 'meantemp', data=df_test)
chart = sns.lineplot(x=df_test_c['date'], y = 0,  data=new_df)
plt.legend(labels = ['train_data', 'test_data', 'predicted_data'])
plt.show()


