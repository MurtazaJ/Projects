#%%
from models import best_model, get_score, grid_cv, plot_cv
from preprocessing import create_df, train_and_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns



df = create_df()
X_train, X_test, y_train, y_test = train_and_test(df)
# df2 = best_model(X_train, y_train)
# df_best = df2.to_csv(r'best_models.csv')
parameters = {  'bootstrap': True,
                'criterion': 'entropy',
                'max_depth':18,
                'max_features':'sqrt',
                'n_estimators':300
                }

print((X_train).shape,(y_train).shape, (X_test).shape, (y_test).shape)
results, ytest, pred = get_score(X_train, y_train, X_test, y_test, model = RandomForestClassifier(**parameters), scaler = StandardScaler())
                         