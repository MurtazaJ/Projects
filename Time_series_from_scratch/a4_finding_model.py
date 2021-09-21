import pandas                as pd
import numpy                 as np
import matplotlib.pyplot     as plt
import seaborn               as sns
import a2_columnenhancement  as ce
import a3_rowenhancement     as rw
import time
import math
import pickle                as pk 
from sklearn.model_selection import train_test_split
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.ensemble        import ExtraTreesRegressor
from sklearn.ensemble        import AdaBoostRegressor
from sklearn.ensemble        import GradientBoostingRegressor
from sklearn.experimental    import enable_hist_gradient_boosting # Necesary for HistGradientBoostingregressorRegressor
from sklearn.ensemble        import HistGradientBoostingRegressor
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor
from catboost                import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics         import r2_score,mean_squared_error
from sklearn.pipeline        import Pipeline, make_pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.compose         import ColumnTransformer





def model_analysis(x, X_train, y_train, X_test,  y_test):
   pipe1 = Pipeline( [ ('scaling', StandardScaler() ) ] )
   preparation = ColumnTransformer(transformers=['all columns', pipe1, x.columns.values.tolist() ] ) 

   
   classifiers = {"Decision Tree": DecisionTreeRegressor(),
                  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
                  "Random Forest": RandomForestRegressor(n_estimators=100),
                  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
                  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
                  "Skl HistGBM":   HistGradientBoostingRegressor(max_iter=100),
                  "XGBoost":       XGBRegressor(n_estimators=100, use_label_encoder=False),
                  "LightGBM":      LGBMRegressor(n_estimators=100),
                  "CatBoost":      CatBoostRegressor(n_estimators=100, verbose=False) }

   
     
   results = pd.DataFrame({'Model': [], 'MSE': [], 'RMSE': [], 'Time': []})

   for model_name, model in classifiers.items():

    #    main_pipe = Pipeline( [ ('preparation', preparation), 
    #                            ('model_name', model)
    #                          ] )
    #    print((main_pipe))
       start_time = time.time()

       model.fit(X_train, y_train)
       pred = model.predict(X_test)
       total_time = time.time() - start_time
       results = results.append( {'Model': model_name,
                                  'MSE':   mean_squared_error(y_test, pred)*100,
                                  'RMSE':  math.sqrt(mean_squared_error(y_test, pred)*100),
                                  'Time':  total_time}, ignore_index=True )
   results_ord = results.sort_values(by=['MSE'], ascending=True, ignore_index=True)
   results_ord.index +=1
   results_ord.style.bar(subset=['MSE', 'RMSE'], vmin=0, vmax=100, color='#5fba7d')
   print((results_ord))

   # Save the best model 
   best_model = classifiers[results_ord.iloc[0,0]]
   best_model.fit(X_train, y_train)

   with open('model_pickle', 'wb') as f:
        pk.dump(best_model, f)
   return results_ord