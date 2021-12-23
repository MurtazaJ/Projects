#%%
from models import best_model, get_score, grid_cv, plot_cv
from preprocessing import create_df, train_and_test
from utils import permutation_importance, mi_score, plot_mi
import seaborn as sns
import pandas as pd


df = create_df()
x,y,X_train, X_test, y_train, y_test = train_and_test(df)
#%%
def graphs (x,y,df):
    sns.countplot(x=df['status'])
    sns.heatmap(x.corr())
    corrwith = pd.DataFrame(x.corrwith(y), columns = ['status']).sort_values('status', ascending = False)
    print(corrwith[corrwith.status > 0.5])
    permutation_result = permutation_importance(x,y)
    print(permutation_result)
    mscore = mi_score(x,y, True)
    print(mscore)

    # co = ['PPE', 'spread1', 'MDVP:Fo(Hz)', 'spread2', 'MDVP:APQ']

    sns.scatterplot(x=df['Shimmer:DDA'], y=df['MDVP:Shimmer'], hue = df['status'])
    # sns.scatterplot(x=df.c[0], y=df.spread1, hue = df['status'])
    
    # sns.scatterplot(x=df.c[1], y=df.c[3], hue = df['status'])
    # sns.scatterplot(x=c[1], y=c[2], hue = df['status'])
    # sns.scatterplot(x=c[1], y=c[4], hue = df['status'])
    # sns.scatterplot(x=c[3], y=c[4], hue = df['status'])
    # sns.scatterplot(x=c[3], y=c[2], hue = df['status'])
    
 
# %%
print(graphs (x,y,df))
# %%
sns.scatterplot(df['MDVP:Fo(Hz)'], df['spread2'], hue = df['status'])
# %%
co = ['PPE', 'spread1', 'MDVP:Fo(Hz)', 'spread2', 'MDVP:APQ']
sns.scatterplot(x=df.co[1], y=df.co[3], hue = df['status'])
#%%
co[0]
# %%
