#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

#%%
df = pd.read_csv('data/export_dataframe.csv')
# df = df.drop('time_stamp', axis = 1)
df = df.head(10000)
df
#%%
df_rating_matrix = df.pivot_table(values='rating', index = 'user_id', columns='product_id', fill_value=0)

#%%
df_rating_matrix.head()
# %%
df_rating_matrix.shape
# %%
df_T = df_rating_matrix.T
df_T.head()
df_T.shape

#%%
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(df_T)
decomposed_matrix.shape
# %%
corealtion_matrix = np.corrcoef(decomposed_matrix)
corealtion_matrix

#%%
# Finding a product
index_search = df_T.index[99]
# %%
i = index_search
product_names = list(df_T.index)
product_ID = product_names.index(i)
product_ID
# %%
corealtion_product_ID = corealtion_matrix[product_ID]
corealtion_product_ID.shape

#%% 
# Recommending the most correlated product
recommendation = list(df_T.index[corealtion_product_ID > 0.90])
recommendation.remove(i) #Removing the already bought product
recommendation[0:5]

