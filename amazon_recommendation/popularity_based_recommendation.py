#%%
import pandas as pd
pd.set_option('precision', 2)


#%%
df = pd.read_csv('data/export_dataframe.csv')

#Recommendation based on product rating
df.head()


# %%
mean_rating = df['rating'].mean()
mean_rating
# %%
# Rating per product
rating_count = df.groupby('product_id')['rating'].sum().sort_values(ascending=False)
rating_count.head(10)
# %%
# Rating per user
no_of_rated_products_per_user = df.groupby('user_id')['rating'].count().sort_values(ascending=False)
no_of_rated_products_per_user.head()
# %%
print('no of rated product more than 50 per user: {}'.format(sum(no_of_rated_products_per_user>=50)))
# %%
new_df = df.groupby('product_id').filter(lambda x:x['rating'].count()>=50)

# %%
new_df
# %%
rating_mean_count = pd.DataFrame(new_df.groupby('product_id')['rating'].mean())
rating_mean_count
# %%
rating_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('product_id')['rating'].count())

# %%
rating_mean_count.sort_values(by='rating_counts', ascending=False)
# %%
most_popular = rating_mean_count.sort_values('rating_counts', ascending = False)
most_popular.head()
# %%
min_rating = most_popular['rating_counts'].quantile(0.90)
min_rating
# %%
products_greater_than_min_rating = most_popular.loc[most_popular['rating_counts'] >= min_rating]
# %%
products_greater_than_min_rating
# %%
def weighterd_rating(x, min_rating = min_rating, mean_rating = mean_rating):
    voters = x['rating_counts']
    avg_vote = x['rating']
    # Calculation based on formula
    return (voters/(voters + min_rating) * avg_vote) + (min_rating/(voters + min_rating) * mean_rating)
# %%
products_greater_than_min_rating['score'] = products_greater_than_min_rating.apply(weighterd_rating, axis = 1)

# %%
products_greater_than_min_rating.rename(columns={'rating': 'mean_rating'}, inplace=True)
# %%
products_greater_than_min_rating.sort_values('score', ascending=False)