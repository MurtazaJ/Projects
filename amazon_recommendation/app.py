import streamlit as st
from popularity_based_recommendation import popularity_based_search
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
st.set_page_config(layout="wide")


st.title('Amazon Recommendation system')
st.header('Product popularity based recommendation')
@st.cache
def load_data():
    df = pd.read_csv('amazon_recommendation/data/export_dataframe.csv')
    df.rating = df.rating.astype(int)
    return df

df = load_data()
df4 = df.head(10000)
st.header('Lets look at the data')
st.write(df.head(5))
mean_rating = df['rating'].mean()
st.header('Ratings done for products in descending')
rating_count = df.groupby('product_id')['rating'].sum().sort_values(ascending=False)
st.write(rating_count.head(5))


st.header('Ratings done by user for more than 50 products')
no_of_rated_products_per_user = df.groupby('user_id')['rating'].count().sort_values(ascending=False)
st.write('no of rated product more than 50 per user: {}'.format(sum(no_of_rated_products_per_user>=50)))

st.header('Products having more than 50 ratings in descending order')

@st.cache
def new_df1(df):
    new_df = df.groupby('product_id').filter(lambda x:x['rating'].count()>=50)
    return new_df
new_df = new_df1(df)

rating_mean_count = pd.DataFrame(new_df.groupby('product_id')['rating'].mean())
rating_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('product_id')['rating'].count())
rating_mean_count.sort_values(by='rating_counts', ascending=False)
most_popular = rating_mean_count.sort_values('rating_counts', ascending = False)

st.header('Most Popular products')
st.write(most_popular.head())

min_rating = most_popular['rating_counts'].quantile(0.90)
products_greater_than_min_rating = most_popular.loc[most_popular['rating_counts'] >= min_rating]

def weighterd_rating(x, min_rating = min_rating, mean_rating = mean_rating):
    voters = x['rating_counts']
    avg_vote = x['rating']
    # Calculation based on formula
    return (voters/(voters + min_rating) * avg_vote) + (min_rating/(voters + min_rating) * mean_rating)

latext = r'''
## Self defined formula for calulating most recommended products to users

### Full equation 
$$
\frac{total \ number \ of \ voters}{(total \ number \ of \ voters + min \ rating) \ast avg \ vote}  + \frac{min \ rating}{(total \ number \ of \ voters + min \ rating) \ast mean \ rating)}
$$
 
'''
st.write(latext)

products_greater_than_min_rating['score'] = products_greater_than_min_rating.apply(weighterd_rating, axis = 1)
products_greater_than_min_rating.rename(columns={'rating': 'mean_rating'}, inplace=True)

st.header('Products displayed to users based on recommendation')
st.write(products_greater_than_min_rating.sort_values('score', ascending=False))







st.title('----------------------------------------------------------------------------------------------------')
st.title('Item based Recommendation (Colloborative Filetering)')


@st.cache
def df_rating_matrix1(df):
    df_rating_matrix = df.pivot_table(values='rating', index = 'user_id', columns='product_id', fill_value=0)
    return df_rating_matrix
df_rating_matrix = df_rating_matrix1(df4)


df_T = df_rating_matrix.T

st.header('Lets look at the previous data in a form that we have all the users in the index who have rated for the products in the columns')
st.dataframe(df_rating_matrix.head(10).style.highlight_max(axis=1))
st.header('')
st.write('')
st.subheader('Now we find the corelation with machine learning algorithms based on the products that were frequently bought along with the selected product by our users')





SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(df_T)
corealtion_matrix = np.corrcoef(decomposed_matrix)
# with st.echo():



st.write(corealtion_matrix[0:10,:])


st.header('')
st.write('')
st.title('Pick an index number to search for corelated products')
index_search = st.number_input('', 1, 10000, 30)
index_search = df_T.index[index_search]
i = index_search
product_names = list(df_T.index)
product_ID = product_names.index(i)

corealtion_product_ID = corealtion_matrix[product_ID]
recommendation = list(df_T.index[corealtion_product_ID > 0.90])
recommendation.remove(i) #Removing the already bought product
st.header('Most corelated products to the searched products are:')
st.write(recommendation[0:9])


st.title('----------------------------------------------------------------------------------------------------')
st.title('So the question is how do we apply this method if we are starting a new business? We have no previous data from the users.')

product_descriptions = pd.read_csv('amazon_recommendation/data/product_descriptions.csv.zip')
st.subheader('Lets look at the data')
st.dataframe(product_descriptions.head(10))
product_descriptions = product_descriptions.dropna()
product_descriptions1 = product_descriptions.head(500)

st.subheader('First we read the text with NLP and convert each word into a respective number and then we cluster them in groups')
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X1)

def print_cluster(i):
    st.write("Cluster %d:" % i)
    for ind in order_centroids[i, :5]:
        st.write(terms[ind])

true_k = 20
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X1)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print_cluster(i)

#%%
def show_recommendations(product):
    #print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    print_cluster(prediction[0])

#%%
item_search =st.text_input('Search your keyword', 'paint')
show_recommendations(item_search)
