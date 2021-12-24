
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from preprocessing import create_df, train_and_test
from models import best_model, get_score, grid_cv
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np


#create a Dataframe
df = create_df()
x,y,X_train, X_test, y_train, y_test = train_and_test(df)

#Streamlit initialisations
import streamlit as st
st.set_page_config(layout = 'wide')
st.title('Classification of Parkinson disease')


#Download the file
df = create_df()
st.subheader('--------------------------------------------------------------------')
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Parkinson_disease.csv">Download CSV File</a>'
    return href
st.subheader('Please the link below to download the CSV file')


# st.markdown(filedownload(df), unsafe_allow_html=True)
st.download_button('Download csv file', filedownload(df))
st.subheader('--------------------------------------------------------------------')

# Sidebar for defining the height and width of the graphs
st.sidebar.subheader('Please select a size of the plots with this slidebar')
width = st.sidebar.slider("Plot width", 0.5, 8., 3.)
height = st.sidebar.slider("Plot height", 0.5, 8., 3.)


#displaying the dataframe
#%%
selected_columns = df.drop('name', axis = 1).columns.tolist()

#%%
st.subheader('Select the features to view the dataframe')
selected_columns_st = st.multiselect('', selected_columns,selected_columns)
df_altered = df[(selected_columns_st)]
st.dataframe(df_altered)
st.header('')

# Heatmap
if st.button('close'):
    m = st.button('Intercorrelation Heatmap')
else:
    m = st.button('Intercorrelation Heatmap')
    if m:
        st.header('Intercorrelation Matrix Heatmap: press close to close the figure')
        corr = df_altered.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True    
        fig, ax = plt.subplots(figsize=(width, height))
        ax = sns.heatmap(corr, square=True)
        st.pyplot(fig)


#Plotting the scatter plots
x_axis = st.sidebar.selectbox('Choose X-axis to plot from the features' ,(x.columns))
y_axis = st.sidebar.selectbox('Choose y-axis to plot from the features' ,(x.columns))
x_axis_df = x[(x_axis)]
y_axis_df = x[(y_axis)]
fig, ax = plt.subplots(figsize=(width, height))
ax.scatter( x_axis_df, y_axis_df, c = y )
buf = BytesIO()
fig.savefig(buf, format="png")
st.image(buf)




# df1 = best_model(X_train, y_train)
 
# %%
