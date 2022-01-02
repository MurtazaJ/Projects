import streamlit as st
st.set_page_config(
     page_title="Used Car Price Detection",
     page_icon="🚗",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
                }
                    )

st.markdown(
            """
                <style>
                .reportview-container {               
                    background: url("https://cdn.pixabay.com/photo/2021/01/01/21/09/challenger-5880009_960_720.jpg");
                    background-repeat: no-repeat;
                    background-size : 100%, 100%;
                }
                </style>
            """,
            unsafe_allow_html=True)

st.title('Lets find your right Price')
st.header('Want to sell your car or wish to buy a used car')

#Importing Files
import pickle
import numpy as np
import pandas as pd
from models import predict_price
df1 = pd.read_csv('data/train-data.csv')
with open('used_Car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Adding Company Feature Engineering
company = [i.split()[0] for i in df1['Name']]
df1.insert(0, 'Company', company)

# Chan ging names of cars with frequence <10 to others
df_name_cars_ = df1.Name.value_counts() 
name_stats_less_than_10_ =df_name_cars_[df_name_cars_<=10]
df1.Name = df1.Name.apply(lambda x: 'other' if x in name_stats_less_than_10_ else x)
df1.Name.to_list()


# Streamlit inputs
col1, col2, col3, col4 = st.columns(4)
year = col1.selectbox('Year', range(1991 ,2023))
kilo_driven = col2.number_input('Existing Kilometers', 0, 1000000)
mileage = col3.number_input('Mileage in kmpl', 0, 50)
engine = col4.number_input('Engine power in cc', 995, 6000)

col5, col6, col7, col8 = st.columns(4)
power = col5.number_input('Engine Power in bhp', 0, 100)
seats = col6.selectbox('What seater', range(0, 10))
location = col7.selectbox('Select yor City', df1.Location.unique())
fuel_type = col8.selectbox('Select your Fuel Type', df1.Fuel_Type.unique())

col9, col10, col11, col12 = st.columns(4)
transmission = col9.selectbox('Select your Fuel Type', df1.Transmission.unique())
owner_type = col10.selectbox('Select your Fuel Type', df1.Owner_Type.unique())
company_name = col11.selectbox('Select car Company', df1.Company.unique())
select_model = col12.selectbox('Select car model', df1.Name.unique())


#Scaling unscaled data with minmaxscaler
unscaled_data = pd.DataFrame(data = [[kilo_driven, mileage]], columns = ['Kilometers_driven', 'mileage'])
scaled_data = scaler.transform(unscaled_data)

# Fetching scaled data inputs
scaled_kilo_driven = scaled_data[0][0]
scaled_mileage = scaled_data[0][1]

# Checking the price
check_price = st.button('Check Price') 
if check_price == 1:
    price = predict_price(year, scaled_kilo_driven, 
                            scaled_mileage, engine, power, 
                            seats, company_name, select_model,
                            location, fuel_type, transmission, owner_type)
    
    st.subheader(f'The price of your car is {price:.2f} lacs')


st.write('')
st.markdown('''**Designed by Murtaza** :sunglasses:''')