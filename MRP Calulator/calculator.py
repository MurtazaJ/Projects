# Importing the Libraries
import numpy as np
import streamlit as st
import logging


logging.basicConfig(level=logging.INFO)
logging.getLogger('numexpr').setLevel(logging.WARNING)

# Defining the header
st.set_page_config(layout = 'wide')
st.header('MRP Calculator')

# Defining the ladies and Gents functions
def mrp_calculator_gents():
    GST        = st.sidebar.selectbox('', (5, 18))
    cp         = st.number_input('Please Enter the Cost Price', 0)

    gst_amt    = cp* (GST/100)

    expenses   = (cp + gst_amt) *(2/100)

    total      = cp + gst_amt + expenses

    if cp >= 600:
        if GST == 18:
            mrp     = total + cp + (total*.1) - expenses
        else:
            mrp = (total ) + cp - expenses - (2*gst_amt)
    else:
        if GST == 18:
            mrp     = total + cp + (total*.1) - expenses
        else:
            mrp = (total ) + cp 
    
    return mrp , gst_amt, expenses
       
def mrp_calculator_ladies():
    GST        = st.sidebar.selectbox('', (5, 18))
    cp         = st.number_input('Please Enter the Cost Price', 0)

    gst_amt    = cp* (GST/100)

    expenses   = (cp + gst_amt) *(2/100)

    total      = cp + gst_amt + expenses

    if GST == 18:
        mrp     = total + cp  - expenses 
    else:
        mrp = (total) + cp - expenses 
    
    return mrp , gst_amt, expenses

# Running the Model
st.sidebar.header('Choose Category')
selected_category = st.sidebar.selectbox('', ('Ladies & Kids', 'Gents'))
st.sidebar.subheader('GST')
if selected_category == 'Gents':
    m , g, e = mrp_calculator_gents()
else:
    m , g, e = mrp_calculator_ladies()

# Printing the Model
st.header(f'MRP:{m: .2f}')
st.write(f'Gst paid: Rs{g: .2f}           , Expenses occured:  Rs{e: .2f}')

st.write('')
st.markdown('''**Designed by Murtaza** :sunglasses:''')