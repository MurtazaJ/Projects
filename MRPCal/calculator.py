# # Importing the Libraries
# import streamlit as st
# from PIL import Image
# import os
# current_dir = os.getcwd()
# st.text(current_dir)
# ch_img_path = os.path.join(current_dir, "CHILDREN.png")
# ld_img_path = os.path.join(current_dir, "LADIES.png")
# jt_img_path = os.path.join(current_dir, "GENTS.png")
# # logging.basicConfig(level=logging.INFO)
# # logging.getLogger('numexpr').setLevel(logging.WARNING)

# # Defining the header
# # st.set_page_config(layout = 'wide')
# st.header('MRP Calculator')
# #nothingC:\Users\Murtaza\Documents\GIT\Projects\MRPCal\children.png

# #Defining rounding functions
# def round_to_nearest(n):
#     if n ==0:
#         return 0
#     else:
#         r = n % 10
    
#         return (n + 10 - r if r + r >= 10 else n - r)-1

# # Defining the ladies and Gents functions    
# def mrp_calculator_gents():
#     cp = st.number_input('Please Enter the Cost Price', 0)

#     if cp <= 999:
#         GST = 12
#     else:
#         GST = 18

#     gst_amt = cp * (GST / 100)

#     expenses = (cp + gst_amt) * (2 / 100)

#     total = cp + expenses

#     if GST == 18:
#         mrp = int(total + cp)
#         mrp = round_to_nearest(mrp)
#     else:
#         mrp = int(total + cp + (total * 0.19))
#         mrp = round_to_nearest(mrp)

#     return mrp, gst_amt, expenses

       
# def mrp_calculator_ladies():
#     cp = st.number_input('Please Enter the Cost Price', 0)

#     if cp <= 999:
#         GST = 12
#     else:
#         GST = 18

#     gst_amt = cp * (GST / 100)

#     expenses = (cp + gst_amt) * (2 / 100)

#     total = cp + gst_amt + expenses

#     if GST == 18:
#         mrp = int(total + cp)
#         mrp = round_to_nearest(mrp)
#     else:
#         mrp = int(total + cp + (2.5/100 * cp) + gst_amt)
#         mrp = round_to_nearest(mrp)

#     return mrp, gst_amt, expenses


# def mrp_calculator_children():
#     cp = st.number_input('Please Enter the Cost Price', 0)
    
#     if cp <= 999:
#         GST = 12
#     else:
#         GST = 18

#     gst_amt = cp * (GST / 100)

#     expenses = (cp + gst_amt) * (2 / 100)

#     total = cp + cp*(10/100) + gst_amt + expenses

#     mrp = int(total + cp)
#     mrp = round_to_nearest(mrp)

#     return mrp, gst_amt, expenses

# # Running the Model
# st.header('Choose Category')
# selected_category = st.selectbox('', ('Ladies', 'Gents', 'Children'))

# if selected_category == 'Gents':
#     st.image(jt_img_path, caption="Gents Categories", use_column_width=True)
#     m , g, e = mrp_calculator_gents()
    

# elif selected_category == 'Ladies':
#     st.image(ld_img_path, caption="Ladies Categories", use_column_width=True)
#     m , g, e = mrp_calculator_ladies()
   

# else:
#     st.image(ch_img_path, caption="Children Categories", use_column_width=True)
#     m , g, e = mrp_calculator_children()
    

# # Printing the Model
# st.title(f'MRP:{m: .2f}')
# st.write(f'Gst paid: Rs{g: .2f}           , Expenses occured:  Rs{e: .2f}')

# st.write('')

# st.markdown('''**Designed by Murtaza** :sunglasses:''')

# Importing the Libraries
import streamlit as st
from PIL import Image
import os

# Set up paths for images
current_dir = os.getcwd()
ch_img_path = os.path.join(current_dir, "MRPCal\CHILDREN.png")
ld_img_path = os.path.join(current_dir, "MRPCal\LADIES.png")
jt_img_path = os.path.join(current_dir, "MRPCal\GENTS.png")

# Defining the header
st.header('MRP Calculator')
st.header(current_dir)
# Rounding function
def round_to_nearest(n):
    if n == 0:
        return 0
    r = n % 10
    return (n + 10 - r if r >= 5 else n - r) - 1

# General MRP Calculator
def mrp_calculator(category):
    cp = st.number_input('Please Enter the Cost Price', min_value=0, step=1)

    # Determine GST rate
    GST = 12 if cp <= 999 else 18
    gst_amt = cp * (GST / 100)

    # Common expense calculation
    expenses = (cp + gst_amt) * 0.02

    if category == 'Gents':
        total = cp + expenses
        if GST == 18:
            mrp = round_to_nearest(int(total + cp))
        else:
            mrp = round_to_nearest(int(total + cp + (total * 0.19)))

    elif category == 'Ladies':
        total = cp + gst_amt + expenses
        if GST == 18:
            mrp = round_to_nearest(int(total + cp))
        else:
            mrp = round_to_nearest(int(total + cp + (0.025 * cp) + gst_amt))

    elif category == 'Children':
        total = cp + (0.1 * cp) + gst_amt + expenses
        mrp = round_to_nearest(int(total + cp))

    return mrp, gst_amt, expenses

# Running the Model
st.header('Choose Category')
selected_category = st.selectbox('', ('Ladies', 'Gents', 'Children'))

# Display appropriate image and calculate MRP
if selected_category == 'Gents':
    st.image(jt_img_path, caption="Gents Categories", use_column_width=True)
elif selected_category == 'Ladies':
    st.image(ld_img_path, caption="Ladies Categories", use_column_width=True)
else:
    st.image(ch_img_path, caption="Children Categories", use_column_width=True)

# Calculate MRP
m, g, e = mrp_calculator(selected_category)

# Display the results
st.title(f'MRP: {m:,.2f}')
st.write(f'GST paid: ₹{g:,.2f}  |  Expenses incurred: ₹{e:,.2f}')

# Footer
st.markdown('''**Designed by Murtaza** :sunglasses:''')
