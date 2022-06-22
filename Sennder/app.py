# importing the libraries
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image



st.set_page_config(
     page_title="Sennder Shipper Analysis App",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df= pd.read_excel(uploaded_file, header = 1 )
    st.write('file loading sucessful...')
    df = df.sort_values(by='Lead Acquisition Date',ascending=True) # Sorting the Dataframe by Date
    df['Month'] = df['Lead Acquisition Date'].dt.month_name() # Adding a month Column
    df = df.astype({'T0 Carrier':'int', 
                    'T1 Carrier':'int', 
                    'Carrier Launched':'int', 
                    'T2 Carrier':'int', 
                    '1st Completed Load on Orcas':'int', 
                    'T3 Carrier':'int'}) # Cahnging values to int
    
    # Viewing the desired columns
    selected_columns = df.columns.tolist() 
    
    st.header('Select Lead source and Stages in our Onboarding funnel')

    selected_columns_st = st.multiselect('', selected_columns,selected_columns)
    st.title('')
    if len(selected_columns_st) > 0:
        
        df_altered = df[(selected_columns_st)]
        col1 , col2 = st.columns(2)
        start_date = col1.multiselect('Choose Month', df.Month.unique(), default = (df.Month.unique()))
        st.title('')
        selected_lead = col2.multiselect('Choose a Lead Source',df['Lead Source'].unique(), default = (df['Lead Source'].unique()))
        df_altered = df_altered.loc[df_altered["Month"].isin(start_date) & (df_altered["Lead Source"].isin(selected_lead))]
        st.dataframe(df_altered.head(5))
        st.title('')
        st.title('')
        # Total leads barplot
        if st.button('Reset')==1:
            col3 , col4  = st.columns(2)
            st.title('')
            col6, col7 = st.columns(2)
            new_df = df_altered.groupby(['Lead Source'])['T0 Carrier'].sum() # grouping by Lead Source and summing the T0 Carrier
            col3.title('')
            col3.title('')
            col3.title('')
            col3.table(new_df) # displaying the dataframe
            col3.subheader(f" Sum of Total leads in T0 Carrier {new_df.sum()}") # displaying the total sum
    
            # Barplot for T0 Carrier
            # fig1, ax = plt.subplots(figsize=(8,3))
            # sns.barplot(x= new_df.index, y = new_df)
            # ax.set_xlabel("Lead Source")
            # ax.set_ylabel("Count", rotation=90)
            # ax.tick_params(axis='x', rotation=45)
            # ax.legend()
            # col4.write(fig1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x= new_df.index, y = new_df,mode='lines', line_width=3, name='Maximum Lead count'))
            fig.add_trace(go.Bar(x= new_df.index, y = new_df, name='Lead Count'))
            col4.write(fig)
            col3.title('')
            col3.title('')
            col3.title('')
            image = Image.open('incorrect_data.png')
            col3.image(image, caption='Incorrect data')

            # Scatter Plot
            fig2, ax2 = plt.subplots(figsize=(8,3))
            sns.scatterplot(x =df_altered['Lead Source'], y=df_altered['T0 Carrier'])
            ax2.tick_params(axis='x', rotation=45)
            col4.write(fig2)
        
            # Calculating Conversions
            df_stage0 =  df_altered['T0 Carrier'].sum() /  df_altered['T0 Carrier'].sum()
            df_stage1 =  df_altered['T1 Carrier'].sum() /  df_altered['T0 Carrier'].sum()
            df_stage2 =  df_altered['T2 Carrier'].sum() / df_altered['T1 Carrier'].sum() 
            df_stage3 =  df_altered['Carrier Launched'].sum() /df_altered['T2 Carrier'].sum()  
            df_stage4 =  df_altered['T3 Carrier'].sum() /df_altered['Carrier Launched'].sum() 
            df_stage5 =  df_altered['1st Completed Load on Orcas'].sum() /df_altered['T3 Carrier'].sum()
            df_stages = pd.Series([df_stage0,       df_stage1,    df_stage2,       df_stage3,      df_stage4,     df_stage5])
            labels =               ['T0 Carrier', 'T1 Carrier','T2 Carrier', 'Carrier Launched', 'T3 Carrier', '1st Completed Load on Orcas']
            df_new = pd.DataFrame({'stages':labels, 'conversion_ratio' :df_stages})
            
            #plotting conversion ratio
            # fig3, ax = plt.subplots(figsize=(8,3))
            
            # sns.barplot(x=df_new['stages'], y= df_new['conversion_ratio'])
            # sns.lineplot(x=df_new['stages'], y= df_new['conversion_ratio'], color='black', linewidth=3)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_new['stages'], y= df_new['conversion_ratio']))
            fig3.add_trace(go.Scatter(x=df_new['stages'], y= df_new['conversion_ratio'], mode='lines', line_width=3, name='conversion_ratio'))

            # ax.set_xlabel("Stages")
            # ax.set_ylabel("Conversion Ratio by Stage", rotation=90)
            # ax.tick_params(axis='x', rotation=45)
            # ax.legend()
            col7.write(fig3)
            col6.header('')
            col6.header('')
            col6.header('')
            col6.table(df_new)
            st.title('')
            df_coversion_ratio = df_altered['1st Completed Load on Orcas'].sum() /df_altered['T0 Carrier'].sum() * 100    
            st.header( f'The conversion Ratio of T0 Carrier to booking its orders on Orcas is {format(df_coversion_ratio, ".2f")}%' )
            st.button('Clean Data')
            










        
        elif st.button('Clean Data')==1:
            df_altered = df_altered[df_altered['T0 Carrier'] < 100]
            col3 , col4  = st.columns(2)
            st.title('')
            col6, col7 = st.columns(2)
            new_df = df_altered.groupby(['Lead Source'])['T0 Carrier'].sum() # grouping by Lead Source and summing the T0 Carrier
            col3.title('')
            col3.title('')
            col3.title('')
            col3.table(new_df) # displaying the dataframe
            col3.subheader(f"Total T0 Carrier: {new_df.sum()}")
            col3.subheader(f"Total Carrier on Orcas: {df_altered['1st Completed Load on Orcas'].sum()}") # displaying the total sum
            col3.subheader(f"Conversion Ratio: {format(df_altered['1st Completed Load on Orcas'].sum() / new_df.sum() * 100, '.2f')} % ")
            # Barplot for T0 Carrier
            # fig1, ax = plt.subplots(figsize=(8,3))
            # sns.barplot(x= new_df.index, y = new_df)
            # ax.set_xlabel("Lead Source")
            # ax.set_ylabel("Count", rotation=90)
            # ax.tick_params(axis='x', rotation=45)
            # ax.legend()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x= new_df.index, y = new_df,mode='lines', line_width=3, name='Maximum Lead count'))
            fig.add_trace(go.Bar(x= new_df.index, y = new_df, name='Lead Count'))
            col4.write(fig)



            # Scatter Plot
            fig2, ax2 = plt.subplots(figsize=(8,3))
            sns.scatterplot(x =df_altered['Lead Source'], y=df_altered['T0 Carrier'])
            ax2.tick_params(axis='x', rotation=45)
            # col5.write(fig2)
        
            # Calculating Conversions
            df_stage0 =  df_altered['T0 Carrier'].sum() /  df_altered['T0 Carrier'].sum()
            df_stage1 =  df_altered['T1 Carrier'].sum() /  df_altered['T0 Carrier'].sum()
            df_stage2 =  df_altered['T2 Carrier'].sum() / df_altered['T1 Carrier'].sum() 
            df_stage3 =  df_altered['Carrier Launched'].sum() /df_altered['T2 Carrier'].sum()  
            df_stage4 =  df_altered['T3 Carrier'].sum() /df_altered['Carrier Launched'].sum() 
            df_stage5 =  df_altered['1st Completed Load on Orcas'].sum() /df_altered['T3 Carrier'].sum()
            df_stages = pd.Series([df_stage0,       df_stage1,    df_stage2,       df_stage3,      df_stage4,     df_stage5])
            labels =               ['T0 Carrier', 'T1 Carrier','T2 Carrier', 'Carrier Launched', 'T3 Carrier', '1st Completed Load on Orcas']
            df_new = pd.DataFrame({'stages':labels, 'conversion_ratio' :df_stages})
            


            df_temp = df_altered.sum().reset_index()
            df_temp = pd.DataFrame(df_temp)
            df_temp = df_temp.drop(df_temp.index[0])
            df_temp = df_temp.drop(df_temp.index[6])
            df_temp = df_temp.reset_index()
            df_temp = df_temp.drop( ['level_0','index'] , axis=1)
            df_new = pd.concat([df_new, df_temp], axis=1)
            df_new.rename(columns={0: 'Total Leads'}, inplace=True, errors='raise')
         
            #plotting conversion ratio)
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_new['stages'], y= df_new['conversion_ratio'],  name='Conversion Ratio'))
            fig3.add_trace(go.Scatter(x=df_new['stages'], y= df_new['conversion_ratio'], mode='lines', line_width=3, name='Conversion Ratio in Percentage'))
            col7.write(fig3)
            col6.header('')
            col6.header('')
            col6.header('')
            col6.table(df_new)
            st.title('')

            # if st.button('Start Analysis')==1:
            col17, col18 = st.columns(2)
            df_coversion_ratio = df_altered['1st Completed Load on Orcas'].sum() /df_altered['T0 Carrier'].sum() * 100    
            st.header( f'The conversion Ratio of T0 Carrier to booking its orders on Orcas is {format(df_coversion_ratio, ".2f")}%' )
     
            # Month Conversion
            fig4 = go.Figure()
            month_conversion = df_altered.groupby('Month')['1st Completed Load on Orcas', 'T0 Carrier'].sum().reset_index()
            month_conversion = Sort_Dataframeby_Month(df=month_conversion, monthcolumnname='Month')
            month_conversion['conversion_by_month'] = (month_conversion['1st Completed Load on Orcas'] / month_conversion['T0 Carrier'] * 100)        
            col17.title('')
            col17.title('')
            col17.title('')
            col17.table(month_conversion)
            fig4.add_trace(go.Scatter(x=month_conversion['Month'], y= month_conversion['1st Completed Load on Orcas'], mode='lines', line_width=3, name='Total leads booking orders on Orcas'))
            fig4.add_trace(go.Scatter(x=month_conversion['Month'], y= month_conversion['conversion_by_month'], mode='lines', line_width=3, name='Conversion Ratio by Month'))
            col18.write(fig4)
            st.title('--------------------------------------------------------------------------------------------')
            st.markdown('# Task 1 #')
            st.markdown('## *Questions that I would like to ask are:*  ##')
            st.header('1. Which stage conversion are you expecting to find out?')
            st.header('2. Which month or quarter did you find a drop in conversion ratio?')
            st.header('3. Do we have a time frame where we can find out if these Leads are getting converted in future?')         
               
            st.title('--------------------------------------------------------------------------------------------')
            st.markdown('# Task 2 #')
            
            st.header('1. The most important parameters that we should consider is the transition of lead at various touch points are:')
            image = Image.open('carrierbook.png')
            st.image(image, caption='')



            
            st.header('2. Scoring Model:')
            image = Image.open('scoring_model.png')
            st.image(image, caption='Scoring Model')

            st.header('3. There is 1 type of outliers that I found from the data:')            
            image = Image.open('incorrect_data.png')
            st.image(image, caption='Incorrect data')
            
            st.header('Q4. Steps towards building a predictive lead scoring model:')            
            image = Image.open('modelsteps.png')
            st.image(image, caption='')

            st.header('5. Many aspects of data that I would like to enrich are based on the following questions:')
            image = Image.open('implictdata.png')
            st.image(image, caption='')
            image = Image.open('explicitdata.png')
            st.image(image, caption='')
            
            st.header('6. I would need to involve data from marketing team and colloboration of sales team to build a correct model.') 
      