import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import date

current_date = date.today()

# Extract year, month, and day as integers
year = current_date.year
month = current_date.month
day = current_date.day
# Title
title = 'Smart Stock Forecaster'
st.title(title)
st.subheader('Empowering You to Stay Ahead in the Market')

# Fetching Online Image
link_to_image = 'https://plus.unsplash.com/premium_photo-1681487769650-a0c3fbaed85a?q=80&w=1255&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D'
st.image(link_to_image)
st.write('**Note:** This app also includes stock predictions for companies listed on the Pakistani Stock Market.')

# Sidebar to take input company
st.sidebar.header('Select the parameters from below')
start_date = st.sidebar.date_input('Start Date', date(2023,11 ,25))
end_date = st.sidebar.date_input('End Date', date(year, month, day))

pakistani_companies = [
    'Engro Corporation',
    'Habib Bank Limited',
    'Pakistan Petroleum Limited',
    'Oil & Gas Development Company',
    'Lucky Cement',
    'United Bank Limited',
    'MCB Bank Limited',
    'Fauji Fertilizer Company',
    'Hub Power Company',
    'K-Electric Limited',
    'Nishat Mills Limited',
    'Pakistan State Oil',
    'D.G. Khan Cement',
    'Millat Tractors Limited',
    'Packages Limited',
    'Mari Petroleum Company',
    'Sui Northern Gas Pipelines Limited',
    'Sui Southern Gas Company',
    'Kot Addu Power Company',
    'Attock Petroleum Limited'
]

pakistani_tickers = [
    'ENGRO.KA',
    'HBL.KA',
    'PPL.KA',
    'OGDC.KA',
    'LUCK.KA',
    'UBL.KA',
    'MCB.KA',
    'FFC.KA',
    'HUBC.KA',
    'KEL.KA',
    'NML.KA',   
    'PSO.KA',  
    'DGKC.KA', 
    'MTL.KA',   
    'PKGS.KA',  
    'MARI.KA',  
    'SNGP.KA', 
    'SSGC.KA', 
    'KAPCO.KA', 
    'APL.KA'    
]

company = st.sidebar.selectbox('Select a stock', pakistani_companies)
# Get the index of the selected company
company_index = pakistani_companies.index(company)

# Use the index to retrieve the corresponding ticker
ticker = pakistani_tickers[company_index]

# Fetching Data
data = yf.download(ticker, start=start_date, end=end_date)
data = data.reset_index()
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
st.write('From ', start_date, ' to ', end_date)
st.dataframe(data,width=1000,height=400)
st.markdown("#### Overview")


# Plot the selected attribute over time
fig = px.line(data, x=data.index, y=data.columns[:-1], title="Stock Prices", width=1000, height=600)
st.plotly_chart(fig)

# Stock Forecasting with ETS
st.markdown('## Stock Price Forecasting')
forecast_period = st.number_input('Select the number of days to forecast', 1, 120, 30)
# Select which attribute to plot
column = st.selectbox('**Which attribute do you want to forecast?**', data.columns[1:])


#Assigning frequency of the dataset which is five buisness days form monday to friday
data = data.asfreq('B')
# Fit the ETS model

# ets_model = ExponentialSmoothing(data[column], seasonal='add', trend='add', seasonal_periods=12).fit()
# forecast = ets_model.forecast(steps=forecast_period)
data[column] = data[column].interpolate(method='time').bfill()

ets_model = ExponentialSmoothing(data[column], trend='mul', seasonal='add', seasonal_periods=5, damped_trend=True).fit()
predictions = ets_model.forecast(steps=forecast_period)
predictions = pd.DataFrame(predictions)
predictions[column] = predictions

# Plot actual data and predicted data
fig = go.Figure()

# Plot actual data
fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name='Actual Data',line=dict(color='blue')))

# Plot predicted data
fig.add_trace(go.Scatter(x=predictions.index, y=predictions[column], mode='lines', name='Predicted Data', line=dict(color='green')))

# Update layout
fig.update_layout(title="Stock Price Forecasting", xaxis_title="Date", yaxis_title="Price", width=1000, height=600)

# Display plot in Streamlit
st.plotly_chart(fig)

# Initialize session state for toggle
if 'show_summary' not in st.session_state:
    st.session_state.show_summary = False

# Toggle button
if st.button('Toggle Model Summary'):
    st.session_state.show_summary = not st.session_state.show_summary

# Show or hide model summary based on the button state
if st.session_state.show_summary:
    st.text(ets_model.summary())
else:
    st.write("Click the button to show the model summary")

# Footer Information
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>Built with ❤️ by Muhammad Suhaib Salman and Muhammad Hamza</p>
        <p>Contact: suhaibsalman200110@gmail.com , mhamza9484@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)