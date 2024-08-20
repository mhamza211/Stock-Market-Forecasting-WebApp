import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

#giving title
app_name="Stock market forcasting app"
st.title(app_name)

#giving subtitle
st.subheader('This app is created to forcast the stock market price of the selected company ')

#adding image
st.image('https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg')

#makind a side bar with information of start date,end date and selecting the company
st.sidebar.header('Select the parameters from below ')
start_data=st.sidebar.date_input("Start date",date(2021,1,1))
end_date=st.sidebar.date_input("End date",date(2021,12,1))

#giving ticker name of compnay to ticker_list
ticker_list=['AAPL','MSFT','GOOG','GOOGL','META','TSLA','NVDA','ADBE','PYPL','INCT','CMCSA','NFLX','PEP']

#making dropdown menu to select the ticker name of company
ticker=st.sidebar.selectbox("Select the company ",ticker_list)

#fech the data of ticker
data=yf.download(ticker,start=start_data,end=end_date)

#making index for date
data.insert(0,"Date",data.index,True)
data.reset_index(drop=True,inplace=True)
st.write("Data from",start_data,"to",end_date)
st.write(data)

#plot the data
st.header("Data visualization")
st.subheader("plot of the data")
st.write("**Note**: Select your specfic date range on the side bar, or zoom in on the plot and select your specfic column.")
fig=px.line(data,x='Date',y=data.columns,title="Closing price of stocks",width=1000,height=600)
st.plotly_chart(fig)    

#add the selct box to select column from data
column=st.selectbox("Select the column to be used for forcasting ",data.columns[1:])

#subsetting the data
data=data[['Date',column]]
st.write("Selected Data ")
st.write(data)

#check stationary
st.header("Is data stationary?")
st.write(adfuller(data[column])[1]<0.05)

#decomposing the data
st.header("Decomposition of data")
decomposition=seasonal_decompose(data[column],model='additive',period=12)
st.write(decomposition.plot())

# Make same plot in plotly
st.write("## Plotting the decomposition in Plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title="Trend", width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title="Seasonality", width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title="Residuals", width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}))

# Let's run the model
# User input for three parameters of the model and seasonal order
p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

#model summary  
model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

# Print model summary
st.header('Model Summary')
st.write(model.summary())
st.write("-----")
#predict the value with user input values
st.write("<p style='color:green; font-size: 50px; font-weight:bold;'>Forcasting the data<p/>",unsafe_allow_html=True)

#predict future values
forcast_period=st.number_input('Select the number of days to forcast ',1,365,10)

predictions=model.get_prediction(start=len(data),end=len(data)+forcast_period)
predictions=predictions.predicted_mean 
#st.write(predictions)

# Add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index,True)
predictions.reset_index(drop=True,inplace=True)
st.write("## Predictions", predictions)
st.write("## Actual Data", data)
st.write("---")

# lets plot the data
fig = go.Figure()
# add actual data to the plot
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
# add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted', line=dict(color='red')))
# set the title and axis labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1200, height=400)
# display the plot
st.plotly_chart(fig)

#add button to show and hide seprate plots
show_plots=False
if st.button("Show seprate plots"):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column], title='Actual', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}).update_xaxes(rangeslider_visible=True))
        st.write(px.line(x=predictions["Date"], y=predictions["predicted_mean"], title='Predicted', width=1200, height=400, labels={'x': 'Date', 'y': 'Price'}) .update_xaxes(rangeslider_visible=True))
        show_plots = True
    else:
        show_plots = False

# Add hide plots button
hide_plots = False
if st.button("Hide Separate Plots"):
    if not hide_plots:
        hide_plots = True
    else:
        hide_plots = False

st.write("----")

st.write("<p style='color:green; font-size: 50px; font-weight:bold;'>Thank you for using this app<p/>",unsafe_allow_html=True)
st.write("---")

st.write("<p style='color:black; font-size: 25px;'>About the author:<p/>",unsafe_allow_html=True)
st.write("---")