import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as pyo

st.title('Stock Price Predictor Web Application')

stock_symbol = st.text_input('Enter stock symbol:', 'AAPL')
start_date = st.date_input('Start date:', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date:', value=pd.to_datetime('today'))
forecast_days = st.slider('Forecast days:', 1, 365, 30)

data = yf.download(stock_symbol, start=start_date, end=end_date)
data.reset_index(inplace=True)

st.write(f'Displaying historical stock data for {stock_symbol} from {start_date} to {end_date}')
st.line_chart(data[['Date', 'Close']].set_index('Date'))

data_for_prediction = data[['Date', 'Close']]
data_for_prediction.columns = ['ds', 'y']

model = Prophet(daily_seasonality=True)
model.fit(data_for_prediction)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

st.write(f'Predicting stock prices for the next {forecast_days} days:')
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)
