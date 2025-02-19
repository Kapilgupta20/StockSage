import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_history(stock_ticker, period, interval):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_history = stock_data.history(period, interval)[["Close"]]
    return stock_data_history

@st.cache_data(ttl=3600)
def fetch_stocks():
    stock_dict = {
        "RELIANCE.NS": "Reliance Industries Ltd.",
        "TCS.NS": "Tata Consultancy Services Ltd.",
        "INFY.NS": "Infosys Ltd.",
        "HDFCBANK.NS": "HDFC Bank Ltd.",
        "TATAMOTORS.NS": "Tata Motors Ltd."
    }
    return stock_dict

def fetch_stock_info(stock_ticker):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_info = stock_data.info

    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")

    stock_data_info = {
        "Basic Information": {
            "symbol": safe_get(stock_data_info, "symbol")
        },
        "Market Data": {
            "currentPrice": safe_get(stock_data_info, "currentPrice"),
            "previousClose": safe_get(stock_data_info, "previousClose"),
            "open": safe_get(stock_data_info, "open"),
            "dayLow": safe_get(stock_data_info, "dayLow"),
            "dayHigh": safe_get(stock_data_info, "dayHigh"),
            "fiftyTwoWeekLow": safe_get(stock_data_info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": safe_get(stock_data_info, "fiftyTwoWeekHigh")
        },
        "Volume and Shares": {
            "averageVolume": safe_get(stock_data_info, "averageVolume")
        },
        "Dividends and Yield": {
            "dividendRate": safe_get(stock_data_info, "dividendRate"),
            "dividendYield": safe_get(stock_data_info, "dividendYield"),
            "payoutRatio": safe_get(stock_data_info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap": safe_get(stock_data_info, "marketCap")
        }
    }
    return stock_data_info

@st.cache_data(ttl=3600)
def predict(stock_ticker):
    from keras.models import load_model # type: ignore
    stockticker = stock_ticker.split(".")[0]
    model = load_model(Path.cwd()/"model"/f"model_{stockticker}.h5")
    scaler = np.load(Path.cwd()/"model"/f"scaler_{stockticker}.npy", allow_pickle=True).item()
    df = yf.download(stock_ticker, period="3mo", progress=False)
    df["Date"] = df.index
    df = df[["Date", "Close"]]
    df.reset_index(drop=True, inplace=True)
    df['Date'] = pd.to_datetime(df.Date, format='%Y/%m/%d')
    df.index = df['Date']
    df.dropna(inplace=True)
    df_new = df[['Close']]    
    inputs = df_new[len(df_new) - 50:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    endprice = np.array([inputs[:, 0]])
    endprice = np.reshape(endprice, (endprice.shape[0], endprice.shape[1], 1))
    predicted_price = model.predict(endprice)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

