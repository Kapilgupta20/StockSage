import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
from math import floor,ceil,sqrt
from pathlib import Path
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_history(stock_ticker, period, interval):
    stock_data = yf.Ticker(stock_ticker)
    stock_data_history = stock_data.history(period, interval)[["Close"]]
    return stock_data_history

def fetch_stocks():
    stock_dict = {
        "RELIANCE.NS": "Reliance Industries Ltd.",
        "TCS.NS": "Tata Consultancy Services Ltd.",
        "INFY.NS": "Infosys Ltd.",
        "HDFCBANK.NS": "HDFC Bank Ltd.",
        "TATAMOTORS.NS": "Tata Motors Ltd."
    }
    return stock_dict

@st.cache_data(ttl=3600)
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

# @st.cache_data(ttl=3600)
# def predict(stock_ticker):
#     from sklearn.preprocessing import MinMaxScaler
#     from keras.models import Sequential
#     from keras.layers import Dense, Dropout, LSTM, Bidirectional
#     from keras.callbacks import EarlyStopping
#     df = yf.download(stock_ticker, start = dt.datetime(2010, 1, 1), end = dt.date.today(), progress=False)
#     df["Date"] = df.index
#     df = df[["Date", "Close"]]
#     df.reset_index(drop=True, inplace=True)
#     df['Date'] = pd.to_datetime(df.Date,format='%Y/%m/%d')
#     df.index = df['Date']
#     df.dropna(inplace=True)
#     df_new=df[['Close']]
#     dataset = df_new.values
#     shape=df.shape[0]
#     train=df_new[:ceil(shape*0.75)]
#     valid=df_new[ceil(shape*0.75):]
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaleddata = scaler.fit_transform(dataset)
#     xtrain, ytrain = [], []
#     for i in range(50,len(train)):
#         xtrain.append(scaleddata[i-50:i,0])
#         ytrain.append(scaleddata[i,0])
#     xtrain, ytrain = np.array(xtrain), np.array(ytrain)
#     xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1],1))
#     model = Sequential()
#     model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(50, 1))))
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#     history = model.fit(xtrain, ytrain, epochs=20, batch_size=32, validation_split=0.2, callbacks=[earlystopping], verbose=1)
#     inputs = df_new[len(df_new) - len(valid) - 50:].values
#     inputs = inputs.reshape(-1,1)
#     inputs  = scaler.transform(inputs)
#     endprice = []
#     endprice.append(inputs[-50:,0])
#     endprice = np.array(endprice)
#     endprice = np.reshape(endprice, (endprice.shape[0],endprice.shape[1],1))
#     predictedprice = model.predict(endprice)
#     predictedprice = scaler.inverse_transform(predictedprice)
#     return predictedprice

def predict(stock_ticker):
    import datetime as dt
    import yfinance as yf
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import load_model # type: ignore

    stockticker = stock_ticker.split(".")[0]

    # Load trained model
    model = load_model(Path.cwd()/".."/"model"/f"model_{stockticker}.h5")

    # Load the scaler
    scaler = np.load(Path.cwd()/".."/"model"/f"scaler_{stockticker}.npy", allow_pickle=True).item()
    
    # Fetch latest stock data
    df = yf.download(stock_ticker, period="3mo", progress=False)
    df["Date"] = df.index
    df = df[["Date", "Close"]]
    df.reset_index(drop=True, inplace=True)
    df['Date'] = pd.to_datetime(df.Date, format='%Y/%m/%d')
    df.index = df['Date']
    df.dropna(inplace=True)
    
    df_new = df[['Close']]
    
    # Prepare data for prediction
    inputs = df_new[len(df_new) - 50:].values  # Take last 50 days
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)  # Scale data
    
    endprice = np.array([inputs[:, 0]])  # Reshape for prediction
    endprice = np.reshape(endprice, (endprice.shape[0], endprice.shape[1], 1))
    
    # Predict future price
    predicted_price = model.predict(endprice)
    predicted_price = scaler.inverse_transform(predicted_price)  # Convert back to original scale
    
    # print(f"Predicted Price for {stock_ticker}: {predicted_price[0][0]}")
    return predicted_price[0][0]

