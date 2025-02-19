import streamlit as st
from Helper import *

st.set_page_config(page_title="Home")


st.sidebar.markdown("## **User Input Features**")
stock_dict = fetch_stocks()
# stock = st.sidebar.selectbox("### **Select stock**", list(stock_dict.keys()), format_func=lambda x: stock_dict[x])
stock = st.sidebar.selectbox("### **Select stock**", list(stock_dict.keys()), format_func=lambda x: stock_dict[x])
stock_ticker = stock
stock = f"{stock_dict[stock]}"
st.sidebar.text_input(label="### **Stock ticker code**", placeholder=stock_ticker, disabled=True)


try:
    stock_data_info = fetch_stock_info(stock_ticker)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

try:
    prediction = predict(stock_ticker)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.markdown("# **StockSage**")
st.markdown("##### **Analyzing today, Predicting tomorrow**")

st.divider()

st.header(stock)

st.markdown("  \n")
col1, col2 = st.columns(2)
col1.dataframe(
    pd.DataFrame({"Current Price": [stock_data_info["Market Data"]["currentPrice"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Predicted Close": [prediction]}),
    hide_index=True,
    width=500,
)

st.markdown("  \n")
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
period = "ytd"
interval = "1d"
if col1.button("1D"):
    period = "1d"
    interval = "1m"
if col2.button("5D"):
    period = "5d"
    interval = "15m"
if col3.button("1M"):
    period = "1mo"
    interval = "60m"
if col4.button("3M"):
    period = "3mo"
    interval = "1d"
if col5.button("6M"):
    period = "6mo"
    interval = "1d"
if col6.button("1Y"):
    period = "1y"
    interval = "1d"
if col7.button("2Y"):
    period = "2y"
    interval = "1wk"
if col8.button("5Y"):
    period = "5y"
    interval = "1wk"
if col9.button("10Y"):
    period = "10y"
    interval = "1mo"
if col10.button("Max"):
    period = "max"
    interval = "3mo"
stock_data = fetch_stock_history(stock_ticker, period, interval)
st.line_chart(stock_data['Close'])
st.markdown("  \n")

col1, col2 = st.columns(2)
col1.dataframe(
    pd.DataFrame({"Open": [stock_data_info["Market Data"]["open"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Previous Close": [stock_data_info["Market Data"]["previousClose"]]}),
    hide_index=True,
    width=500,
)
col1.dataframe(
    pd.DataFrame({"Day Low": [stock_data_info["Market Data"]["dayLow"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Day High": [stock_data_info["Market Data"]["dayHigh"]]}),
    hide_index=True,
    width=500,
)
col1.dataframe(
    pd.DataFrame(
        {"Average Volume": [stock_data_info["Volume and Shares"]["averageVolume"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Market Cap": [stock_data_info["Valuation and Ratios"]["marketCap"]]}
    ),
    hide_index=True,
    width=500,
)
col1.dataframe(
    pd.DataFrame(
        {"Fifty-Two Week Low": [stock_data_info["Market Data"]["fiftyTwoWeekLow"]]}
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Fifty-Two Week High": [stock_data_info["Market Data"]["fiftyTwoWeekHigh"]]}
    ),
    hide_index=True,
    width=500,
)

col1, col2, col3 = st.columns(3)
col1.dataframe(
    pd.DataFrame(
        {"Dividend Rate": [stock_data_info["Dividends and Yield"]["dividendRate"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {"Dividend Yield": [stock_data_info["Dividends and Yield"]["dividendYield"]]}
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {"Payout Ratio": [stock_data_info["Dividends and Yield"]["payoutRatio"]]}
    ),
    hide_index=True,
    width=300,
)