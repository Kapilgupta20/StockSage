import streamlit as st

st.set_page_config(page_title="About")

st.markdown(
    """# **StockSage**
### **Analyzing today, Predicting tomorrow**

 **StockSage is a web application that predicts stock prices using LSTM (Long Short-Term Memory) machine learning models. Built with Streamlit, this app provides an intuitive interface for users to analyze historical stock data and get predictions for future stock prices.**

## **Tech Stack**

StockSage is built with these core frameworks and modules:

- **Streamlit** - To create the web app UI and interactivity 
- **YFinance** - To fetch financial data from Yahoo Finance API
- **TensorFlow/Keras** - To build the LSTM time series forecasting model


## **Key Features**

- **Real-time data** - Fetch latest prices and fundamentals using yfinance API
- **Financial charts** - Interactive historical stock price visualization
- **LSTM forecasting** - Make statistically robust predictions
- **Backtesting** - Evaluate model performance
- **Responsive design** - Works on all devices


## **Future Roadmap**

We're constantly working to improve StockSage. Here are some features we're considering for future updates:

- **News Feed** - Integration of real-time news related to selected stocks to provide more context for predictions.
- **User Login System** - Personalized experience with features like:
        - Saving favorite stocks
        - Customized watchlists
        - Tracking prediction accuracy over time
- **Extended Prediction Timeframes** - Options for weekly and monthly predictions in addition to daily.
- **Sentiment Analysis** - Incorporate social media sentiment into predictions.
- **Portfolio Optimization** - Suggest optimal portfolio allocations based on predictions and risk tolerance.

We welcome contributions and suggestions for these or other features!


## **Disclaimer**
 **This application is for educational purposes only. The predictions should not be used as financial advice or for making investment decisions.**
"""
)
