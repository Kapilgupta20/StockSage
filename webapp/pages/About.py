import streamlit as st

st.set_page_config(page_title="About")

st.markdown(
    """# **StockSage**
#### **Analyzing today, Predicting tomorrow**

 **StockSage is a web application that predicts stock prices using LSTM (Long Short-Term Memory) machine learning models. Built with Streamlit, this app provides an intuitive interface for users to analyze historical stock data and get predictions for future stock prices.**


### **Key Features**

- **Real-time data** - Fetches the latest stock prices and fundamentals using `yfinance` API
- **Financial charts** - Interactive historical stock price visualization.
- **LSTM forecasting** - Uses deep learning to make statistically robust predictions
- **Backtesting** - Evaluates model performance on past stock data
- **Responsive design** - Works on all devices.


### **Tech Stack**

StockSage is built with these core frameworks and modules:

- **Streamlit** - To create the web app UI and interactivity 
- **YFinance** - To fetch financial data from Yahoo Finance API
- **TensorFlow/Keras** - To build the LSTM time series forecasting model


### **Performance Metrics**

- **Model Accuracy (R² Score):** ~90-95% (Strong trend capture)
- **MAPE (Mean Absolute Percentage Error):** ~2-5% (High prediction reliability) 
- **RMSE (Root Mean Squared Error):** ~30-60 (Average deviation of ₹30-60 per prediction)  
- **Training Time:** Approximately **2-3 minutes** on a standard GPU  
- **Prediction Speed:** **<1 second** per stock after loading the trained model  


### **Getting Started**

#### Prerequisites:
- Python 3.8 or higher.
- Required Python libraries (listed in `requirements.txt`).

#### Installation:
1. Clone this repository

   ```bash
   git clone https://github.com/Kapilgupta20/StockSage.git

2. Change directory

    ```bash
   cd StockSage
   cd webapp

3. Install dependencies

    ```bash
    pip install -r requirements.txt

4. Run the app

    ```bash
    streamlit run Home.py

The app will be live at `http://localhost:8501`


### **Future Roadmap**

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


### **Disclaimer**
 **This application is for educational purposes only. The predictions should not be used as financial advice or for making investment decisions.**
"""
)
