# 📈 Stock Market Prediction using Hybrid LSTM Model

## 📄 Description
This project focuses on predicting stock market trends using a hybrid approach that combines feature expansion techniques and LSTM-based deep learning models. The goal is to improve prediction accuracy by incorporating technical indicators and filtering noisy market movements.

---

## 🎯 Objectives
- Predict stock price movement (up/down)
- Improve prediction accuracy using feature engineering
- Reduce noise using threshold-based filtering
- Build a hybrid deep learning model (LSTM / BiLSTM)

---

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- yfinance (for stock data)
- ta (technical analysis indicators)

---

## 📊 Dataset
- Source: Yahoo Finance (via `yfinance` API)
- Data includes:
  - Open, High, Low, Close, Volume
- Additional features:
  - Technical indicators (SMA, EMA, RSI, ATR, etc.)
  - Lag features
  - Rolling statistics

---

## ⚙️ Features
- Feature expansion using technical indicators
- Hybrid LSTM / Bidirectional LSTM model
- Threshold-based classification for better accuracy
- Balanced dataset handling
- Time-series data processing

---

## ⚙️ Setup Instructions

Follow these steps to run the project locally:

Project Structure
project/
│── main.py
│── requirements.txt
│── README.md
│── data/
│── models/