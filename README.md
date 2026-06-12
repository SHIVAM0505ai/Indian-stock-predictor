# 🇮🇳 Nifty Oracle — Indian Stock Market Intelligence Terminal

A **Bloomberg-style** NSE/BSE stock analysis dashboard with live data, 15+ technical indicators, and ML-powered price predictions.

---

# Features

 Feature & Details 

 *Live Data* | NSE/BSE data via Yahoo Finance API, auto-refreshes every 10s 
 *Stocks** | 500+ stocks across 20+ sectors 
 *Technical Analysis** | RSI, MACD, EMA, SMA, Bollinger Bands, ATR, Stochastic, OBV, Momentum, ROC 
 *ML Prediction** | XGBoost + Random Forest + Gradient Boosting ensemble 
 *Multi-Horizon** | Forecasts for 1, 3, 5, 10, 20 days ahead 
 *Signals** | Buy/Sell/Hold recommendations with signal breakdown 
 *Fundamentals** | P/E, P/B, EPS, ROE, Market Cap, Revenue 
 *UI** | Bloomberg-style dark terminal, animated ticker tape 

# Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Open in Browser

The app will open at `http://localhost:8501`

---

# Project Structure

```
indian-stock-predictor/
├── app.py             # Main Streamlit app (UI + layout)
├── nse_stocks.py      # 500+ NSE stocks by sector
├── data_fetcher.py    # Live data + 15+ technical indicators
├── model_trainer.py   # XGBoost + RF + GB ensemble ML models
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

# ML Model Details

The prediction engine uses a **VotingRegressor** ensemble:

| Model | Purpose |
|---|---|
| **XGBoost** | Gradient boosted trees, handles non-linear patterns |
| **Random Forest** | Bagging ensemble, reduces overfitting |
| **Gradient Boosting** | Sequential boosting, captures complex relationships |

**Feature Engineering:**
- 15+ technical indicator values
- Price return over multiple timeframes (1d, 5d, 20d)
- Volume ratio (vs 20-day average)
- Moving average crossover ratios
- Time features (day of week, month)

**Validation:**
- TimeSeriesSplit cross-validation (respects temporal order)
- Confidence score = 100 - 2×MAPE (capped 30–90%)

---

# Technical Indicators

| Category | Indicators |
|---|---|
| **Momentum** | RSI(14), Stochastic(14), Williams %R, Momentum, ROC |
| **Trend** | SMA(20/50/200), EMA(9/21/50), MACD(12/26/9) |
| **Volatility** | Bollinger Bands(20,2), ATR(14), BB Width |
| **Volume** | OBV, Volume SMA(20), Volume Ratio |
| **Support/Resistance** | Pivot Points, R1/R2, S1/S2 |

---

# Disclaimer

> This application is for **educational purposes only**. It does not constitute financial advice. Stock market investments involve risk. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.

---

# Troubleshooting

**yfinance rate limiting:** If you get errors fetching data, wait a few minutes and try again.

**Missing data:** Some small-cap stocks may have limited data. The ML model requires at least 60 days of history.

**XGBoost not installed:** The app falls back to Random Forest + Gradient Boosting ensemble if XGBoost is unavailable.

**pandas-ta not working:** The app uses manual indicator calculations as fallback.

