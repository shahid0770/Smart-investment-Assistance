# 🤖💸 Smart Investment Assistant  

An AI-powered stock companion built with **Flask, TensorFlow, and NLP**.  
It predicts stock prices, analyzes news sentiment, and provides smart investment recommendations in an interactive web dashboard.  

---

## ✨ Features  
- 🔍 Search stocks by name, ticker, or sector  
- 📈 AI-driven price forecasting (LSTM model)  
- 📰 News sentiment analysis (TextBlob + keywords)  
- ⭐ Buy / Hold / Sell recommendations  
- 🌍 Market overview (S&P 500, NASDAQ, Dow Jones, Nifty, Sensex)  
- 🎤 Voice commands + speech feedback  
- 📊 Interactive price & forecast charts  
- 🧾 Stock details (sector, P/E ratio, dividend yield, 52W high/low, etc.)  

---

## 🛠️ Tech Stack  
**Backend:** Flask, TensorFlow/Keras, Pandas, NumPy, yFinance, TextBlob  
**Frontend:** TailwindCSS, Chart.js, Speech Recognition & TTS  
**Data Sources:** Yahoo Finance, NewsAPI (optional)  

---

## ⚡ API Endpoints  
- `/api/health` → App & model status  
- `/api/search?q=<query>` → Search stocks  
- `/api/data?ticker=<symbol>` → Get OHLCV data  
- `/api/predict` → Price predictions  
- `/api/news/<ticker>` → News + sentiment  
- `/api/recommend` → AI recommendation  
- `/api/market-overview` → Market indices  

---

## 🚀 Getting Started  

```bash
# Clone repo
git clone https://github.com/your-username/smart-investment-assistant.git
cd smart-investment-assistant

# Install dependencies
pip install -r requirements.txt

# (Optional) Set API Keys
export NEWSAPI_KEY="your_newsapi_key"

# Run the app
python peko.py
