
import os, re, logging, traceback
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import json

from flask import Flask, jsonify, request, Response
import numpy as np, pandas as pd, yfinance as yf, requests
from textblob import TextBlob

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- Configuration ----------------
MODEL_PATH = "lstm_model.h5"
LOOKBACK = 30
DEFAULT_HORIZON = 14
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
MAX_NEWS = 8
TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smart-invest")

POPULAR_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Communication Services"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "sector": "Financial Services"},
    {"symbol": "V", "name": "Visa Inc.", "sector": "Financial Services"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "sector": "Healthcare"},
    {"symbol": "WMT", "name": "Walmart Inc.", "sector": "Consumer Defensive"},
    {"symbol": "PG", "name": "Procter & Gamble", "sector": "Consumer Defensive"},
    {"symbol": "MA", "name": "Mastercard Inc.", "sector": "Financial Services"},
    {"symbol": "DIS", "name": "Walt Disney Co.", "sector": "Communication Services"},
    {"symbol": "BAC", "name": "Bank of America", "sector": "Financial Services"},
    {"symbol": "XOM", "name": "Exxon Mobil Corp.", "sector": "Energy"},
    {"symbol": "HD", "name": "Home Depot Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "PYPL", "name": "PayPal Holdings", "sector": "Financial Services"},
    {"symbol": "CSCO", "name": "Cisco Systems", "sector": "Technology"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"symbol": "NKE", "name": "Nike Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "PFE", "name": "Pfizer Inc.", "sector": "Healthcare"},
    {"symbol": "ABT", "name": "Abbott Laboratories", "sector": "Healthcare"},
    {"symbol": "TMO", "name": "Thermo Fisher Scientific", "sector": "Healthcare"},
    {"symbol": "AVGO", "name": "Broadcom Inc.", "sector": "Technology"},
    {"symbol": "COST", "name": "Costco Wholesale", "sector": "Consumer Defensive"},
    {"symbol": "ACN", "name": "Accenture plc", "sector": "Technology"},
    {"symbol": "T", "name": "AT&T Inc.", "sector": "Communication Services"},
    {"symbol": "DHR", "name": "Danaher Corporation", "sector": "Healthcare"},
    {"symbol": "VZ", "name": "Verizon Communications", "sector": "Communication Services"},
    {"symbol": "NEE", "name": "NextEra Energy", "sector": "Utilities"},
    {"symbol": "UNH", "name": "UnitedHealth Group", "sector": "Healthcare"},
    {"symbol": "LIN", "name": "Linde plc", "sector": "Basic Materials"},
    {"symbol": "RTX", "name": "Raytheon Technologies", "sector": "Industrials"},
    {"symbol": "HON", "name": "Honeywell International", "sector": "Industrials"},
    {"symbol": "SBUX", "name": "Starbucks Corporation", "sector": "Consumer Cyclical"},
    {"symbol": "LOW", "name": "Lowe's Companies", "sector": "Consumer Cyclical"},
    {"symbol": "BMY", "name": "Bristol-Myers Squibb", "sector": "Healthcare"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technology"},
    {"symbol": "AXP", "name": "American Express", "sector": "Financial Services"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
    {"symbol": "IBM", "name": "International Business Machines", "sector": "Technology"},
    {"symbol": "CAT", "name": "Caterpillar Inc.", "sector": "Industrials"},
    {"symbol": "GS", "name": "Goldman Sachs Group", "sector": "Financial Services"},
    {"symbol": "UNP", "name": "Union Pacific Corporation", "sector": "Industrials"},
    {"symbol": "SPGI", "name": "S&P Global Inc.", "sector": "Financial Services"},
    {"symbol": "PLD", "name": "Prologis Inc.", "sector": "Real Estate"},
    {"symbol": "DE", "name": "Deere & Company", "sector": "Industrials"},
    {"symbol": "NOW", "name": "ServiceNow Inc.", "sector": "Technology"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financial Services"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy"},
    {"symbol": "TCS.NS", "name": "TCS", "sector": "Technology"},
    {"symbol": "INFY.NS", "name": "Infosys", "sector": "Technology"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Financial Services"},
    {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Financial Services"},
    {"symbol": "ITC.NS", "name": "ITC", "sector": "Consumer Defensive"},
    {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "Technology"},
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises", "sector": "Conglomerate"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Financial Services"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "Consumer Defensive"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Communication Services"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Financial Services"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Industrials"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Financial Services"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Basic Materials"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Automotive"},
    {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Consumer Cyclical"},
    {"symbol": "NTPC.NS", "name": "NTPC Limited", "sector": "Utilities"},
    {"symbol": "ONGC.NS", "name": "Oil and Natural Gas Corporation", "sector": "Energy"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation", "sector": "Utilities"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical", "sector": "Healthcare"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank", "sector": "Financial Services"},
    {"symbol": "COALINDIA.NS", "name": "Coal India", "sector": "Energy"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Basic Materials"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "Technology"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra", "sector": "Automotive"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra", "sector": "Technology"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries", "sector": "Consumer Defensive"},
    {"symbol": "NESTLEIND.NS", "name": "Nestle India", "sector": "Consumer Defensive"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries", "sector": "Basic Materials"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Basic Materials"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "sector": "Financial Services"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automotive"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries", "sector": "Basic Materials"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories", "sector": "Healthcare"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories", "sector": "Healthcare"},
    {"symbol": "CIPLA.NS", "name": "Cipla", "sector": "Healthcare"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Energy"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "sector": "Automotive"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors", "sector": "Automotive"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports", "sector": "Industrials"},
    {"symbol": "DLF.NS", "name": "DLF Limited", "sector": "Real Estate"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance", "sector": "Financial Services"},
    {"symbol": "HINDZINC.NS", "name": "Hindustan Zinc", "sector": "Basic Materials"},
    {"symbol": "BERGEPAINT.NS", "name": "Berger Paints", "sector": "Basic Materials"},
    {"symbol": "DABUR.NS", "name": "Dabur India", "sector": "Consumer Defensive"},
    {"symbol": "PIDILITIND.NS", "name": "Pidilite Industries", "sector": "Basic Materials"},
    {"symbol": "HAVELLS.NS", "name": "Havells India", "sector": "Industrials"},
    {"symbol": "GODREJCP.NS", "name": "Godrej Consumer Products", "sector": "Consumer Defensive"},
    {"symbol": "BOSCHLTD.NS", "name": "Bosch Limited", "sector": "Automotive"},
    {"symbol": "BIOCON.NS", "name": "Biocon Limited", "sector": "Healthcare"},
]

POS_WORDS = {"good","up","positive","gain","bull","beat","surge","rise","optimistic","upgrade","soared","strong","growth","profit","success","win","high","record","breakthrough","innovative","leader","exceed","outperform","rally","boom","thrive","flourish","prosper","expand","increase","soar","jump","climb","advance","improve","recover","rebound"}
NEG_WORDS = {"bad","down","negative","loss","bear","miss","sell","drop","plummet","pessimistic","downgrade","weak","decline","fall","crash","slump","trouble","worry","risk","danger","crisis","fail","bankrupt","cut","reduce","layoff","fire","fraud","scandal","investigation","lawsuit","debt","default","bankruptcy","recession","downturn","volatile","uncertain","uncertainty","fear","panic","selloff","plunge","tumble","slide","dip","downturn","slowdown","contraction","shrink","deteriorate","worsen"}

# ---------------- Sentiment Analysis ----------------
def advanced_sentiment_score(texts):
    if isinstance(texts, str): 
        texts = [texts]
    
    total_score = 0
    count = 0
    
    for text in texts:
        if not text:
            continue
            
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        
        # Keyword-based sentiment
        text_lower = text.lower()
        pos_count = sum(1 for word in POS_WORDS if word in text_lower)
        neg_count = sum(1 for word in NEG_WORDS if word in text_lower)
        
        # Combined score (weighted average)
        keyword_score = (pos_count - neg_count) / max(1, (pos_count + neg_count))
        combined_score = (polarity * 0.7) + (keyword_score * 0.3)
        
        total_score += combined_score
        count += 1
    
    if count == 0:
        return 0
    
    # Normalize to -1 to 1 range
    final_score = total_score / count
    return round(final_score, 3)

def get_sentiment_label(score):
    if score >= 0.3:
        return "Very Positive"
    elif score >= 0.1:
        return "Positive"
    elif score > -0.1:
        return "Neutral"
    elif score > -0.3:
        return "Negative"
    else:
        return "Very Negative"

def sample_news():
    return [
        "Tech shares surge after strong earnings reports beat expectations",
        "Investors cautious as inflation data remains elevated",
        "Company exceeds revenue estimates with record quarterly profits",
        "Analysts downgrade stock citing increased market volatility",
        "New product launch drives investor optimism for growth",
        "Regulatory concerns weigh on sector performance",
        "Merger announcement creates positive momentum for shares",
        "Supply chain issues may impact future earnings guidance"
    ]

@lru_cache(maxsize=256)
def fetch_news(ticker):
    if not NEWSAPI_KEY: 
        return sample_news()
    
    try:
        # Try NewsAPI first
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": ticker,
                "pageSize": MAX_NEWS,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": NEWSAPI_KEY
            },
            timeout=10
        )
        
        if r.ok:
            articles = r.json().get("articles", [])
            titles = [a.get("title") for a in articles if a.get("title") and a.get("title") != "[Removed]"]
            if titles: 
                return titles[:MAX_NEWS]
    except Exception as e:
        log.warning("NewsAPI fetch failed: %s", e)
    
    # Fallback to sample news
    return sample_news()

# ---------------- Stock Data Functions ----------------
def get_stock_info(ticker):
    """Get additional stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "marketCap": info.get("marketCap", 0),
            "peRatio": info.get("trailingPE", 0),
            "dividendYield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "52WeekHigh": info.get("fiftyTwoWeekHigh", 0),
            "52WeekLow": info.get("fiftyTwoWeekLow", 0),
            "volume": info.get("volume", 0),
            "avgVolume": info.get("averageVolume", 0)
        }
    except Exception as e:
        log.warning("Failed to get stock info for %s: %s", ticker, e)
        return {}

# ---------------- Model ----------------
class PriceModel:
    def __init__(self, path=MODEL_PATH, lookback=LOOKBACK):
        self.path = path
        self.lookback = lookback
        self.model = None
        
        if os.path.exists(path):
            try:
                self.model = load_model(path, compile=False)
                log.info("Loaded model from %s", path)
            except Exception as e: 
                log.warning("Model load failed: %s", e)

    def _prep(self, series):
        arr = np.array(series, dtype="float32")
        X = []
        y = []
        
        for i in range(len(arr) - self.lookback):
            X.append(arr[i:i+self.lookback])
            y.append(arr[i+self.lookback])
            
        return np.array(X).reshape(-1, self.lookback, 1), np.array(y)

    def train(self, series, epochs=20):
        X, y = self._prep(series)
        if len(X) < 20: 
            raise ValueError("Not enough data for training")
            
        m = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(80, return_sequences=False),
            Dropout(0.2),
            Dense(40, activation="relu"),
            Dense(1)
        ])
        
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])
        m.fit(
            X, y, 
            epochs=epochs, 
            batch_size=16,
            callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
            verbose=0
        )
        
        m.save(self.path)
        self.model = m
        log.info("Model trained and saved to %s", self.path)

    def forecast(self, series, horizon):
        if self.model is None or len(series) < self.lookback:
            self.train(series)
            
        seq = list(series[-self.lookback:])
        preds = []
        
        for _ in range(horizon):
            x = np.array(seq[-self.lookback:]).reshape(1, self.lookback, 1)
            p = float(self.model.predict(x, verbose=0)[0, 0])
            preds.append(round(p, 4))
            seq.append(p)
            
        return preds

# ---------------- Flask Application ----------------
app = Flask(__name__)
model = PriceModel()

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok", 
        "time": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model.model is not None
    })

@app.route("/api/search")
def search():
    q = (request.args.get("q") or "").strip().lower()
    if not q: 
        return jsonify([])
    
    # Filter by symbol, name, or sector
    results = [
        s for s in POPULAR_STOCKS 
        if q in s["symbol"].lower() or 
           q in s["name"].lower() or 
           (s.get("sector") and q in s["sector"].lower())
    ]
    
    # Add custom ticker if it matches pattern
    if not results and TICKER_RE.match(q): 
        results = [{"symbol": q.upper(), "name": f"{q.upper()} (custom)", "sector": "Unknown"}]
    
    return jsonify(results[:30])

@app.route("/api/data")
def data():
    ticker = request.args.get("ticker", "AAPL")
    n = int(request.args.get("n", 180))  # Default to 180 days
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{n}d").reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = df["date"].astype(str)
        
        # Add stock info
        info = get_stock_info(ticker)
        
        return jsonify({
            "prices": df.to_dict("records"),
            "info": info
        })
    except Exception as e: 
        log.error("Data fetch error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "AAPL")
    horizon = int(data.get("horizon", DEFAULT_HORIZON))
    
    try:
        prices = yf.Ticker(ticker).history(period="1y")["Close"].values
        predictions = model.forecast(prices, horizon)
        
        return jsonify({
            "ticker": ticker,
            "horizon": horizon,
            "predictions": predictions,
            "last_price": float(prices[-1]) if len(prices) > 0 else 0
        })
    except Exception as e: 
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/news/<ticker>")
def news(ticker):
    try:
        news_items = fetch_news(ticker)
        sentiment = advanced_sentiment_score(news_items)
        
        return jsonify({
            "ticker": ticker,
            "news": news_items,
            "sentiment_score": sentiment,
            "sentiment_label": get_sentiment_label(sentiment)
        })
    except Exception as e:
        log.error("News fetch error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "AAPL")
    horizon = int(data.get("horizon", DEFAULT_HORIZON))
    
    try:
        # Get price data
        prices = yf.Ticker(ticker).history(period="1y")["Close"].values
        if len(prices) == 0:
            return jsonify({"error": "No price data available"}), 400
            
        # Get predictions
        preds = model.forecast(prices, horizon)
        
        # Calculate price change
        price_change = (preds[-1] - prices[-1]) / prices[-1] * 100
        
        # Get news sentiment
        news_items = fetch_news(ticker)
        sentiment = advanced_sentiment_score(news_items)
        
        # Calculate recommendation score
        # Weight: 60% price momentum, 40% sentiment
        score = (0.6 * price_change) + (40 * sentiment)
        
        # Determine action
        if score > 5:
            action = "Strong Buy"
        elif score > 2:
            action = "Buy"
        elif score > -2:
            action = "Hold"
        elif score > -5:
            action = "Sell"
        else:
            action = "Strong Sell"
        
        # Get additional metrics
        stock_info = get_stock_info(ticker)
        
        return jsonify({
            "action": action,
            "score": round(score, 2),
            "predicted_return_pct": round(price_change, 2),
            "sentiment_score": sentiment,
            "sentiment_label": get_sentiment_label(sentiment),
            "predictions": preds,
            "current_price": round(float(prices[-1]), 2),
            "target_price": round(float(preds[-1]), 2),
            "news_sample": news_items[:3],
            "stock_info": stock_info
        })
        
    except Exception as e: 
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/market-overview")
def market_overview():
    """Get market overview data"""
    try:
        # Get major indices
        indices = {
            "SPY": "S&P 500",
            "QQQ": "NASDAQ 100",
            "DIA": "Dow Jones",
            "IWM": "Russell 2000",
            "^NSEI": "Nifty 50",
            "^BSESN": "Sensex"
        }
        
        overview = {}
        for symbol, name in indices.items():
            try:
                data = yf.Ticker(symbol).history(period="5d")
                if len(data) > 0:
                    current = data["Close"].iloc[-1]
                    previous = data["Close"].iloc[-2] if len(data) > 1 else current
                    change = ((current - previous) / previous) * 100
                    
                    overview[symbol] = {
                        "name": name,
                        "price": round(current, 2),
                        "change": round(change, 2),
                        "isPositive": change >= 0
                    }
            except Exception as e:
                log.warning("Failed to get data for %s: %s", symbol, e)
                continue
                
        return jsonify(overview)
    except Exception as e:
        log.error("Market overview error: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------------- Frontend ----------------
INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Smart Investment Assistant</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.3.0/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-gray-50 min-h-screen">
    <header class="bg-gradient-to-r from-indigo-600 to-blue-500 text-white shadow-lg">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex justify-between items-center">
                <div>
                    <h1 class="text-2xl font-bold"><i class="fas fa-chart-line mr-2"></i>Smart Investment Assistant</h1>
                    <p class="text-indigo-100 text-sm">AI-powered stock analysis and predictions</p>
                </div>
                <div id="marketStatus" class="text-sm bg-white/10 backdrop-blur-sm rounded-lg px-3 py-1">
                    <span>Loading market data...</span>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <!-- Search Section -->
        <section class="bg-white rounded-xl shadow-md p-6">
            <div class="flex flex-col md:flex-row gap-4 items-start md:items-center">
                <div class="flex-1">
                    <label class="block text-gray-700 font-medium mb-2">Search Stocks:</label>
                    <div class="relative">
                        <input id="searchBox" class="w-full border border-gray-300 rounded-lg px-4 py-3 pl-10 focus:ring-2 focus:ring-indigo-500 focus:border-transparent" placeholder="🔍 Type company name or ticker symbol">
                        <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                            <i class="fas fa-search text-gray-400"></i>
                        </div>
                        <button id="voiceBtn" class="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-indigo-500">
                            <i class="fas fa-microphone"></i>
                        </button>
                    </div>
                    <ul id="searchResults" class="mt-2 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto hidden"></ul>
                </div>
                <div class="flex gap-2">
                    <select id="horizon" class="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-indigo-500">
                        <option value="7">7 days</option>
                        <option value="14" selected>14 days</option>
                        <option value="30">30 days</option>
                    </select>
                    <button id="refreshBtn" class="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition flex items-center gap-2">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
            </div>
        </section>

        <!-- Market Overview -->
        <section id="marketOverview" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <!-- Market data will be loaded here -->
        </section>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Column - Chart -->
            <div class="lg:col-span-2">
                <div class="bg-white rounded-xl shadow-md p-6">
                    <div class="flex justify-between items-center mb-6">
                        <h2 class="text-xl font-semibold text-gray-800" id="companyName">Price & Forecast</h2>
                        <div class="text-sm text-gray-500" id="stockInfo"></div>
                    </div>
                    <div class="h-80">
                        <canvas id="priceChart"></canvas>
                    </div>
                    <div class="mt-4 flex justify-between items-center text-sm text-gray-500">
                        <span id="lastUpdated">Updated: -</span>
                        <div id="priceChange" class="font-medium"></div>
                    </div>
                </div>
            </div>

            <!-- Right Column - Recommendation & News -->
            <div class="space-y-6">
                <!-- Recommendation Card -->
                <div class="bg-gradient-to-br from-indigo-50 to-blue-100 rounded-xl shadow-md p-6">
                    <h3 class="font-bold text-lg text-gray-800 mb-4"><i class="fas fa-star mr-2"></i>Investment Recommendation</h3>
                    <div id="action" class="text-4xl font-extrabold text-center my-4">-</div>
                    <div id="recommendationDetails" class="space-y-3 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-600">Predicted Return:</span>
                            <span id="predReturn" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Sentiment:</span>
                            <span id="sentiment" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Confidence Score:</span>
                            <span id="confidence" class="font-medium">-</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-600">Target Price:</span>
                            <span id="targetPrice" class="font-medium">-</span>
                        </div>
                    </div>
                </div>

                <!-- News Card -->
                <div class="bg-white rounded-xl shadow-md p-6">
                    <h3 class="font-bold text-lg text-gray-800 mb-4"><i class="fas fa-newspaper mr-2"></i>Latest News</h3>
                    <div id="newsContainer" class="space-y-3 max-h-80 overflow-y-auto">
                        <div class="text-center text-gray-500 py-4">Search for a stock to see related news</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Stock Details -->
        <section id="stockDetails" class="bg-white rounded-xl shadow-md p-6 hidden">
            <h3 class="font-bold text-lg text-gray-800 mb-4"><i class="fas fa-info-circle mr-2"></i>Stock Details</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="detailsGrid">
                <!-- Details will be populated here -->
            </div>
        </section>
    </main>

    <footer class="bg-gray-800 text-white text-center py-6 mt-12">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <p>© 2025 Smart Investment Assistant. Powered by AI and Machine Learning.</p>
            <p class="text-gray-400 text-sm mt-2">This is for educational purposes only. Not financial advice.</p>
        </div>
    </footer>

    <script>
        let selectedTicker = "AAPL";
        let priceChart = null;
        let currentData = null;
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadMarketOverview();
            refreshData();
            setupEventListeners();
        });

        function setupEventListeners() {
            // Search functionality
            document.getElementById('searchBox').addEventListener('input', debounce(handleSearch, 300));
            document.getElementById('refreshBtn').addEventListener('click', refreshData);
            document.getElementById('horizon').addEventListener('change', refreshData);
            document.getElementById('voiceBtn').addEventListener('click', startVoiceRecognition);
        }

        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }

        function speak(text) {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'en-US';
                utterance.volume = 1;
                utterance.rate = 1;
                utterance.pitch = 1;
                window.speechSynthesis.speak(utterance);
            } else {
                console.warn('Text-to-speech not supported in this browser.');
            }
        }

        function startVoiceRecognition() {
            if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
                speak('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
                alert('Speech recognition not supported in this browser. Try Chrome or Edge.');
                return;
            }

            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-US';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;

            recognition.start();
            speak('Listening for your command.');

            recognition.onstart = () => {
                document.getElementById('voiceBtn').classList.add('text-red-500');
            };

            recognition.onend = () => {
                document.getElementById('voiceBtn').classList.remove('text-red-500');
            };

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript.trim().toLowerCase();
                processVoiceCommand(transcript);
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                if (event.error !== 'no-speech') {
                    const errorMsg = 'Error in speech recognition: ' + event.error;
                    speak(errorMsg);
                    alert(errorMsg);
                } else {
                    speak('No speech detected. Please try again.');
                }
            };
        }

        async function processVoiceCommand(transcript) {
            console.log('Voice command:', transcript);

            // Refresh or update data
            if (transcript.includes('refresh') || transcript.includes('update')) {
                speak('Refreshing data.');
                refreshData();
                return;
            }

            // Set prediction horizon
            if (transcript.includes('horizon') || transcript.includes('days')) {
                const match = transcript.match(/(\d+) days/);
                if (match) {
                    const days = parseInt(match[1]);
                    if ([7, 14, 30].includes(days)) {
                        speak(`Setting forecast horizon to ${days} days.`);
                        document.getElementById('horizon').value = days;
                        refreshData();
                        return;
                    } else {
                        speak('Invalid horizon. Please choose 7, 14, or 30 days.');
                        return;
                    }
                }
            }

            // Market overview
            if (transcript.includes('market overview') || transcript.includes('market status')) {
                speak('Loading market overview.');
                loadMarketOverview();
                return;
            }

            // Search for stock (handles "search Apple", "show Apple", "predict Apple", etc.)
            const searchTerms = ['search', 'show', 'predict', 'find', 'lookup', 'stock'];
            if (searchTerms.some(term => transcript.includes(term))) {
                let query = transcript;
                searchTerms.forEach(term => {
                    query = query.replace(term, '').trim();
                });
                if (query) {
                    speak(`Searching for ${query}.`);
                    document.getElementById('searchBox').value = query;
                    await handleSearch({target: {value: query}});
                    // If search returns results, select the first one
                    const results = document.getElementById('searchResults').querySelectorAll('li');
                    if (results.length > 0) {
                        selectedTicker = results[0].dataset.symbol;
                        document.getElementById('companyName').textContent = results[0].dataset.name;
                        document.getElementById('searchBox').value = '';
                        document.getElementById('searchResults').classList.add('hidden');
                        speak(`Displaying data for ${results[0].dataset.name}.`);
                        refreshData();
                    } else {
                        speak('No stocks found for ' + query);
                    }
                    return;
                }
            }

            // Default: treat as stock search
            if (transcript) {
                speak(`Searching for ${transcript}.`);
                document.getElementById('searchBox').value = transcript;
                await handleSearch({target: {value: transcript}});
                const results = document.getElementById('searchResults').querySelectorAll('li');
                if (results.length > 0) {
                    selectedTicker = results[0].dataset.symbol;
                    document.getElementById('companyName').textContent = results[0].dataset.name;
                    document.getElementById('searchBox').value = '';
                    document.getElementById('searchResults').classList.add('hidden');
                    speak(`Displaying data for ${results[0].dataset.name}.`);
                    refreshData();
                } else {
                    speak('No stocks found for ' + transcript);
                }
            } else {
                speak('No command recognized. Please try again.');
            }
        }

        async function handleSearch(e) {
            const query = e.target.value.trim();
            const resultsContainer = document.getElementById('searchResults');
            
            if (!query) {
                resultsContainer.classList.add('hidden');
                return;
            }
            
            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const results = await response.json();
                
                if (results.length > 0) {
                    resultsContainer.innerHTML = results.map(stock => `
                        <li class="px-4 py-3 hover:bg-indigo-50 cursor-pointer border-b border-gray-100 last:border-b-0" 
                            data-symbol="${stock.symbol}" data-name="${stock.name}">
                            <div class="font-medium">${stock.name}</div>
                            <div class="text-sm text-gray-500">${stock.symbol} ${stock.sector ? '• ' + stock.sector : ''}</div>
                        </li>
                    `).join('');
                    
                    resultsContainer.classList.remove('hidden');
                    
                    // Add click event to results
                    resultsContainer.querySelectorAll('li').forEach(item => {
                        item.addEventListener('click', () => {
                            selectedTicker = item.dataset.symbol;
                            document.getElementById('companyName').textContent = item.dataset.name;
                            document.getElementById('searchBox').value = '';
                            resultsContainer.classList.add('hidden');
                            speak(`Displaying data for ${item.dataset.name}.`);
                            refreshData();
                        });
                    });
                } else {
                    resultsContainer.innerHTML = '<li class="px-4 py-3 text-gray-500">No results found</li>';
                    resultsContainer.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Search error:', error);
                speak('Error searching for stocks. Please try again.');
            }
        }

        async function loadMarketOverview() {
            try {
                const response = await fetch('/api/market-overview');
                const data = await response.json();
                
                const container = document.getElementById('marketOverview');
                container.innerHTML = '';
                
                for (const [symbol, info] of Object.entries(data)) {
                    const changeClass = info.change >= 0 ? 'text-green-600' : 'text-red-600';
                    const changeIcon = info.change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
                    
                    container.innerHTML += `
                        <div class="bg-white rounded-lg shadow p-4 flex items-center justify-between">
                            <div>
                                <h4 class="font-medium text-gray-900">${info.name}</h4>
                                <p class="text-2xl font-bold">${info.price}</p>
                            </div>
                            <div class="text-right">
                                <p class="${changeClass} font-medium">
                                    <i class="fas ${changeIcon}"></i> ${Math.abs(info.change)}%
                                </p>
                                <p class="text-sm text-gray-500">${symbol}</p>
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Market overview error:', error);
                speak('Error loading market overview.');
            }
        }

        async function refreshData() {
            const refreshBtn = document.getElementById('refreshBtn');
            const horizon = parseInt(document.getElementById('horizon').value);
            
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            
            try {
                const [priceData, recommendation, news] = await Promise.all([
                    fetch(`/api/data?ticker=${selectedTicker}&n=180`).then(r => r.json()),
                    fetch('/api/recommend', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ticker: selectedTicker, horizon: horizon})
                    }).then(r => r.json()),
                    fetch(`/api/news/${selectedTicker}`).then(r => r.json())
                ]);
                
                currentData = priceData;
                updateChart(priceData, recommendation);
                updateRecommendation(recommendation);
                updateNews(news);
                updateStockDetails(priceData.info);
                
                speak(`Data refreshed for ${document.getElementById('companyName').textContent}.`);
                
            } catch (error) {
                console.error('Error refreshing data:', error);
                speak('Failed to load data. Please try again.');
                alert('Failed to load data. Please try again.');
            } finally {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh';
                document.getElementById('lastUpdated').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
            }
        }

        function updateChart(priceData, recommendation) {
            const prices = priceData.prices || [];
            const preds = recommendation.predictions || [];
            
            if (prices.length === 0) return;
            
            const labels = prices.map(p => p.date);
            const priceValues = prices.map(p => p.close);
            
            // Generate prediction dates
            const lastDate = new Date(prices[prices.length - 1].date);
            const predDates = [];
            for (let i = 1; i <= preds.length; i++) {
                const nextDate = new Date(lastDate);
                nextDate.setDate(nextDate.getDate() + i);
                predDates.push(nextDate.toISOString().split('T')[0]);
            }
            
            const allLabels = [...labels, ...predDates];
            const historicalData = [...priceValues, ...Array(preds.length).fill(null)];
            const predictionData = [...Array(priceValues.length).fill(null), ...preds];
            
            if (priceChart) {
                priceChart.destroy();
            }
            
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: allLabels,
                    datasets: [
                        {
                            label: 'Historical Price',
                            data: historicalData,
                            borderColor: 'rgb(79, 70, 229)',
                            backgroundColor: 'rgba(79, 70, 229, 0.1)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 0,
                            borderWidth: 2
                        },
                        {
                            label: 'Prediction',
                            data: predictionData,
                            borderColor: 'rgb(16, 185, 129)',
                            borderDash: [5, 5],
                            tension: 0.3,
                            pointRadius: 3,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                usePointStyle: true,
                                padding: 20
                            }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    return `${context.dataset.label}: $${context.raw.toFixed(2)}`;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                display: false
                            },
                            ticks: {
                                maxRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 8
                            }
                        },
                        y: {
                            grid: {
                                color: 'rgba(0, 0, 0, 0.05)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return '$' + value;
                                }
                            }
                        }
                    }
                }
            });
            
            // Update price change
            const firstPrice = priceValues[0];
            const lastPrice = priceValues[priceValues.length - 1];
            const change = ((lastPrice - firstPrice) / firstPrice) * 100;
            const changeElem = document.getElementById('priceChange');
            
            if (change >= 0) {
                changeElem.innerHTML = `<span class="text-green-600"><i class="fas fa-arrow-up"></i> ${change.toFixed(2)}%</span>`;
            } else {
                changeElem.innerHTML = `<span class="text-red-600"><i class="fas fa-arrow-down"></i> ${change.toFixed(2)}%</span>`;
            }
        }

        function updateRecommendation(data) {
            if (data.error) {
                document.getElementById('action').textContent = 'Error';
                document.getElementById('recommendationDetails').innerHTML = `<div class="text-red-500">${data.error}</div>`;
                speak('Error in recommendation: ' + data.error);
                return;
            }
            
            // Update action with color coding
            const action = document.getElementById('action');
            action.textContent = data.action;
            
            switch(data.action) {
                case 'Strong Buy':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-green-700';
                    break;
                case 'Buy':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-green-500';
                    break;
                case 'Hold':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-yellow-500';
                    break;
                case 'Sell':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-orange-500';
                    break;
                case 'Strong Sell':
                    action.className = 'text-4xl font-extrabold text-center my-4 text-red-700';
                    break;
                default:
                    action.className = 'text-4xl font-extrabold text-center my-4 text-gray-500';
            }
            
            // Update details
            document.getElementById('predReturn').textContent = 
                `${data.predicted_return_pct >= 0 ? '+' : ''}${data.predicted_return_pct}%`;
            document.getElementById('predReturn').className = 
                `font-medium ${data.predicted_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`;
                
            document.getElementById('sentiment').textContent = 
                `${data.sentiment_label} (${data.sentiment_score})`;
            document.getElementById('sentiment').className = 
                `font-medium ${data.sentiment_score >= 0 ? 'text-green-600' : 'text-red-600'}`;
                
            document.getElementById('confidence').textContent = `${data.score}`;
            document.getElementById('targetPrice').textContent = `$${data.target_price}`;
        }

        function updateNews(data) {
            const container = document.getElementById('newsContainer');
            
            if (data.error || !data.news || data.news.length === 0) {
                container.innerHTML = '<div class="text-center text-gray-500 py-4">No news available</div>';
                speak('No news available for this stock.');
                return;
            }
            
            container.innerHTML = data.news.slice(0, 5).map(item => `
                <div class="p-3 bg-gray-50 rounded-lg border border-gray-100">
                    <p class="text-sm text-gray-800">${item}</p>
                </div>
            `).join('');
        }

        function updateStockDetails(info) {
            const detailsSection = document.getElementById('stockDetails');
            const detailsGrid = document.getElementById('detailsGrid');
            
            if (!info || Object.keys(info).length === 0) {
                detailsSection.classList.add('hidden');
                speak('No stock details available.');
                return;
            }
            
            detailsSection.classList.remove('hidden');
            
            detailsGrid.innerHTML = `
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Sector</div>
                    <div class="font-medium">${info.sector || 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Market Cap</div>
                    <div class="font-medium">${formatMarketCap(info.marketCap)}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">P/E Ratio</div>
                    <div class="font-medium">${info.peRatio ? info.peRatio.toFixed(2) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Dividend Yield</div>
                    <div class="font-medium">${info.dividendYield ? info.dividendYield.toFixed(2) + '%' : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">52W High</div>
                    <div class="font-medium">${info['52WeekHigh'] ? '$' + info['52WeekHigh'].toFixed(2) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">52W Low</div>
                    <div class="font-medium">${info['52WeekLow'] ? '$' + info['52WeekLow'].toFixed(2) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Volume</div>
                    <div class="font-medium">${info.volume ? formatNumber(info.volume) : 'N/A'}</div>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <div class="text-gray-500 text-sm">Avg Volume</div>
                    <div class="font-medium">${info.avgVolume ? formatNumber(info.avgVolume) : 'N/A'}</div>
                </div>
            `;
        }

        function formatMarketCap(value) {
            if (!value) return 'N/A';
            if (value >= 1e12) return '$' + (value / 1e12).toFixed(2) + 'T';
            if (value >= 1e9) return '$' + (value / 1e9).toFixed(2) + 'B';
            if (value >= 1e6) return '$' + (value / 1e6).toFixed(2) + 'M';
            return '$' + value.toFixed(2);
        }

        function formatNumber(num) {
            if (!num) return 'N/A';
            return num.toLocaleString();
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")

if __name__ == "__main__":
    try:   
        from waitress import serve
        log.info("Running with Waitress on http://127.0.0.1:8000")
        serve(app, host="0.0.0.0", port=8000)
    except ImportError:
        log.info("Waitress not installed; using Flask dev server")
        app.run(host="0.0.0.0", port=8000, debug=True)
