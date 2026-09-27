"""
Crypto Signal Dashboard -- Clean Streamlit Version
"""
import time
import requests
import feedparser
import pandas as pd
import streamlit as st

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Crypto Signal Dashboard", layout="wide")

st.title("🚀 Crypto Signal Dashboard & Screener")
st.write("A clean rules-based crypto screener powered by Binance & CoinGecko APIs.")

# ==================== CONFIGURATION ====================
WATCHLIST = [
    "BTCUSDT",
    "XLMUSDT",
    "INJUSDT",
    "AEROUSDT",
]

TIMEFRAME = "4h"
CANDLE_LOOKBACK = 200

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
]
NEWS_LOOKBACK_HOURS = 24


# ==================== DATA SOURCES ====================
@st.cache_data(ttl=300)
def get_klines(symbol: str, interval: str = "4h", limit: int = 200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    raw = r.json()
    return [{"open_time": c[0], "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]), "close_time": c[6]} for c in raw]

@st.cache_data(ttl=300)
def get_top_gainers():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 50, "page": 1, "price_change_percentage": "24h"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    coins = [c for c in r.json() if c.get("price_change_percentage_24h") is not None]
    coins.sort(key=lambda c: c["price_change_percentage_24h"], reverse=True)
    return coins

def fetch_recent_headlines():
    cutoff = time.time() - (NEWS_LOOKBACK_HOURS * 3600)
    items = []
    for feed_url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            continue
        for entry in parsed.entries:
            published = entry.get("published_parsed")
            ts = time.mktime(published) if published else time.time()
            if ts < cutoff:
                continue
            items.append({"title": entry.get("title", ""), "link": entry.get("link", ""), "published_ts": ts})
    return items

def headlines_for_coin(all_headlines, coin_symbol: str):
    return [h for h in all_headlines if coin_symbol.lower() in h["title"].lower()]

def score_sentiment(headlines):
    bullish_words = ["surge", "rally", "soar", "breakout", "bullish", "record high", "adoption", "upgrade", "partnership", "approval", "inflow", "accumulate", "buy"]
    bearish_words = ["crash", "dump", "plunge", "bearish", "hack", "exploit", "lawsuit", "ban", "outflow", "sell-off", "selloff", "liquidation", "delist", "fraud"]
    
    bull = sum(1 for h in headlines for w in bullish_words if w in h["title"].lower())
    bear = sum(1 for h in headlines for w in bearish_words if w in h["title"].lower())
    return bull - bear


# ==================== ANALYSIS & INDICATORS ====================
def ema(series: pd.Series, period: int):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean().replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_trend(df: pd.DataFrame):
    fast = ema(df["close"], EMA_FAST)
    slow = ema(df["close"], EMA_SLOW)
    recent = df.tail(10)
    if fast.iloc[-1] > slow.iloc[-1] and recent["high"].iloc[-1] > recent["high"].iloc[0]:
        return "uptrend"
    if fast.iloc[-1] < slow.iloc[-1] and recent["high"].iloc[-1] < recent["high"].iloc[0]:
        return "downtrend"
    return "ranging"

def build_signal(candles, news_sentiment_score: int = 0):
    df = pd.DataFrame(candles)
    if len(df) < max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD) + 5:
        return {"error": "not enough candles"}

    trend = detect_trend(df)
    rsi_val = float(rsi(df["close"], RSI_PERIOD).iloc[-1])
    atr_val = float(atr(df, ATR_PERIOD).iloc[-1])
    last_close = float(df["close"].iloc[-1])

    score = 0
    reasons = []

    if trend == "uptrend":
        score += 2
        reasons.append("uptrend structure confirmed")
    elif trend == "downtrend":
        score -= 2
        reasons.append("downtrend structure confirmed")
    else:
        reasons.append("ranging market structure")

    if rsi_val < 30:
        score += 1
        reasons.append(f"RSI {rsi_val:.1f} oversold")
    elif rsi_val > 70:
        score -= 1
        reasons.append(f"RSI {rsi_val:.1f} overbought")

    if news_sentiment_score > 0:
        score += 1
    elif news_sentiment_score < 0:
        score -= 1

    if score >= 2:
        bias = "BUY"
    elif score <= -2:
        bias = "SELL"
    else:
        bias = "HOLD"

    sl, tp = None, None
    if bias == "BUY":
        sl = round(last_close - (atr_val * ATR_SL_MULTIPLIER), 6)
        tp = round(last_close + (atr_val * ATR_TP_MULTIPLIER), 6)
    elif bias == "SELL":
        sl = round(last_close + (atr_val * ATR_SL_MULTIPLIER), 6)
        tp = round(last_close - (atr_val * ATR_TP_MULTIPLIER), 6)

    return {
        "bias": bias,
        "score": score,
        "last_close": last_close,
        "trend": trend,
        "rsi": round(rsi_val, 2),
        "suggested_sl": sl,
        "suggested_tp": tp,
        "reasons": reasons,
    }


# ==================== WEB INTERFACE ====================
st.subheader("📊 Watchlist Signals")
if st.button("Run Analysis"):
    with st.spinner("Analyzing market data..."):
        all_headlines = fetch_recent_headlines()
        table_data = []

        for symbol in WATCHLIST:
            try:
                candles = get_klines(symbol, TIMEFRAME, CANDLE_LOOKBACK)
                coin_symbol = symbol.replace("USDT", "")
                matches = headlines_for_coin(all_headlines, coin_symbol)
                sentiment_score = score_sentiment(matches)

                res = build_signal(candles, sentiment_score)
                if "error" not in res:
                    table_data.append({
                        "Pair": symbol,
                        "Signal": res["bias"],
                        "Trend": res["trend"],
                        "RSI": res["RSI"] if "RSI" in res else res["rsi"],
                        "Close Price": res["last_close"],
                        "Stop Loss": res["suggested_sl"] if res["suggested_sl"] else "-",
                        "Take Profit": res["suggested_tp"] if res["suggested_tp"] else "-"
                    })
            except Exception as e:
                st.error(f"Error for {symbol}: {e}")

        if table_data:
            df_res = pd.DataFrame(table_data)
            st.dataframe(df_res, use_container_width=True)

st.subheader("🔥 Top 24h Gainers (Pump Scanner)")
if st.button("Scan Pumps"):
    with st.spinner("Scanning CoinGecko gainers..."):
        gainers = get_top_gainers()
        pump_data = []
        for c in gainers:
            change = c.get("price_change_percentage_24h", 0)
            if change >= 10.0:
                pump_data.append({
                    "Symbol": c.get("symbol", "").upper() + "USDT",
                    "Name": c.get("name"),
                    "24h Change (%)": round(change, 2),
                    "Current Price": c.get("current_price")
                })
        if pump_data:
            df_pumps = pd.DataFrame(pump_data)
            st.dataframe(df_pumps, use_container_width=True)
        else:
            st.info("No coins meeting pump threshold.")
              
