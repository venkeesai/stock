import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import feedparser
from transformers import pipeline
from tvDatafeed import TvDatafeed, Interval
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. APP CONFIGURATION & FINTECH UI
# ==========================================
st.set_page_config(page_title="Index Options AI Terminal", layout="wide", initial_sidebar_state="collapsed")

css = """<style>
.stApp { background-color: #0b0e14; color: #d1d4dc; font-family: 'Inter', sans-serif; }
.top-bar { display: flex; justify-content: space-between; align-items: center; background-color: #131722; padding: 15px 25px; border-bottom: 1px solid #2a2e39; margin-top: -50px; margin-bottom: 20px; border-radius: 0 0 8px 8px; }
.top-bar-title { font-size: 20px; font-weight: bold; color: #ffffff; display: flex; align-items: center; gap: 10px;}
.live-badge { background-color: #2196f3; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; animation: pulse 2s infinite;}
.panel { background-color: #131722; border: 1px solid #2a2e39; border-radius: 8px; padding: 20px; height: 100%; }
div[data-testid="stSelectbox"] label, div[data-testid="stCheckbox"] label { color: #8a93a6; font-size: 14px; }
div[data-baseweb="select"] { background-color: #1e222d; border: 1px solid #2a2e39; }
div[data-testid="stButton"] button { background-color: #2196f3; color: white; border: none; width: 100%; font-weight: bold; padding: 10px; border-radius: 6px; transition: 0.3s; }
div[data-testid="stButton"] button:hover { background-color: #1976d2; }
.insight-row { display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #2a2e39; padding-bottom: 5px;}
.buy-text { color: #26a69a; font-weight: bold; }
.sell-text { color: #ef5350; font-weight: bold; }
.neutral-text { color: #d1d4dc; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
</style>"""
st.markdown(css, unsafe_allow_html=True)

# Strictly Indian Indices
INDICES = {
    "NIFTY 50": {"tv": ("NIFTY", "NSE"), "yf": "^NSEI"},
    "NIFTY BANK": {"tv": ("BANKNIFTY", "NSE"), "yf": "^NSEBANK"},
    "BSE SENSEX": {"tv": ("SENSEX", "BSE"), "yf": "^BSESN"},
    "NIFTY IT": {"tv": ("CNXIT", "NSE"), "yf": "^CNXIT"},
    "FINNIFTY": {"tv": ("FINNIFTY", "NSE"), "yf": "NIFTY_FIN_SERVICE.NS"},
    "NIFTY MIDCAP 50": {"tv": ("NIFTY_MIDCAP_50", "NSE"), "yf": "^NSEMDCP50"}
}

# ==========================================
# 2. ENGINES
# ==========================================
@st.cache_resource
def load_finbert_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

@st.cache_data(ttl=300)
def analyze_market_sentiment(query):
    safe_query = query.replace(' ', '%20')
    feed_url = f"https://news.google.com/rss/search?q={safe_query}+india+market&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(feed_url)
        finbert = load_finbert_model()
        sentiments, news_items = [], []
        for entry in feed.entries[:10]:
            title = entry.title
            result = finbert(title)[0]
            label = result['label']
            confidence = result['score']
            if label == 'positive': score = confidence
            elif label == 'negative': score = -confidence
            else: score = 0
            sentiments.append(score)
            news_items.append({"title": title, "score": score, "label": label.upper(), "confidence": confidence})
        return np.mean(sentiments) if sentiments else 0, news_items
    except Exception:
        return 0, [{"title": "News Feed Unavailable", "score": 0, "label": "NEUTRAL", "confidence": 0}]

def find_support_resistance(df, window=10):
    supports, resistances = [], []
    if df is None or len(df) < window * 2: return supports, resistances
    for i in range(window, len(df) - window):
        if df['Low'].iloc[i] == min(df['Low'].iloc[i-window:i+window]):
            supports.append((df['Date'].iloc[i], df['Low'].iloc[i]))
        elif df['High'].iloc[i] == max(df['High'].iloc[i-window:i+window]):
            resistances.append((df['Date'].iloc[i], df['High'].iloc[i]))
    return supports, resistances

@st.cache_resource
def get_tv_connection():
    try: return TvDatafeed()
    except: return None

@st.cache_data(ttl=300)
def fetch_and_train_lstm(tv_data, yf_data, timeframe_key):
    df = None
    data_source = "TradingView"
    tv_symbol, tv_exchange = tv_data
    
    try:
        tv = get_tv_connection()
        if tv is not None:
            tv_intervals = {"15 Minute": Interval.in_15_minute, "30 Minute": Interval.in_30_minute, "1 Hour": Interval.in_1_hour}
            interval = tv_intervals.get(timeframe_key, Interval.in_15_minute)
            temp_df = tv.get_hist(symbol=tv_symbol, exchange=tv_exchange, interval=interval, n_bars=500)
            if temp_df is not None and not temp_df.empty:
                df = temp_df.reset_index()
                df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    except Exception:
        pass

    if df is None or df.empty:
        data_source = "Yahoo Finance (Fallback)"
        yf_tf = {"15 Minute": "15m", "30 Minute": "30m", "1 Hour": "1h"}
        df = yf.download(yf_data, period="60d", interval=yf_tf.get(timeframe_key, "15m"), progress=False)
        if df.empty: return None, None, None
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.rename(columns={'Datetime': 'Date', 'index': 'Date'}, inplace=True)

    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df.dropna(inplace=True)

    features = ['Close', 'RSI', 'EMA_9', 'EMA_21']
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    time_step = 60 
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), :])
        y.append(scaled_data[i + time_step, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='huber')
    model.fit(X, y, batch_size=32, epochs=5, verbose=0) 

    last_60_candles = scaled_data[-time_step:]
    X_live = np.array([last_60_candles])
    predicted_scaled = model.predict(X_live, verbose=0)
    
    dummy = np.zeros((1, len(features)))
    dummy[0, 0] = predicted_scaled[0][0]
    raw_predicted_price = scaler.inverse_transform(dummy)[0, 0]

    return df, raw_predicted_price, data_source

# ==========================================
# 3. UI LAYOUT & DASHBOARD
# ==========================================
top_bar = """
<div class="top-bar">
<div class="top-bar-title">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2196f3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
Index Options AI Terminal
<span class="live-badge">DUAL-DATA ENGINE</span>
</div>
<div style="color: #8a93a6; font-size: 14px;">System: <span style="color: #2196f3;">Online</span></div>
</div>
"""
st.markdown(top_bar, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 3.5, 1.5])

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    
    selected_asset = st.selectbox("Select Index", list(INDICES.keys()))
    
    tf_mapping = {"15 Minute": "15m", "30 Minute": "30m", "1 Hour": "1h"}
    selected_tf = st.selectbox("AI Timeframe", list(tf_mapping.keys()), index=0)
    
    st.markdown("---")
    st.markdown("<div style='color: #8a93a6; font-size: 12px; margin-bottom: 5px; text-transform: uppercase;'>Chart Overlays</div>", unsafe_allow_html=True)
    show_emas = st.checkbox("Show Trend EMAs", value=False)
    show_sr = st.checkbox("Show Topo Support/Resistance", value=True)
    show_signals = st.checkbox("Show EMA Crossovers", value=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Initialize Master Engine")
    st.markdown('</div>', unsafe_allow_html=True)

if analyze_btn:
    tv_data = INDICES[selected_asset]["tv"]
    yf_data = INDICES[selected_asset]["yf"]
    
    with st.spinner(f"Brain 1: Establishing Link to {selected_asset}..."):
        df, raw_predicted_price, data_source = fetch_and_train_lstm(tv_data, yf_data, selected_tf)
        
    if df is None:
        st.error("Critical Failure: Both TradingView and Yahoo Finance data connections failed. Check network.")
    else:
        with st.spinner(f"Brain 2: FinBERT contextualizing live news for {selected_asset}..."):
            sentiment_score, news_headlines = analyze_market_sentiment(selected_asset)
            sentiment_multiplier = 1 + (sentiment_score * 0.001) 
            final_predicted_price = raw_predicted_price * sentiment_multiplier

        with st.spinner(f"Brain 3: Vision Engine mapping structural topology..."):
            supports, resistances = find_support_resistance(df)

            with col2:
                plot_df = df.tail(100).copy()
                current_price = plot_df['Close'].iloc[-1]
                
                fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

                # Pure Price Action Candlesticks
                fig.add_trace(go.Candlestick(x=plot_df['Date'], open=plot_df['Open'], high=plot_df['High'],
                                             low=plot_df['Low'], close=plot_df['Close'], name='Price',
                                             increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
                
                # TOGGLE LOGIC: Historical EMA Crossovers (Labeled clearly as CE/PE Entry)
                if show_signals:
                    plot_df['Signal'] = 0
                    plot_df.loc[(plot_df['EMA_9'] > plot_df['EMA_21']) & (plot_df['EMA_9'].shift(1) <= plot_df['EMA_21'].shift(1)), 'Signal'] = 1
                    plot_df.loc[(plot_df['EMA_9'] < plot_df['EMA_21']) & (plot_df['EMA_9'].shift(1) >= plot_df['EMA_21'].shift(1)), 'Signal'] = -1
                    
                    buys = plot_df[plot_df['Signal'] == 1]
                    sells = plot_df[plot_df['Signal'] == -1]
                    
                    fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Low'] * 0.998, mode='markers', 
                                             marker=dict(symbol='triangle-up', size=14, color='#26a69a', line=dict(width=1, color='white')), 
                                             name='CE Entry (EMA Cross)'))
                    fig.add_trace(go.Scatter(x=sells['Date'], y=sells['High'] * 1.002, mode='markers', 
                                             marker=dict(symbol='triangle-down', size=14, color='#ef5350', line=dict(width=1, color='white')), 
                                             name='PE Entry (EMA Cross)'))

                # TOGGLE LOGIC: EMAs
                if show_emas:
                    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['EMA_9'], line=dict(color='rgba(41, 98, 255, 0.6)', width=1.5), name='EMA 9'))
                    fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['EMA_21'], line=dict(color='rgba(246, 195, 67, 0.6)', width=1.5), name='EMA 21'))

                # TOGGLE LOGIC: Support/Resistance
                if show_sr:
                    for res in resistances[-2:]: 
                        fig.add_hline(y=res[1], line_dash="dot", line_color="rgba(239, 83, 80, 0.4)", annotation_text="Res", annotation_position="top left")
                    for sup in supports[-2:]: 
                        fig.add_hline(y=sup[1], line_dash="dot", line_color="rgba(38, 166, 154, 0.4)", annotation_text="Sup", annotation_position="bottom left")

                minutes_to_add = int(tf_mapping[selected_tf].replace('m', '').replace('h', '60'))
                next_time = plot_df['Date'].iloc[-1] + pd.Timedelta(minutes=minutes_to_add)
                pred_color = '#26a69a' if final_predicted_price > current_price else '#ef5350'
                
                # Highlighted AI Projection line
                fig.add_trace(go.Scatter(x=[plot_df['Date'].iloc[-1], next_time], y=[current_price, final_predicted_price], 
                                         mode='lines+markers', line=dict(color='#2196f3', width=2, dash='dash'), 
                                         marker=dict(size=10, color=pred_color), name='Fused AI Target'))

                fig.update_layout(title=f"{selected_asset} | Target: {final_predicted_price:.2f} | Source: {data_source}", 
                                  template="plotly_dark", paper_bgcolor='#131722', plot_bgcolor='#131722',
                                  margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False, height=650, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col3:
                is_bullish = final_predicted_price > current_price
                diff_val = final_predicted_price - current_price
                
                action_text = "BUY CALL (CE)" if is_bullish else "BUY PUT (PE)"
                move_text = f"{'+' if is_bullish else ''}{diff_val:.2f} pts"
                target = final_predicted_price + abs(diff_val * 0.5) if is_bullish else final_predicted_price - abs(diff_val * 0.5)
                stop_loss = current_price - abs(diff_val * 0.8) if is_bullish else current_price + abs(diff_val * 0.8)
                tp_text = "Take Profit (CE/PE)"

                bg_color = "rgba(38, 166, 154, 0.1)" if is_bullish else "rgba(239, 83, 80, 0.1)"
                score_color = '#26a69a' if is_bullish else '#ef5350'
                
                if sentiment_score > 0.2: sentiment_text = "Bullish Greed"
                elif sentiment_score < -0.2: sentiment_text = "Bearish Fear"
                else: sentiment_text = "Neutral"

                html_parts = [
                    "<div class='panel'>",
                    f"<div style='background-color: {bg_color}; padding: 15px; text-align: center; border-radius: 6px; margin-bottom: 20px; border: 1px solid {score_color};'>",
                    "<div style='color: #ffffff; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;'>Fused Master Signal</div>",
                    f"<div style='font-size: 26px; font-weight: 900; color: {score_color}; letter-spacing: 1px;'>{action_text}</div>",
                    "</div>",
                    f"<div style='font-size: 32px; font-weight: bold; margin-bottom: 5px; color: #ffffff;'>{current_price:.2f}</div>",
                    "<div style='color: #8a93a6; font-size: 14px; margin-bottom: 20px;'>Current Level</div>",
                    "<div class='insight-row'><span class='neutral-text'>FinBERT Sentiment</span>",
                    f"<span style='color: #2196f3; font-weight: bold;'>{sentiment_text}</span></div>",
                    "<div class='insight-row'><span class='neutral-text'>Fused Target Move</span>",
                    f"<span style='color: {score_color}; font-weight: bold;'>{move_text}</span></div>",
                    "<hr style='border-color: #2a2e39; margin: 20px 0;'>",
                    "<h4 style='color:#8a93a6; margin-bottom: 10px; font-size: 14px;'>Suggested Setup</h4>",
                    f"<div class='insight-row'><span class='neutral-text'>{tp_text}</span>",
                    f"<span class='buy-text'>{target:.2f}</span></div>",
                    "<div class='insight-row'><span class='neutral-text'>Hard Stop Loss</span>",
                    f"<span class='sell-text'>{stop_loss:.2f}</span></div>",
                    "<hr style='border-color: #2a2e39; margin: 20px 0;'>",
                    "<h4 style='color:#8a93a6; margin-bottom: 5px; font-size: 12px;'>FinBERT Read Feed:</h4>"
                ]
                
                for news in news_headlines[:3]:
                    color = "#26a69a" if news['score'] > 0 else "#ef5350" if news['score'] < 0 else "#8a93a6"
                    html_parts.append(
                        f"<div style='font-size: 11px; margin-bottom: 6px; line-height: 1.3; color: #d1d4dc;'>"
                        f"• {news['title'][:50]}...<br>"
                        f"<span style='color: {color}; font-weight: bold; margin-left: 10px;'>[{news['label']} - {news['confidence']*100:.1f}%]</span>"
                        f"</div>"
                    )
                
                html_parts.append("</div>")
                st.markdown("".join(html_parts), unsafe_allow_html=True)