import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import os
from datetime import datetime

# --- Improved Data Fetching (Bypasses Large File Virus Warnings) ---
def get_gdrive_download_url(url):
    """Converts a Google Drive URL into a direct download link with a large-file bypass."""
    try:
        file_id = None
        if '/file/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        elif 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        
        if file_id:
            # Adding &confirm=t forces Google to bypass the virus scan warning for files > 100MB
            return f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
        return url
    except Exception:
        return url

# --- Logic Constants (Synced with Source of Truth: divergence_make_dashboard.py) ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA_PERIOD = 8
EMA21_PERIOD = 21

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Divergence Scanner", layout="wide")
st.title("📈 RSI Price Divergence Scanner")

# Dataset Selection Sidebar
data_option = st.sidebar.selectbox(
    "Select Dataset to Analyze",
    ("Divergences Data", "S&P 500 Data")
)

# Accessing secrets
try:
    if data_option == "Divergences Data":
        raw_url = st.secrets["URL_DIVERGENCES"]
    else:
        raw_url = st.secrets["URL_SP500"]
    DATA_URL = get_gdrive_download_url(raw_url)
except KeyError:
    st.error("Secrets not found. Please ensure URL_DIVERGENCES and URL_SP500 are set in Streamlit Secrets.")
    st.stop()

# --- Core Functions ---

@st.cache_data(ttl=3600) # Cache data for 1 hour to handle large 160MB file
def load_and_prepare_all_data(url):
    """Downloads and cleans the master dataset."""
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    df = pd.read_csv(StringIO(response.text))
    
    # Identify Ticker Column (Matches 'Ticker', 'SYMBOL', etc.)
    t_col = None
    for col in df.columns:
        if col.strip().upper() in ['TICKER', 'SYMBOL', 'SYM', 'CODE']:
            t_col = col
            break
    
    if not t_col:
        return None
    
    return df, t_col

def prepare_ticker_data(df):
    """Clean and map columns for a single ticker. Sync with SOT logic."""
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    w_close_col, w_vol_col, w_rsi_col = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_ema8_col, w_ema21_col = 'W_EMA_8', 'W_EMA_21'
    w_high_col, w_low_col = 'W_HIGH', 'W_LOW'

    if not all([date_col, close_col, vol_col, high_col, low_col]):
        return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Daily
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={
        close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low',
        d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'
    }, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])

    # Weekly
    df_w = df[[w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col, w_ema8_col, w_ema21_col]].copy()
    df_w.rename(columns={
        w_close_col: 'Price', w_vol_col: 'Volume', w_high_col: 'High', w_low_col: 'Low',
        w_rsi_col: 'RSI', w_ema8_col: 'EMA8', w_ema21_col: 'EMA21'
    }, inplace=True)
    df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
    df_w = df_w.dropna(subset=['Price', 'RSI'])
    
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    """Detects RSI divergences matching SOT logic."""
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences

    def get_date_str(point):
        return df_tf.loc[point.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else point.name.strftime('%Y-%m-%d')
            
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        # Bullish Divergence
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                # RSI Threshold Check (Between P1 and P2)
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                    # Post-Signal Invalidation
                    post_signal_df = df_tf.iloc[i + 1 :]
                    if not (not post_signal_df.empty and (post_signal_df['RSI'] <= p1['RSI']).any()):
                        tags = []
                        if p2['Price'] >= p2['EMA8']: tags.append(f"EMA{EMA_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"
                        })

        # Bearish Divergence
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                # RSI Threshold Check
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                    # Post-Signal Invalidation
                    post_signal_df = df_tf.iloc[i + 1 :]
                    if not (not post_signal_df.empty and (post_signal_df['RSI'] >= p1['RSI']).any()):
                        tags = []
                        if p2['Price'] <= p2['EMA21']: tags.append(f"EMA{EMA21_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['High']:,.2f}", 'P2 Price': f"${p2['High']:,.2f}"
                        })
    return divergences

# --- Execution ---

st.info(f"Downloading dataset (approx 160MB)... Please wait.")
data_result = load_and_prepare_all_data(DATA_URL)

if data_result:
    master, t_col = data_result
    tickers = master[t_col].unique()
    raw_results = []
    
    progress_text = "Scanning tickers for divergences..."
    progress_bar = st.progress(0, text=progress_text)

    # Optimize by grouping rather than repetitive filtering
    grouped = master.groupby(t_col)
    for i, (ticker, group) in enumerate(grouped):
        d_d, d_w = prepare_ticker_data(group.copy())
        if d_d is not None:
            raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
        if d_w is not None:
            raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
        progress_bar.progress((i + 1) / len(tickers))

    if raw_results:
        res_df = pd.DataFrame(raw_results)
        res_df = res_df.sort_values(by='Signal Date', ascending=False)
        # Keeps latest Bullish AND latest Bearish for each ticker (NFLX fix)
        consolidated_df = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
        
        for timeframe in ['Daily', 'Weekly']:
            st.markdown(f"## {timeframe} Analysis")
            cols = st.columns(2)
            for idx, (s_type, emoji) in enumerate([('Bullish', '🟢'), ('Bearish', '🔴')]):
                with cols[idx]:
                    st.markdown(f"### {emoji} {s_type}")
                    tbl_df = consolidated_df[(consolidated_df['Type'] == s_type) & (consolidated_df['Timeframe'] == timeframe)]
                    if not tbl_df.empty:
                        st.table(tbl_df.drop(columns=['Type', 'Timeframe']))
                    else:
                        st.write(f"No {timeframe.lower()} {s_type.lower()} signals.")
    else:
        st.write("No divergences found in the lookback period.")
else:
    st.error("Failed to load or parse data. Check your Google Drive permissions.")
