import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from datetime import datetime

# --- Advanced Data Fetching (Bypasses Google Drive Token Handshake) ---
def get_confirmed_gdrive_data(url):
    """Handles the cookie-based confirmation required for files >100MB."""
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        
        # Pass 1: Get the warning page and the 'download_warning' cookie
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # Pass 2: If a token exists, request again with the 'confirm' parameter
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(download_url, params=params, stream=True)
        
        if response.status_code == 200:
            return StringIO(response.text)
        else:
            return None
    except Exception as e:
        st.error(f"Fetch Error: {e}")
        return None

# --- Logic Constants (Source of Truth) ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA_PERIOD = 8
EMA21_PERIOD = 21

# --- Streamlit UI ---
st.set_page_config(page_title="RSI Divergence Scanner", layout="wide")
st.title("📈 RSI Divergence Scanner")

data_option = st.sidebar.selectbox("Select Dataset", ("Divergences Data", "S&P 500 Data"))

try:
    target_url = st.secrets["URL_DIVERGENCES"] if data_option == "Divergences Data" else st.secrets["URL_SP500"]
except KeyError:
    st.error("Secrets not found. Check your Streamlit Secrets configuration.")
    st.stop()

# --- Logic Functions ---

def prepare_data(df):
    """Sync with Source of Truth prepare_data logic."""
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    # Required SOT columns
    d_rsi, d_ema8, d_ema21 = 'RSI_14', 'EMA_8', 'EMA_21'
    w_close, w_vol, w_rsi = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'

    if not all([date_col, close_col, vol_col, high_col, low_col]):
        return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Daily
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi, d_ema8, d_ema21]].copy()
    df_d.rename(columns={close_col:'Price', vol_col:'Volume', high_col:'High', low_col:'Low', d_rsi:'RSI', d_ema8:'EMA8', d_ema21:'EMA21'}, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])

    # Weekly
    df_w = df[[w_close, w_vol, w_rsi]].copy() # Standardized Weekly
    df_w.rename(columns={w_close:'Price', w_vol:'Volume', w_rsi:'RSI'}, inplace=True)
    # Using Price for High/Low on weekly if columns aren't explicit in snippet
    df_w['High'] = df_w['Price']
    df_w['Low'] = df_w['Price']
    df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
    df_w = df_w.dropna(subset=['Price', 'RSI'])
    
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    """Detects RSI divergences with invalidation checks."""
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences

    def get_date_str(p):
        return df_tf.loc[p.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else p.name.strftime('%Y-%m-%d')
            
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        # Bullish
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                # In-between RSI validation
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                    # Post-signal invalidation check
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] <= p1['RSI']).any()):
                        tags = []
                        if 'EMA8' in p2 and p2['Price'] >= p2['EMA8']: tags.append(f"EMA{EMA_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe, 'Tags': ", ".join(tags), 'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2), 'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}", 'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"})

        # Bearish
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] >= p1['RSI']).any()):
                        tags = []
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROWTH")
                        divergences.append({'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe, 'Tags': ", ".join(tags), 'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2), 'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}", 'P1 Price': f"${p1['High']:,.2f}", 'P2 Price': f"${p2['High']:,.2f}"})
    return divergences

# --- Main App Execution ---

st.info(f"Connecting to {data_option}...")
csv_buffer = get_confirmed_gdrive_data(target_url)

if csv_buffer:
    try:
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
        
        if not t_col:
            st.error("No Ticker column found.")
            st.stop()

        raw_results = []
        progress_bar = st.progress(0, text="Processing Tickers...")
        
        grouped = master.groupby(t_col)
        total_tickers = len(grouped)
        
        for i, (ticker, group) in enumerate(grouped):
            d_d, d_w = prepare_data(group.copy())
            if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
            if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
            progress_bar.progress((i + 1) / total_tickers)

        if raw_results:
            res_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
            consolidated = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
            
            for tf in ['Daily', 'Weekly']:
                st.markdown(f"---")
                st.header(f"📅 {tf} Divergence Analysis")
                
                # Vertical Stacked View
                st.subheader("🟢 Bullish Signals")
                bull_df = consolidated[(consolidated['Type']=='Bullish') & (consolidated['Timeframe']==tf)]
                if not bull_df.empty:
                    st.table(bull_df.drop(columns=['Type','Timeframe']))
                else:
                    st.write("No Bullish signals found.")

                st.subheader("🔴 Bearish Signals")
                bear_df = consolidated[(consolidated['Type']=='Bearish') & (consolidated['Timeframe']==tf)]
                if not bear_df.empty:
                    st.table(bear_df.drop(columns=['Type','Timeframe']))
                else:
                    st.write("No Bearish signals found.")
        else:
            st.warning("No signals detected.")
    except Exception as e:
        st.error(f"Processing Error: {e}")
else:
    st.error("Google Drive is still blocking the 100MB file. Please reduce the S&P 500 dataset to 120 weeks (approx 30-40MB) to fix this.")
