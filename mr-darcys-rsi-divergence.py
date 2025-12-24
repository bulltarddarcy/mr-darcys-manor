import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from datetime import datetime

# --- Advanced Data Fetching (Bypasses Google Drive Virus Warning Handshake) ---
def get_confirmed_gdrive_data(url):
    """Automates the 'Download Anyway' click for large Google Drive files."""
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        
        # Pass 1: Attempt to get the file. If it's large, this returns the HTML warning page.
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        # Look for the confirmation token in the HTML body or cookies
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            # Fallback: Extract token from the "Download Anyway" link in the HTML
            match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if match:
                confirm_token = match.group(1)

        # Pass 2: If we found a token, request the file again with the 'confirm' parameter
        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
        
        # Final validation: check if the content looks like HTML or CSV
        content_start = response.text[:200]
        if "<!DOCTYPE html>" in content_start or "<html>" in content_start:
            return "HTML_ERROR"
            
        return StringIO(response.text)
    except Exception as e:
        st.error(f"Fetch Error: {e}")
        return None

# --- Logic Constants (Synced with Source of Truth) ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA_PERIOD = 8
EMA21_PERIOD = 21

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Scanner", layout="wide")
st.title("📈 RSI Divergence Scanner")

data_option = st.sidebar.selectbox("Select Dataset", ("Divergences Data", "S&P 500 Data"))

try:
    target_url = st.secrets["URL_DIVERGENCES"] if data_option == "Divergences Data" else st.secrets["URL_SP500"]
except KeyError:
    st.error("Secrets missing. Please check your Streamlit Secrets.")
    st.stop()

# --- Logic Functions ---

def prepare_data(df):
    """Clean and map columns. Sync with SOT logic."""
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    # SOT Weekly Column Names
    w_close, w_vol, w_rsi = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_high, w_low = 'W_HIGH', 'W_LOW'

    if not all([date_col, close_col, vol_col, high_col, low_col]):
        return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Daily Processing
    df_d = df[[close_col, vol_col, high_col, low_col, 'RSI_14', 'EMA_8', 'EMA_21']].copy()
    df_d.rename(columns={close_col:'Price', vol_col:'Volume', high_col:'High', low_col:'Low', 'RSI_14':'RSI', 'EMA_8':'EMA8', 'EMA_21':'EMA21'}, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])

    # Weekly Processing (Matching SOT column names)
    weekly_cols = [w_close, w_vol, w_high, w_low, w_rsi]
    if all(c in df.columns for c in weekly_cols):
        df_w = df[weekly_cols].copy()
        df_w.rename(columns={w_close:'Price', w_vol:'Volume', w_high:'High', w_low:'Low', w_rsi:'RSI'}, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else:
        df_w = None
    
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    """Detection logic. Sync with SOT."""
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
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
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

# --- Execution ---

st.info(f"Connecting to {data_option} (Performing Token Handshake for large file)...")
csv_buffer = get_confirmed_gdrive_data(target_url)

if csv_buffer == "HTML_ERROR":
    st.error("Google Drive is still serving an HTML warning page. Please reduce the dataset to 120 weeks (approx 30MB) to fix this permanently.")
elif csv_buffer:
    try:
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
        
        if not t_col:
            st.error(f"Ticker column not found. Available: {list(master.columns)}")
            st.stop()

        raw_results = []
        progress_bar = st.progress(0, text="Scanning for Divergences...")
        
        grouped = master.groupby(t_col)
        total = len(grouped)
        
        for i, (ticker, group) in enumerate(grouped):
            d_d, d_w = prepare_data(group.copy())
            if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
            if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
            progress_bar.progress((i + 1) / total)

        if raw_results:
            res_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
            consolidated = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
            
            for tf in ['Daily', 'Weekly']:
                st.markdown(f"---")
                st.header(f"📅 {tf} Divergence Analysis")
                
                for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                    st.subheader(f"{emoji} {s_type} Signals")
                    tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)]
                    if not tbl_df.empty:
                        st.table(tbl_df.drop(columns=['Type', 'Timeframe']))
                    else:
                        st.write(f"No {tf} {s_type} signals found.")
        else:
            st.warning("No signals detected.")
    except Exception as e:
        st.error(f"Processing Error: {e}")
