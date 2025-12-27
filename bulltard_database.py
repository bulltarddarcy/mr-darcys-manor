import warnings
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import math
import requests
import re
from io import StringIO

# --- 1. GLOBAL DATA LOADING & UTILITIES ---

COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=None),
    "Strike": st.column_config.TextColumn("Strike", width=None),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=None),
    "Contracts": st.column_config.NumberColumn("Qty", width=None),
    "Dollars": st.column_config.NumberColumn("Dollars", width=None),
}

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21
EV_LOOKBACK_YEARS = 3
MIN_N_THRESHOLD = 5
URL_TICKER_MAP_DEFAULT = "https://drive.google.com/file/d/1MlVp6yF7FZjTdRFMpYCxgF-ezyKvO4gG/view?usp=sharing"

@st.cache_data(ttl=600, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    want = ["Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"]
    
    existing_cols = df.columns
    keep = [c for c in want if c in existing_cols]
    df = df[keep].copy()
    
    # Batch strip string columns
    str_cols = [c for c in ["Order Type", "Symbol", "Strike", "Expiry"] if c in df.columns]
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
    
    # Optimized regex replacement
    if "Dollars" in df.columns:
        df["Dollars"] = (df["Dollars"].astype(str)
                         .str.replace(r'[$,]', '', regex=True))
        df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)

    if "Contracts" in df.columns:
        df["Contracts"] = (df["Contracts"].astype(str)
                           .str.replace(',', '', regex=False))
        df["Contracts"] = pd.to_numeric(df["Contracts"], errors="coerce").fillna(0)
    
    if "Trade Date" in df.columns:
        df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    
    if "Expiry" in df.columns:
        df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
        
    if "Strike (Actual)" in df.columns:
        df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
        
    if "Error" in df.columns:
        # Vectorized filtering
        mask = df["Error"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
        df = df[~mask]
        
    return df

@st.cache_data(ttl=3600)
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        mc = t.fast_info.get('marketCap')
        if mc: return float(mc)
        return float(t.info.get('marketCap', 0))
    except:
        return 0.0

@st.cache_data(ttl=300)
def is_above_ema21(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol)
        h = ticker.history(period="60d")
        if len(h) < 21:
            return True 
        ema21_last = h["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
        latest_price = h["Close"].iloc[-1]
        return latest_price > ema21_last
    except:
        return True

@st.cache_data(ttl=300)
def get_stock_indicators(sym: str):
    try:
        ticker_obj = yf.Ticker(sym)
        h_full = ticker_obj.history(period="2y", interval="1d")
        
        if len(h_full) == 0: return None, None, None, None, None
        
        sma200 = float(h_full["Close"].rolling(window=200).mean().iloc[-1]) if len(h_full) >= 200 else None
        
        h_recent = h_full.iloc[-60:].copy() if len(h_full) > 60 else h_full.copy()
        
        if len(h_recent) == 0: return None, None, None, None, None
        
        close = h_recent["Close"]
        spot_val = float(close.iloc[-1])
        ema8  = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        
        return spot_val, ema8, ema21, sma200, h_recent
    except: 
        return None, None, None, None, None

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0:
        return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

def highlight_expiry(val):
    try:
        if not isinstance(val, str): return ""
        # Optimization: parsing format specific to the pivot output
        expiry_date = datetime.strptime(val, "%d %b %y").date()
        today = date.today()
        # Pre-calculation of friday delta
        days_ahead = (4 - today.weekday()) % 7
        this_fri = today + timedelta(days=days_ahead)
        
        if expiry_date < today: return "" 
        if expiry_date == this_fri: return "background-color: #b7e1cd; color: black;" 
        if expiry_date == this_fri + timedelta(days=7): return "background-color: #fce8b2; color: black;" 
        if expiry_date == this_fri + timedelta(days=14): return "background-color: #f4c7c3; color: black;" 
        return ""
    except: return ""

def clean_strike_fmt(val):
    try:
        f = float(val)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except: return str(val)

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid_dates = df["Trade Date"].dropna()
        if not valid_dates.empty:
            return valid_dates.max().date()
    return date.today() - timedelta(days=1)

def get_confirmed_gdrive_data(url):
    try:
        file_id = ""
        if 'id=' in url:
            file_id = url.split('id=')[1].split('&')[0]
        elif '/d/' in url:
            file_id = url.split('/d/')[1].split('/')[0]
        
        if not file_id: return None
            
        download_url = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(download_url, params={'id': file_id}, stream=True)
        
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_token = value
                break
        
        if not confirm_token:
            match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
            if match: confirm_token = match.group(1)

        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
        
        if response.text.strip().startswith("<!DOCTYPE html>"): return "HTML_ERROR"
            
        return StringIO(response.text)
    except Exception as e:
        # st.error(f"Fetch Error: {e}") 
        return None

def load_dataset_config():
    try:
        if "URL_CONFIG" not in st.secrets:
            return {"Darcy Data": "URL_DARCY", "S&P 100 Data": "URL_SP100"}
        config_url = st.secrets["URL_CONFIG"]
        buffer = get_confirmed_gdrive_data(config_url)
        if buffer and buffer != "HTML_ERROR":
            lines = buffer.getvalue().splitlines()
            config_dict = {}
            for line in lines:
                if ',' in line:
                    name, key = line.split(',')
                    config_dict[name.strip()] = key.strip()
            return config_dict
    except Exception as e:
        st.error(f"Error loading config file: {e}")
    return {"Darcy Data": "URL_DARCY"}

def style_tags(tag_str):
    if not tag_str: return ''
    tags = tag_str.split(", ")
    colors = {f"EMA{EMA8_PERIOD}": "#4a90e2", f"EMA{EMA21_PERIOD}": "#9b59b6", "VOL_HIGH": "#e67e22", "VOL_GROW": "#27ae60"}
    html_parts = []
    for t in tags:
        color = colors.get(t, "#7f8c8d")
        html_parts.append(f'<span class="tag-bubble" style="background-color: {color};">{t}</span>')
    return "".join(html_parts)

def calculate_ev_data_numpy(rsi_array, price_array, target_rsi, periods, current_price):
    mask = (rsi_array >= target_rsi - 2) & (rsi_array <= target_rsi + 2)
    indices = np.where(mask)[0]
    
    if len(indices) == 0: return None

    exit_indices = indices + periods
    valid_mask = exit_indices < len(price_array)
    
    if not np.any(valid_mask): return None
    
    valid_starts = indices[valid_mask]
    valid_exits = exit_indices[valid_mask]
    
    entry_prices = price_array[valid_starts]
    exit_prices = price_array[valid_exits]
    
    valid_entries_mask = entry_prices > 0
    if not np.any(valid_entries_mask): return None
    
    returns = (exit_prices[valid_entries_mask] - entry_prices[valid_entries_mask]) / entry_prices[valid_entries_mask]
    
    if len(returns) < MIN_N_THRESHOLD: return None
    
    avg_ret = np.mean(returns)
    ev_price = current_price * (1 + avg_ret)
    return {"price": ev_price, "n": len(returns), "return": avg_ret}

def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    
    # RSI fallback check
    rsi_col = next((c for c in cols if 'RSI' in c and 'W_' not in c), None)
    
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Daily Data
    # Use existing RSI column if found, else default name
    d_rsi = rsi_col if rsi_col else 'RSI_14'
    d_ema8, d_ema21 = 'EMA_8', 'EMA_21'
    
    needed_cols = [close_col, vol_col, high_col, low_col]
    if d_rsi in df.columns: needed_cols.append(d_rsi)
    if d_ema8 in df.columns: needed_cols.append(d_ema8)
    if d_ema21 in df.columns: needed_cols.append(d_ema21)
    
    df_d = df[needed_cols].copy()
    
    rename_dict = {close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low'}
    if d_rsi in df_d.columns: rename_dict[d_rsi] = 'RSI'
    if d_ema8 in df_d.columns: rename_dict[d_ema8] = 'EMA8'
    if d_ema21 in df_d.columns: rename_dict[d_ema21] = 'EMA21'
    
    df_d.rename(columns=rename_dict, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    if 'RSI' not in df_d.columns:
        # Fallback if RSI not in file (shouldn't happen per prompt)
        delta = df_d['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_d['RSI'] = 100 - (100 / (1 + rs))
        
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    
    # Weekly Data (Only needed for divergences, but good to have logic consistent)
    w_close, w_vol, w_rsi = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_high, w_low, w_ema8, w_ema21 = 'W_HIGH', 'W_LOW', 'W_EMA_8', 'W_EMA_21'
    
    if all(c in df.columns for c in [w_close, w_vol, w_high, w_low, w_rsi]):
        df_w = df[[w_close, w_vol, w_high, w_low, w_rsi, w_ema8, w_ema21]].copy()
        df_w.rename(columns={w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low', w_rsi: 'RSI', w_ema8: 'EMA8', w_ema21: 'EMA21'}, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: 
        df_w = None
        
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    divergences = []
    n_rows = len(df_tf)
    if n_rows < DIVERGENCE_LOOKBACK + 1: return divergences
    
    cutoff_date = df_tf.index.max() - timedelta(days=365 * EV_LOOKBACK_YEARS)
    hist_mask = df_tf.index >= cutoff_date
    rsi_hist = df_tf.loc[hist_mask, 'RSI'].values
    price_hist = df_tf.loc[hist_mask, 'Price'].values
    
    latest_p = df_tf.iloc[-1]
    
    ev30 = calculate_ev_data_numpy(rsi_hist, price_hist, latest_p['RSI'], 30, latest_p['Price'])
    ev90 = calculate_ev_data_numpy(rsi_hist, price_hist, latest_p['RSI'], 90, latest_p['Price'])
    
    def get_date_str(p): return df_tf.loc[p.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else p.name.strftime('%Y-%m-%d')
    
    start_idx = max(DIVERGENCE_LOOKBACK, n_rows - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, n_rows):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        for s_type in ['Bullish', 'Bearish']:
            trigger = False
            p1 = None
            
            if s_type == 'Bullish':
                if p2['Low'] < lookback['Low'].min():
                    p1_idx = lookback['RSI'].idxmin()
                    p1 = lookback.loc[p1_idx]
                    
                    if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                        subset = df_tf.loc[p1.name : p2.name, 'RSI']
                        if not (subset > 50).any(): trigger = True
            else: 
                if p2['High'] > lookback['High'].max():
                    p1_idx = lookback['RSI'].idxmax()
                    p1 = lookback.loc[p1_idx]
                    
                    if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                        subset = df_tf.loc[p1.name : p2.name, 'RSI']
                        if not (subset < 50).any(): trigger = True
            
            if trigger and p1 is not None:
                post_df = df_tf.iloc[i + 1 :]
                valid = True
                if not post_df.empty:
                    if s_type == 'Bullish' and (post_df['RSI'] <= p1['RSI']).any(): valid = False
                    if s_type == 'Bearish' and (post_df['RSI'] >= p1['RSI']).any(): valid = False
                
                if valid:
                    tags = []
                    if s_type == 'Bullish':
                        if latest_p['Price'] >= latest_p.get('EMA8', 0): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] >= latest_p.get('EMA21', 0): tags.append(f"EMA{EMA21_PERIOD}")
                    else:
                        if latest_p['Price'] <= latest_p.get('EMA8', 999999): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] <= latest_p.get('EMA21', 999999): tags.append(f"EMA{EMA21_PERIOD}")
                    
                    if is_vol_high: tags.append("VOL_HIGH")
                    if p2['Volume'] > p1['Volume']: tags.append("VOL_GROW")
                    
                    divergences.append({
                        'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                        'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                        'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                        'P1 Price': f"${p1['Low' if s_type=='Bullish' else 'High']:,.2f}", 
                        'P2 Price': f"${p2['Low' if s_type=='Bullish' else 'High']:,.2f}", 
                        'Last Close': f"${latest_p['Price']:,.2f}", 
                        'ev30_raw': ev30, 'ev90_raw': ev90
                    })
    return divergences

def calculate_crossover_ev(rsi_arr, price_arr, threshold, mode, periods):
    # rsi_arr and price_arr are aligned
    if len(rsi_arr) < periods + 2: return None
    
    prev_rsi = rsi_arr[:-1]
    curr_rsi = rsi_arr[1:]
    
    if mode == 'bull': # Leaving Oversold ( < p10 to >= p10)
        mask = (prev_rsi < threshold) & (curr_rsi >= threshold)
    else: # Leaving Overbought ( > p90 to <= p90)
        mask = (prev_rsi > threshold) & (curr_rsi <= threshold)
        
    # indices in curr_rsi correspond to indices+1 in original arrays
    match_indices = np.where(mask)[0] + 1
    
    if len(match_indices) == 0: return None
    
    # Filter for valid exits
    exit_indices = match_indices + periods
    valid_mask = exit_indices < len(price_arr)
    
    if not np.any(valid_mask): return None
    
    valid_entries = match_indices[valid_mask]
    valid_exits = exit_indices[valid_mask]
    
    entry_prices = price_arr[valid_entries]
    exit_prices = price_arr[valid_exits]
    
    # Avoid div by zero (unlikely but safe)
    valid_prices = entry_prices > 0
    if not np.any(valid_prices): return None
    
    entry_prices = entry_prices[valid_prices]
    exit_prices = exit_prices[valid_prices]
    
    returns = (exit_prices - entry_prices) / entry_prices
    
    if len(returns) < MIN_N_THRESHOLD: return None 
    
    return {'return': np.mean(returns), 'n': len(returns)}

def find_rsi_percentile_signals(df, ticker, periods_to_scan=10, oversold_pct=0.10, overbought_pct=0.90):
    signals = []
    if len(df) < 200: return signals
    
    # Restrict to last 10 years for percentile calculation
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    
    if hist_df.empty: return signals
    
    p10 = hist_df['RSI'].quantile(oversold_pct)
    p90 = hist_df['RSI'].quantile(overbought_pct)
    
    # Calculate historical EV stats for both sides
    rsi_vals = hist_df['RSI'].values
    price_vals = hist_df['Price'].values
    
    ev_stats = {
        'bull': {
            '30': calculate_crossover_ev(rsi_vals, price_vals, p10, 'bull', 30),
            '90': calculate_crossover_ev(rsi_vals, price_vals, p10, 'bull', 90)
        },
        'bear': {
            '30': calculate_crossover_ev(rsi_vals, price_vals, p90, 'bear', 30),
            '90': calculate_crossover_ev(rsi_vals, price_vals, p90, 'bear', 90)
        }
    }
    
    # Check last N periods
    # Need at least periods_to_scan + 1 rows to check crossing
    if len(hist_df) < periods_to_scan + 2: return signals
    
    scan_window = hist_df.iloc[-(periods_to_scan+1):]
    
    for i in range(1, len(scan_window)):
        prev = scan_window.iloc[i-1]
        curr = scan_window.iloc[i]
        
        # Bullish Exit: Previously < p10, Now >= p10 (Leaving bottom)
        if prev['RSI'] < p10 and curr['RSI'] >= p10:
            signals.append({
                'Ticker': ticker,
                'Date': curr.name.strftime('%Y-%m-%d'),
                'Type': 'Bullish (Leaving Oversold)',
                'RSI': curr['RSI'],
                'Threshold': p10,
                'Percentile': f"{int(oversold_pct*100)}th %ile",
                'ev30': ev_stats['bull']['30'],
                'ev90': ev_stats['bull']['90']
            })
            
        # Bearish Exit: Previously > p90, Now <= p90 (Leaving top)
        if prev['RSI'] > p90 and curr['RSI'] <= p90:
            signals.append({
                'Ticker': ticker,
                'Date': curr.name.strftime('%Y-%m-%d'),
                'Type': 'Bearish (Leaving Overbought)',
                'RSI': curr['RSI'],
                'Threshold': p90,
                'Percentile': f"{int(overbought_pct*100)}th %ile",
                'ev30': ev_stats['bear']['30'],
                'ev90': ev_stats['bear']['90']
            })
            
    return signals

def run_rsi_percentiles_app():
    st.title("🔢 RSI Percentiles")
    
    st.markdown("""
        <style>
        /* CSS for row hovering effect if needed, though background colors will dominate */
        tr:hover { filter: brightness(95%); }
        </style>
    """, unsafe_allow_html=True)
    
    # Load dataset map
    dataset_map = load_dataset_config()
    options = list(dataset_map.keys())
    
    c1, c2 = st.columns(2)
    with c1:
        oversold_pct_in = st.number_input("Oversold Percentile", min_value=1, max_value=49, value=10, step=1)
    with c2:
        overbought_pct_in = st.number_input("Overbought Percentile", min_value=51, max_value=99, value=90, step=1)
    
    # Use pills for selection (mimicking Divergences tab)
    default_opt = options[0] if options else None
    data_option = st.pills("Dataset", options=options, selection_mode="single", default=default_opt, label_visibility="collapsed")
    
    with st.expander("ℹ️ Strategy Logic & Explanations"):
        st.markdown('<div class="footer-header">📊 PERCENTILE LOGIC</div>', unsafe_allow_html=True)
        st.markdown(f"""
        * **Historical Context**: The algorithm analyzes up to **10 years** of daily price history for each ticker to establish its unique RSI behavior.
        * **Percentile Thresholds**: It calculates the **User-Selected Percentiles** (Default: 10th for Oversold, 90th for Overbought) specific to that stock.
        * **Signal Trigger**: 
            * **Bullish (Leaving Oversold)**: Triggered when the RSI crosses **ABOVE** the low percentile threshold (recovering from extreme lows).
            * **Bearish (Leaving Overbought)**: Triggered when the RSI crosses **BELOW** the high percentile threshold (cooling off from extreme highs).
        * **Scan Window**: The system checks the last **10 trading periods** to catch recent signals.
        * **Expected Value (EV)**: The average historical return (30d and 90d) specifically following these crossover events for this ticker. 
        * **Reliability Filter**: EV stats are only displayed if at least **{MIN_N_THRESHOLD} historical events (N)** were found.
        """)
    
    # Auto-run analysis when a dataset is selected
    if data_option:
        results = []
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            target_url = st.secrets[dataset_map[data_option]]
            csv_buffer = get_confirmed_gdrive_data(target_url)
            
            if csv_buffer and csv_buffer != "HTML_ERROR":
                master = pd.read_csv(csv_buffer)
                t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                grouped = master.groupby(t_col)
                grouped_list = list(grouped)
                total = len(grouped_list)
                
                for i, (ticker, group) in enumerate(grouped_list):
                    status_text.text(f"Scanning {ticker}...")
                    d_d, _ = prepare_data(group.copy())
                    if d_d is not None:
                        sigs = find_rsi_percentile_signals(
                            d_d, ticker, 
                            oversold_pct=oversold_pct_in/100.0, 
                            overbought_pct=overbought_pct_in/100.0
                        )
                        results.extend(sigs)
                    
                    if i % 10 == 0: progress_bar.progress((i + 1) / total)
                progress_bar.progress(100)
            else:
                st.error("Could not load dataset.")

            status_text.empty()
            
            if results:
                res_df = pd.DataFrame(results)
                res_df = res_df.sort_values(by='Date', ascending=False)
                
                st.subheader(f"Found {len(res_df)} Events (Last 10 Periods)")
                
                # Custom HTML table with row shading
                html_rows = ['<table style="width:100%; border-collapse: collapse;"><thead><tr style="border-bottom: 2px solid #eee; text-align: left;"><th>Date</th><th>Ticker</th><th>RSI</th><th>Threshold</th><th>EV 30d</th><th>EV 90d</th></tr></thead><tbody>']
                
                def fmt_ev(ev_dict):
                    if not ev_dict: return "N/A"
                    val = ev_dict['return'] * 100
                    n = ev_dict['n']
                    color = "green" if val > 0 else "red"
                    # Removed font-weight:bold per user request
                    return f"<span style='color:{color};'>{val:+.1f}%</span> <span style='color:#666; font-size:0.85em'>(N={n})</span>"

                for r in res_df.itertuples():
                    # Apply row background color based on signal type
                    if "Bullish" in r.Type:
                        row_style = "background-color: #e6f4ea; border-bottom: 1px solid #fff;" # Greenish
                    else:
                        row_style = "background-color: #fce8e6; border-bottom: 1px solid #fff;" # Reddish
                    
                    ev30_str = fmt_ev(r.ev30)
                    ev90_str = fmt_ev(r.ev90)
                    
                    row_html = f'<tr style="{row_style}"><td style="padding: 10px;">{r.Date}</td><td style="padding: 10px; font-weight:bold;">{r.Ticker}</td><td style="padding: 10px;">{r.RSI:.1f}</td><td style="padding: 10px;">{r.Threshold:.1f}</td><td style="padding: 10px;">{ev30_str}</td><td style="padding: 10px;">{ev90_str}</td></tr>'
                    html_rows.append(row_html)
                
                html_rows.append("</tbody></table>")
                st.markdown("".join(html_rows), unsafe_allow_html=True)
                
            else:
                st.info(f"No tickers found crossing their {oversold_pct_in}th or {overbought_pct_in}th percentile thresholds in the last 10 periods.")
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# --- 3. MAIN EXECUTION ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

st.markdown("""<style>
.block-container{padding-top:3.5rem;padding-bottom:1rem;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}
.zone-wrapper{
    flex-grow: 1; 
    position: relative; 
    height: 24px; 
    background-color: rgba(0,0,0,0.03);
    border-radius: 4px;
    overflow: hidden;
}
.zone-bar{
    position: absolute;
    left: 0; 
    top: 0; 
    bottom: 0; 
    z-index: 1;
    border-radius: 3px;
    opacity: 0.65;
}
.zone-bull{background-color: #71d28a;}
.zone-bear{background-color: #f29ca0;}
.zone-value{
    position: absolute;
    right: 8px;
    top: 0;
    bottom: 0;
    display: flex;
    align-items: center;
    z-index: 2;
    font-size: 12px; 
    font-weight: 700;
    color: #1f1f1f;
    white-space: nowrap;
    text-shadow: 0 0 4px rgba(255,255,255,0.8);
}
.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: #66b7ff; opacity: 0.4; }
.price-badge { background: rgba(102, 183, 255, 0.1); color: #66b7ff; border: 1px solid rgba(102, 183, 255, 0.5); border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; white-space: nowrap; margin: 0 12px; z-index: 1; }
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
.badge{background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius:18px; padding:6px 10px; font-weight:700}
.price-badge-header{background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 10px; font-weight:800}
.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; }
</style>""", unsafe_allow_html=True)

try:
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)
    last_updated_date = df_global["Trade Date"].max().strftime("%d %b %y")

    pg = st.navigation([
        st.Page(lambda: run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(run_rsi_divergences_app, title="RSI Divergences", icon="📈", url_path="rsi_divergences"),
        st.Page(run_rsi_percentiles_app, title="RSI Percentiles", icon="🔢", url_path="rsi_percentiles"),
    ])

    st.sidebar.caption("🖥️ Everything is best viewed with a wide desktop monitor in light mode.")
    st.sidebar.caption(f"📅 **Last Updated:** {last_updated_date}")
    
    pg.run()
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")
