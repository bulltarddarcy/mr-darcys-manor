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

def find_rsi_percentile_signals(df, ticker, periods_to_scan=10):
    signals = []
    if len(df) < 200: return signals
    
    # Restrict to last 10 years for percentile calculation
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    
    if hist_df.empty: return signals
    
    p10 = hist_df['RSI'].quantile(0.10)
    p90 = hist_df['RSI'].quantile(0.90)
    
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
                'Percentile': '10th %ile'
            })
            
        # Bearish Exit: Previously > p90, Now <= p90 (Leaving top)
        if prev['RSI'] > p90 and curr['RSI'] <= p90:
            signals.append({
                'Ticker': ticker,
                'Date': curr.name.strftime('%Y-%m-%d'),
                'Type': 'Bearish (Leaving Overbought)',
                'RSI': curr['RSI'],
                'Threshold': p90,
                'Percentile': '90th %ile'
            })
            
    return signals

# --- 2. APP MODULES ---

def run_database_app(df):
    st.title("📂 Database")
    max_data_date = get_max_trade_date(df)
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        default_ticker = st.session_state.get("db_ticker", "")
        db_ticker = st.text_input("Ticker", value=default_ticker.upper(), key="db_ticker_input").strip().upper()
        st.session_state["db_ticker"] = db_ticker
    with c2: start_date = st.date_input("Trade Start Date", value=max_data_date, key="db_start")
    with c3: end_date = st.date_input("Trade End Date", value=max_data_date, key="db_end")
    with c4:
        exp_range_default = (date.today() + timedelta(days=365))
        db_exp_end = st.date_input("Expiration Range (end)", value=exp_range_default, key="db_exp")
    
    ot1, ot2, ot3, ot_pad = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: inc_cb = st.checkbox("Calls Bought", value=True, key="db_inc_cb")
    with ot2: inc_ps = st.checkbox("Puts Sold", value=True, key="db_inc_ps")
    with ot3: inc_pb = st.checkbox("Puts Bought", value=True, key="db_inc_pb")
    
    f = df.copy()
    if db_ticker: f = f[f["Symbol"].astype(str).str.upper().eq(db_ticker)]
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    if db_exp_end: f = f[f["Expiry_DT"].dt.date <= db_exp_end]
    
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed_types = []
    if inc_cb: allowed_types.append("Calls Bought")
    if inc_pb: allowed_types.append("Puts Bought")
    if inc_ps: allowed_types.append("Puts Sold")
    f = f[f[order_type_col].isin(allowed_types)]
    
    if f.empty:
        st.warning("No data found matching these filters.")
        return
        
    f = f.sort_values(by=["Trade Date", "Symbol"], ascending=[False, True])
    display_cols = ["Trade Date", order_type_col, "Symbol", "Strike", "Expiry", "Contracts", "Dollars"]
    f_display = f[display_cols].copy()
    f_display["Trade Date"] = f_display["Trade Date"].dt.strftime("%d %b %y")
    f_display["Expiry"] = pd.to_datetime(f_display["Expiry"]).dt.strftime("%d %b %y")
    
    def highlight_db_order_type(val):
        if val in ["Calls Bought", "Puts Sold"]: return 'background-color: rgba(113, 210, 138, 0.15); color: #71d28a; font-weight: 600;'
        elif val == "Puts Bought": return 'background-color: rgba(242, 156, 160, 0.15); color: #f29ca0; font-weight: 600;'
        return ''
        
    st.subheader("Non-Expired Trades")
    st.caption("⚠️ User should check OI to confirm trades are still open")
    st.dataframe(f_display.style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}).applymap(highlight_db_order_type, subset=[order_type_col]), use_container_width=True, hide_index=True, height=get_table_height(f_display, max_rows=30))

def run_rankings_app(df):
    st.title("🏆 Rankings")
    max_data_date = get_max_trade_date(df)
    start_default = max_data_date - timedelta(days=14)
    
    c1, c2, c3, c_pad = st.columns([1.2, 1.2, 0.8, 3], gap="small")
    with c1: rank_start = st.date_input("Trade Start Date", value=start_default, key="rank_start")
    with c2: rank_end = st.date_input("Trade End Date", value=max_data_date, key="rank_end")
    with c3: limit = st.number_input("Limit", value=15, min_value=1, max_value=200, key="rank_limit")
    
    f = df.copy()
    if rank_start: f = f[f["Trade Date"].dt.date >= rank_start]
    if rank_end: f = f[f["Trade Date"].dt.date <= rank_end]
    
    if f.empty:
        st.warning("No data found matching these dates.")
        return

    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f_filtered = f[f[order_type_col].isin(target_types)].copy()
    
    if f_filtered.empty:
        st.warning("No trades of the specified sentiment types found in this range.")
        return

    st.markdown("---")
    st.subheader("🧠 Smart Money (Multivariate Score)")
    
    with st.expander("ℹ️ About Smart Money Methodology"):
        st.markdown("""
        * **The Algorithm:** Generates a **Smart Score (0-100)** by normalizing and weighing three key variables.
        * **Net Flow (40% Weight):** The raw dollar conviction (Calls + Puts Sold - Puts Bought).
        * **Impact (30% Weight):** The Net Flow calculated as a percentage of Market Cap. This boosts smaller tickers receiving outsized flow.
        * **Momentum (30% Weight):** The flow intensity over just the **last 3 trading days**, rewarding urgent action.
        """)
    
    with st.spinner("Processing Smart Money calculations... Please be patient, it is totally worth it!"):
        f_filtered["Signed_Dollars"] = np.where(
            f_filtered[order_type_col].isin(["Calls Bought", "Puts Sold"]), 
            f_filtered["Dollars"], 
            -f_filtered["Dollars"]
        )
        
        smart_stats = f_filtered.groupby("Symbol").agg(
            Signed_Dollars=("Signed_Dollars", "sum"),
            Trade_Count=("Symbol", "count"),
            Last_Trade=("Trade Date", "max")
        ).reset_index()
        
        smart_stats.rename(columns={"Signed_Dollars": "Net Sentiment ($)"}, inplace=True)
        smart_stats["Market Cap"] = smart_stats["Symbol"].apply(lambda x: get_market_cap(x))
        
        unique_dates = sorted(f_filtered["Trade Date"].unique())
        recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
        f_momentum = f_filtered[f_filtered["Trade Date"].isin(recent_dates)]
        mom_stats = f_momentum.groupby("Symbol")["Signed_Dollars"].sum().reset_index()
        mom_stats.rename(columns={"Signed_Dollars": "Momentum ($)"}, inplace=True)
        
        smart_stats = smart_stats.merge(mom_stats, on="Symbol", how="left").fillna(0)
        
        valid_data = smart_stats[smart_stats["Market Cap"] > 0].copy()
        
        if not valid_data.empty:
            valid_data["Impact"] = valid_data["Net Sentiment ($)"] / valid_data["Market Cap"]
            
            def normalize(series):
                mn, mx = series.min(), series.max()
                return (series - mn) / (mx - mn) if (mx != mn) else 0

            bull_flow = valid_data["Net Sentiment ($)"].clip(lower=0)
            bull_imp = valid_data["Impact"].clip(lower=0)
            bull_mom = valid_data["Momentum ($)"].clip(lower=0)
            
            valid_data["Score_Bull"] = (
                (0.40 * normalize(bull_flow)) + 
                (0.30 * normalize(bull_imp)) + 
                (0.30 * normalize(bull_mom))
            ) * 100

            bear_flow = -valid_data["Net Sentiment ($)"].clip(upper=0)
            bear_imp = -valid_data["Impact"].clip(upper=0)
            bear_mom = -valid_data["Momentum ($)"].clip(upper=0)
            
            valid_data["Score_Bear"] = (
                (0.40 * normalize(bear_flow)) + 
                (0.30 * normalize(bear_imp)) + 
                (0.30 * normalize(bear_mom))
            ) * 100
            
            valid_data["Last Trade"] = valid_data["Last_Trade"].dt.strftime("%d %b")

            top_bulls = valid_data.sort_values(by=["Score_Bull", "Net Sentiment ($)"], ascending=[False, False]).head(limit)
            top_bears = valid_data.sort_values(by=["Score_Bear", "Net Sentiment ($)"], ascending=[False, True]).head(limit)
            
            fmt_curr = lambda x: f"${x:,.0f}" if x >= 0 else f"(${abs(x):,.0f})"
            
            sm_config = {
                "Symbol": st.column_config.TextColumn("Ticker", width=50),
                "Net Sentiment ($)": st.column_config.TextColumn("Net Flow", width=85),
                "Trade_Count": st.column_config.NumberColumn("Qty", width=40, format="%d"),
                "Last Trade": st.column_config.TextColumn("Last", width=50),
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    format="%d",
                    min_value=0,
                    max_value=100,
                    width=None 
                ),
            }

            sm1, sm2 = st.columns(2, gap="large")
            
            with sm1:
                st.markdown("<div style='text-align:left; color: #71d28a; font-weight:bold; margin-bottom:5px;'>Top Bullish Scores</div>", unsafe_allow_html=True)
                disp_bull = top_bulls[["Symbol", "Score_Bull", "Net Sentiment ($)", "Trade_Count", "Last Trade"]].copy()
                disp_bull.rename(columns={"Score_Bull": "Score"}, inplace=True)
                
                st.dataframe(
                    disp_bull.style.format({"Net Sentiment ($)": fmt_curr}),
                    use_container_width=True, 
                    hide_index=True, 
                    height=get_table_height(disp_bull), 
                    column_config=sm_config
                )
            
            with sm2:
                st.markdown("<div style='text-align:left; color: #f29ca0; font-weight:bold; margin-bottom:5px;'>Top Bearish Scores</div>", unsafe_allow_html=True)
                disp_bear = top_bears[["Symbol", "Score_Bear", "Net Sentiment ($)", "Trade_Count", "Last Trade"]].copy()
                disp_bear.rename(columns={"Score_Bear": "Score"}, inplace=True)
                
                st.dataframe(
                    disp_bear.style.format({"Net Sentiment ($)": fmt_curr}),
                    use_container_width=True, 
                    hide_index=True, 
                    height=get_table_height(disp_bear), 
                    column_config=sm_config
                )

        else:
            st.warning("Not enough data with valid Market Caps to generate scores.")

    st.markdown("---")
    st.subheader("🤡 Bulltard Rankings (Volume Based)")
    st.caption("ℹ️ **Legacy Methodology:** Score = (Calls Bought + Puts Sold) - (Puts Bought). Ranked by Score first, then Dollars. Ranking tables vary from Bulltard's as he includes expired trades and these do not.")
    
    counts = f_filtered.groupby(["Symbol", order_type_col]).size().unstack(fill_value=0)
    dollars = f_filtered.groupby(["Symbol", order_type_col])["Dollars"].sum().unstack(fill_value=0)
    last_trades = f_filtered.groupby("Symbol")["Trade Date"].max().dt.strftime("%d %b %y")
    
    for col in target_types:
        if col not in counts.columns: counts[col] = 0
        if col not in dollars.columns: dollars[col] = 0
        
    scores_df = pd.DataFrame(index=counts.index)
    scores_df["Score"] = counts["Calls Bought"] + counts["Puts Sold"] - counts["Puts Bought"]
    scores_df["Trade Count"] = counts["Calls Bought"] + counts["Puts Sold"] + counts["Puts Bought"]
    scores_df["Dollars"] = dollars["Calls Bought"] + dollars["Puts Sold"] - dollars["Puts Bought"]
    
    res = scores_df.reset_index().merge(last_trades, on="Symbol")
    res = res.rename(columns={"Trade Date": "Last Trade"})
    display_cols = ["Symbol", "Trade Count", "Last Trade", "Dollars", "Score"]
    
    rank_col_config = {
        "Symbol": st.column_config.TextColumn("Sym", width=40),
        "Trade Count": st.column_config.NumberColumn("Trade Count", width=40),
        "Last Trade": st.column_config.TextColumn("Last Trade", width=70),
        "Dollars": st.column_config.TextColumn("Dollars", width=90),
        "Score": st.column_config.NumberColumn("Score", width=40),
    }
    fmt_currency_legacy = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"
    fmt_score_legacy = lambda x: f"({abs(int(x))})" if x < 0 else f"{int(x)}"
    
    bull_df = res[display_cols].sort_values(by=["Score", "Dollars"], ascending=[False, False]).head(limit)
    bear_df = res[display_cols].sort_values(by=["Score", "Dollars"], ascending=[True, True]).head(limit)
    
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown("<h4 style='color: #71d28a; margin:0;'>Bullish Volume</h4>", unsafe_allow_html=True)
        b_disp = bull_df.copy()
        b_disp["Dollars"] = b_disp["Dollars"].apply(fmt_currency_legacy)
        st.dataframe(b_disp.style.format({"Trade Count": "{:,.0f}", "Score": fmt_score_legacy}), use_container_width=True, hide_index=True, height=get_table_height(bull_df), column_config=rank_col_config)
    with col_right:
        st.markdown("<h4 style='color: #f29ca0; margin:0;'>Bearish Volume</h4>", unsafe_allow_html=True)
        br_disp = bear_df.copy()
        br_disp["Dollars"] = br_disp["Dollars"].apply(fmt_currency_legacy)
        st.dataframe(br_disp.style.format({"Trade Count": "{:,.0f}", "Score": fmt_score_legacy}), use_container_width=True, hide_index=True, height=get_table_height(bear_df), column_config=rank_col_config)

def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    exp_range_default = (date.today() + timedelta(days=365))
    
    col_settings, col_visuals = st.columns([1, 2.5], gap="large")
    
    with col_settings:
        ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
        td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
        td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
        exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
        
        st.markdown("---")
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown("**View Mode**")
            view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], label_visibility="collapsed")
            
            st.markdown("**Zone Width**")
            width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], label_visibility="collapsed")
            if width_mode == "Fixed": 
                fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=10)
            else: fixed_size_choice = 10
        
        with c_sub2:
            st.markdown("**Include**")
            inc_cb = st.checkbox("Calls Bought", value=True)
            inc_ps = st.checkbox("Puts Sold", value=True)
            inc_pb = st.checkbox("Puts Bought", value=True)
            
        hide_empty = True
        show_table = True
    
    with col_visuals:
        chart_container = st.container()

    f_base = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start: f_base = f_base[f_base["Trade Date"].dt.date >= td_start]
    if td_end: f_base = f_base[f_base["Trade Date"].dt.date <= td_end]
    today_val = date.today()
    f_base = f_base[(f_base["Expiry_DT"].dt.date >= today_val) & (f_base["Expiry_DT"].dt.date <= exp_end)]
    order_type_col = "Order Type" if "Order Type" in f_base.columns else "Order type"
    
    allowed_sz_types = []
    if inc_cb: allowed_sz_types.append("Calls Bought")
    if inc_ps: allowed_sz_types.append("Puts Sold")
    if inc_pb: allowed_sz_types.append("Puts Bought")
    
    edit_pool_raw = f_base[f_base[order_type_col].isin(allowed_sz_types)].copy()
    
    if edit_pool_raw.empty:
        with col_visuals:
            st.warning("No trades match current filters.")
        return

    if "Include" not in edit_pool_raw.columns:
        edit_pool_raw.insert(0, "Include", True)
    
    edit_pool_raw["Trade Date Str"] = edit_pool_raw["Trade Date"].dt.strftime("%d %b %y")
    edit_pool_raw["Expiry Str"] = edit_pool_raw["Expiry_DT"].dt.strftime("%d %b %y")

    if show_table:
        editor_input = edit_pool_raw[["Include", "Trade Date Str", order_type_col, "Symbol", "Strike", "Expiry Str", "Contracts", "Dollars"]].copy()
        
        editor_input["Dollars"] = editor_input["Dollars"].apply(lambda x: f"${x:,.0f}")
        editor_input["Contracts"] = editor_input["Contracts"].apply(lambda x: f"{x:,.0f}")

        st.markdown("---")
        st.subheader("Data Table & Selection")
        
        edited_df = st.data_editor(
            editor_input,
            column_config={
                "Include": st.column_config.CheckboxColumn("Include", default=True),
                "Dollars": st.column_config.TextColumn("Dollars"),
                "Contracts": st.column_config.TextColumn("Qty"),
                "Trade Date Str": "Trade Date",
                "Expiry Str": "Expiry"
            },
            disabled=["Trade Date Str", order_type_col, "Symbol", "Strike", "Expiry Str", "Contracts", "Dollars"],
            hide_index=True,
            use_container_width=True,
            key="sz_editor"
        )
        f = edit_pool_raw[edited_df["Include"]].copy()
    else:
        f = edit_pool_raw.copy()

    with chart_container:
        if f.empty:
            st.info("No rows selected. Check the 'Include' boxes below.")
        else:
            spot, ema8, ema21, sma200, history = get_stock_indicators(ticker)
            if spot is None: spot = 100.0

            def pct_from_spot(x):
                if x is None or np.isnan(x): return "—"
                return f"{(x/spot-1)*100:+.1f}%"
            
            badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
            if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct_from_spot(ema8)})</span>')
            if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct_from_spot(ema21)})</span>')
            if sma200: badges.append(f'<span class="badge">SMA(200): ${sma200:,.2f} ({pct_from_spot(sma200)})</span>')
            st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

            f["Signed Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0.0)
            
            fmt_neg = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

            if view_mode == "Price Zones":
                strike_vals = f["Strike (Actual)"].values
                strike_min, strike_max = float(np.nanmin(strike_vals)), float(np.nanmax(strike_vals))
                if width_mode == "Auto": 
                    denom = 12.0
                    zone_w = float(next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, strike_max - strike_min) / denom)), 100))
                else: zone_w = float(fixed_size_choice)
                
                n_dn = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w))
                n_up = int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
                
                lower_edge = spot - n_dn * zone_w
                total = max(1, n_dn + n_up)
                
                f["ZoneIdx"] = np.clip(
                    np.floor((f["Strike (Actual)"] - lower_edge) / zone_w).astype(int), 
                    0, 
                    total - 1
                )

                agg = f.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                
                zone_df = pd.DataFrame([(z, lower_edge + z*zone_w, lower_edge + (z+1)*zone_w) for z in range(total)], columns=["ZoneIdx","Zone_Low","Zone_High"])
                zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
                
                if hide_empty: zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
                
                html_out = ['<div class="zones-panel">']
                
                max_val = max(1.0, zs["Net_Dollars"].abs().max())
                sorted_zs = zs.sort_values("ZoneIdx", ascending=False)
                
                upper_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) > spot]
                lower_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) <= spot]
                
                for _, r in upper_zones.iterrows():
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    pct = (abs(r['Net_Dollars']) / max_val) * 100
                    val_str = fmt_neg(r["Net_Dollars"])
                    # Use single line string to avoid markdown indentation issues
                    html_out.append(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
                
                html_out.append(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>')
                
                for _, r in lower_zones.iterrows():
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    pct = (abs(r['Net_Dollars']) / max_val) * 100
                    val_str = fmt_neg(r["Net_Dollars"])
                    html_out.append(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
                
                html_out.append('</div>')
                st.markdown("".join(html_out), unsafe_allow_html=True)
                
            else:
                e = f.copy()
                days_diff = (pd.to_datetime(e["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days)
                e["Bucket"] = pd.cut(days_diff, bins=[0, 7, 30, 90, 180, 10000], labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"], include_lowest=True)
                
                agg = e.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                
                max_val = max(1.0, agg["Net_Dollars"].abs().max())
                html_out = []
                for _, r in agg.iterrows():
                    color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
                    pct = (abs(r['Net_Dollars']) / max_val) * 100
                    val_str = fmt_neg(r["Net_Dollars"])
                    html_out.append(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
                
                st.markdown("".join(html_out), unsafe_allow_html=True)
            
            st.caption("ℹ️ You can exclude individual trades from the graphic by unchecking them in the Data Tables box below.")

def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = get_max_trade_date(df)
            
    col_filters, col_calculator = st.columns([1, 1], gap="medium")
    
    st.markdown("""
        <style>
            .st-key-calc_out_ann input, .st-key-calc_out_coc input, .st-key-calc_out_dte input {
                background-color: rgba(113, 210, 138, 0.1) !important;
                color: #71d28a !important;
                border: 1px solid #71d28a !important;
                font-weight: 700 !important;
                pointer-events: none !important;
                cursor: default !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with col_filters:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>🔍 Filters</h4>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1: td_start = st.date_input("Trade Start Date", value=max_data_date, key="pv_start")
        with fc2: td_end = st.date_input("Trade End Date", value=max_data_date, key="pv_end")
        with fc3: ticker_filter = st.text_input("Ticker (blank=all)", value="", key="pv_ticker").strip().upper()
        
        fc4, fc5, fc6 = st.columns(3)
        with fc4: min_notional = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}[st.selectbox("Min Dollars", options=["0M", "5M", "10M", "50M", "100M"], index=0, key="pv_notional")]
        with fc5: min_mkt_cap = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}[st.selectbox("Mkt Cap Min", options=["0B", "10B", "50B", "100B", "200B", "500B", "1T"], index=0, key="pv_mkt_cap")]
        with fc6: ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")

    with col_calculator:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
        
        cc1, cc2, cc3 = st.columns(3)
        with cc1: c_strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0, format="%.2f", key="calc_strike")
        with cc2: c_premium = st.number_input("Premium", min_value=0.00, value=2.50, step=0.05, format="%.2f", key="calc_premium")
        with cc3: c_expiry = st.date_input("Expiration", value=date.today() + timedelta(days=30), key="calc_expiry")
        
        dte = (c_expiry - date.today()).days
        coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
        annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0

        st.session_state["calc_out_ann"] = f"{annual_ret:.1f}%"
        st.session_state["calc_out_coc"] = f"{coc_ret:.1f}%"
        st.session_state["calc_out_dte"] = str(max(0, dte))

        cc4, cc5, cc6 = st.columns(3)
        with cc4: st.text_input("Annualised Return", key="calc_out_ann")
        with cc5: st.text_input("Cash on Cash Return", key="calc_out_coc")
        with cc6: st.text_input("Days to Expiration", key="calc_out_dte")

    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 10px; margin-bottom: 20px; align-items: center;">
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#b7e1cd"></div> This Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#fce8b2"></div> Next Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#f4c7c3"></div> Two Fridays</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Market Cap filtering can be buggy. If empty, reset \'Mkt Cap Min\' to 0B.</div>', unsafe_allow_html=True)
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Scroll down to see the Risk Reversals table.</div>', unsafe_allow_html=True)

    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    if d_range.empty: return

    order_type_col = "Order Type" if "Order Type" in d_range.columns else "Order type"
    
    cb_pool = d_range[d_range[order_type_col] == "Calls Bought"].copy()
    ps_pool = d_range[d_range[order_type_col] == "Puts Sold"].copy()
    pb_pool = d_range[d_range[order_type_col] == "Puts Bought"].copy()
    
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    cb_pool['occ'], ps_pool['occ'] = cb_pool.groupby(keys).cumcount(), ps_pool.groupby(keys).cumcount()
    rr_matches = pd.merge(cb_pool, ps_pool, on=keys + ['occ'], suffixes=('_c', '_p'))
    
    if not rr_matches.empty:
        rr_c = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_c', 'Strike_c']].copy()
        rr_c.rename(columns={'Dollars_c': 'Dollars', 'Strike_c': 'Strike'}, inplace=True)
        rr_c['Pair_ID'] = rr_matches.index
        rr_c['Pair_Side'] = 0
        
        rr_p = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_p', 'Strike_p']].copy()
        rr_p.rename(columns={'Dollars_p': 'Dollars', 'Strike_p': 'Strike'}, inplace=True)
        rr_p['Pair_ID'] = rr_matches.index
        rr_p['Pair_Side'] = 1
        
        df_rr = pd.concat([rr_c, rr_p])
        df_rr['Strike'] = df_rr['Strike'].apply(clean_strike_fmt)
        
        match_keys = keys + ['occ']
        def filter_out_matches(pool, matches):
            temp_matches = matches[match_keys].copy()
            temp_matches['_remove'] = True
            merged = pool.merge(temp_matches, on=match_keys, how='left')
            return merged[merged['_remove'].isna()].drop(columns=['_remove'])
        cb_pool = filter_out_matches(cb_pool, rr_matches)
        ps_pool = filter_out_matches(ps_pool, rr_matches)
    else:
        df_rr = pd.DataFrame(columns=['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars', 'Strike', 'Pair_ID', 'Pair_Side'])

    def apply_f(data):
        if data.empty: return data
        f = data.copy()
        if ticker_filter: f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
        f = f[f["Dollars"] >= min_notional]
        
        if not f.empty:
            unique_symbols = f["Symbol"].unique()
            valid_symbols = set(unique_symbols)
            
            if min_mkt_cap > 0:
                valid_symbols = {s for s in valid_symbols if get_market_cap(s) >= float(min_mkt_cap)}
            
            if ema_filter == "Yes":
                valid_symbols = {s for s in valid_symbols if is_above_ema21(s)}
            
            f = f[f["Symbol"].isin(valid_symbols)]
            
        return f

    df_cb_f, df_ps_f, df_pb_f, df_rr_f = apply_f(cb_pool), apply_f(ps_pool), apply_f(pb_pool), apply_f(df_rr)

    def get_p(data, is_rr=False):
        if data.empty: return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
        sr = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        if is_rr: piv = data.merge(sr, on="Symbol").sort_values(by=["Total_Sym_Dollars", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
        else:
            piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().merge(sr, on="Symbol")
            piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
        
        piv["Symbol_Display"] = np.where(piv["Symbol"] == piv["Symbol"].shift(1), "", piv["Symbol"])
        
        return piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    row1_c1, row1_c2, row1_c3 = st.columns(3); fmt = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}
    with row1_c1:
        st.subheader("Calls Bought"); tbl = get_p(df_cb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
    with row1_c2:
        st.subheader("Puts Sold"); tbl = get_p(df_ps_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
    with row1_c3:
        st.subheader("Puts Bought"); tbl = get_p(df_pb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
    
    st.markdown("---")
    st.subheader("Risk Reversals")
    tbl_rr = get_p(df_rr_f, is_rr=True)
    if not tbl_rr.empty: 
        st.dataframe(tbl_rr.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl_rr), column_config=COLUMN_CONFIG_PIVOT)
    else: st.caption("No matched RR pairs found.")

def run_rsi_divergences_app():
    st.title("📈 RSI Divergences")
    st.caption("ℹ️ On mobile, set your browser to View Desktop Site")

    st.markdown("""
        <style>
        .top-note { color: #888888; font-size: 14px; margin-bottom: 2px; font-family: inherit; }
        .rsi-table { width: 100%; border-collapse: collapse; table-layout: fixed; margin-bottom: 2rem; }
        .rsi-table thead tr th { background-color: #f0f2f6 !important; color: #31333f !important; padding: 12px !important; border-bottom: 2px solid #dee2e6; }
        .rsi-table tbody tr td { padding: 10px !important; border-bottom: 1px solid #eee; word-wrap: break-word; font-size: 14px; }
        .ev-positive { background-color: #e6f4ea !important; color: #1e7e34; font-weight: 500; }
        .ev-negative { background-color: #fce8e6 !important; color: #c5221f; font-weight: 500; }
        .ev-neutral { color: #5f6368; }
        .tag-bubble { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin: 2px 4px 2px 0; color: white; white-space: nowrap; }
        .footer-header { color: #31333f; margin-top: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
        
    dataset_map = load_dataset_config()
    
    data_option = st.pills("Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0], label_visibility="collapsed")
    
    with st.expander("ℹ️ Strategy Logic & Tag Explanations"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown('<div class="footer-header">📉 SIGNAL LOGIC</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **Signal Identification**: Scans for price extremes (New Low for Bullish, New High for Bearish) within a **{SIGNAL_LOOKBACK_PERIOD}-period window**.
            * **Divergence Mechanism**: Compares the RSI at a new price extreme to a previous RSI extreme found within the **{DIVERGENCE_LOOKBACK}-period lookback**.
            * **True Pivot Logic**: The algorithm ensures the price point is a true local extreme (True Low/High). If the RSI crosses the 50 line between the two comparison points, the divergence is invalidated (reset).
            * **Bullish Standards**: Price hits a new low while RSI is at least **{RSI_DIFF_THRESHOLD} points higher** than at the previous low.
            * **Bearish Standards**: Price hits a new high while RSI is at least **{RSI_DIFF_THRESHOLD} points lower** than at the previous high.
            """)
        with f_col2:
            st.markdown('<div class="footer-header">🔮 EXPECTED VALUE (EV) ANALYSIS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **Data Pool**: Analyzes the **entire 3-year historical dataset** to find matching RSI environments.
            * **RSI Matching**: Identifies historical instances where the RSI was within **±2 points** of the current RSI level.
            * **Forward Projection**: Calculates the **Average (Mean)** percentage return for those matching instances exactly 30 and 90 periods into the future.
            * **Statistical Filter**: EV is only displayed if at least **{MIN_N_THRESHOLD} historical matches (N)** are found to ensure reliability.
            * **Color Coding**: 🟢 Green supports the trade direction (Bullish positive / Bearish negative). 🔴 Red indicates a contrary historical outcome.
            """)
        with f_col3:
            st.markdown('<div class="footer-header">🏷️ TECHNICAL TAGS</div>', unsafe_allow_html=True)
            st.markdown(f"""
            * **EMA{EMA8_PERIOD} / EMA{EMA21_PERIOD}**: Added if the current price is trading **above** (Bullish) or **below** (Bearish) these exponential moving averages.
            * **VOL_HIGH**: Triggered if the signal candle volume is > 150% of the **{VOL_SMA_PERIOD}-period average**.
            * **VOL_GROW**: Triggered if volume at the current signal point (P2) is higher than the volume at the previous extreme (P1).
            """)

    if data_option:
        try:
            target_url = st.secrets[dataset_map[data_option]]
            csv_buffer = get_confirmed_gdrive_data(target_url)
            if csv_buffer and csv_buffer != "HTML_ERROR":
                master = pd.read_csv(csv_buffer)
                
                t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                all_tickers = sorted(master[t_col].unique())
                with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                    sq = st.text_input("Filter...").upper()
                    ft = [t for t in all_tickers if sq in t]
                    cols = st.columns(6)
                    for i, ticker in enumerate(ft): cols[i % 6].write(ticker)
                
                raw_results = []
                progress_bar = st.progress(0, text="Scanning...")
                grouped = master.groupby(t_col)
                grouped_list = list(grouped)
                total_groups = len(grouped_list)
                
                for i, (ticker, group) in enumerate(grouped_list):
                    d_d, d_w = prepare_data(group.copy())
                    if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
                    if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
                    
                    if i % 5 == 0 or i == total_groups - 1:
                        progress_bar.progress((i + 1) / total_groups)
                
                if raw_results:
                    res_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
                    consolidated = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                    
                    for tf in ['Daily', 'Weekly']:
                        st.divider()
                        for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                            st.subheader(f"{emoji} {tf} {s_type} Signals")
                            tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                            
                            if not tbl_df.empty:
                                html_rows = ['<table class="rsi-table"><thead><tr><th style="width:7%">Ticker</th><th style="width:25%">Tags</th><th style="width:8%">P1 Date</th><th style="width:8%">Signal Date</th><th style="width:8%">RSI</th><th style="width:8%">P1 Price</th><th style="width:8%">P2 Price</th><th style="width:8%">Last Close</th><th style="width:10%">EV 30p</th><th style="width:10%">EV 90p</th></tr></thead><tbody>']
                                
                                for row in tbl_df.itertuples():
                                    row_html = [
                                        '<tr>',
                                        f'<td style="text-align:left"><b>{row.Ticker}</b></td>',
                                        f'<td style="text-align:left">{style_tags(row.Tags)}</td>',
                                        f'<td style="text-align:center">{row._5}</td>', 
                                        f'<td style="text-align:center">{row._6}</td>', 
                                        f'<td style="text-align:center">{row.RSI}</td>',
                                        f'<td style="text-align:left">{row._8}</td>', 
                                        f'<td style="text-align:left">{row._9}</td>', 
                                        f'<td style="text-align:left">{row._10}</td>' 
                                    ]
                                    
                                    for data in [row.ev30_raw, row.ev90_raw]:
                                        if data:
                                            is_pos = data['return'] > 0
                                            cls = ("ev-positive" if is_pos else "ev-negative") if s_type == 'Bullish' else ("ev-positive" if not is_pos else "ev-negative")
                                            row_html.append(f'<td class="{cls}">{data["return"]*100:+.1f}% <br><small>(${data["price"]:,.2f}, N={data["n"]})</small></td>')
                                        else: row_html.append('<td class="ev-neutral">N/A</td>')
                                    
                                    row_html.append('</tr>')
                                    html_rows.append("".join(row_html))
                                
                                html_rows.append('</tbody></table>')
                                st.markdown("".join(html_rows), unsafe_allow_html=True)
                            else: st.write("No signals.")
                else: st.warning("No signals.")
                
        except Exception as e: st.error(f"Error: {e}")

def run_rsi_percentiles_app():
    st.title("🔢 RSI Percentiles")
    st.caption("Identify stocks leaving their historical 10% (Oversold) or 90% (Overbought) RSI levels.")
    
    st.markdown("""
        <style>
        .signal-badge { padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 13px; }
        .signal-bull { background-color: #e6f4ea; color: #1e7e34; border: 1px solid #1e7e34; }
        .signal-bear { background-color: #fce8e6; color: #c5221f; border: 1px solid #c5221f; }
        </style>
    """, unsafe_allow_html=True)
    
    # Load dataset map but add the "Live Drive Map" option
    dataset_map = load_dataset_config()
    options = list(dataset_map.keys()) + ["Live Drive Map (Individual Files)"]
    
    data_option = st.selectbox("Select Dataset", options=options, index=0)
    
    scan_limit = 50
    if data_option == "Live Drive Map (Individual Files)":
        st.info("⚠️ This mode fetches individual files from Google Drive. It is slower than compiled datasets.")
        scan_limit = st.slider("Max Tickers to Scan (Prevent Timeout)", 10, 200, 50)
    
    if st.button("Run Analysis"):
        results = []
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            if data_option == "Live Drive Map (Individual Files)":
                # Use the provided URL_TICKER_MAP
                map_url = st.secrets.get("URL_TICKER_MAP", URL_TICKER_MAP_DEFAULT)
                map_buffer = get_confirmed_gdrive_data(map_url)
                
                if map_buffer and map_buffer != "HTML_ERROR":
                    map_df = pd.read_csv(map_buffer, header=None)
                    # Assume format: Ticker, URL/ID
                    # Rename columns just in case
                    if len(map_df.columns) >= 2:
                        map_df = map_df.iloc[:, :2]
                        map_df.columns = ['Ticker', 'Link']
                    
                    tickers_to_process = map_df.head(scan_limit)
                    total = len(tickers_to_process)
                    
                    for i, row in enumerate(tickers_to_process.itertuples()):
                        ticker = str(row.Ticker).strip().upper()
                        link = str(row.Link).strip()
                        
                        status_text.text(f"Scanning {ticker}...")
                        
                        csv_buffer = get_confirmed_gdrive_data(link)
                        if csv_buffer and csv_buffer != "HTML_ERROR":
                            df_raw = pd.read_csv(csv_buffer)
                            d_d, _ = prepare_data(df_raw)
                            
                            if d_d is not None:
                                sigs = find_rsi_percentile_signals(d_d, ticker)
                                results.extend(sigs)
                        
                        progress_bar.progress((i + 1) / total)
                else:
                    st.error("Could not load Ticker Map file.")
                    
            else:
                # Use standard compiled dataset logic
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
                            sigs = find_rsi_percentile_signals(d_d, ticker)
                            results.extend(sigs)
                        
                        if i % 10 == 0: progress_bar.progress((i + 1) / total)
                    progress_bar.progress(100)

            status_text.empty()
            
            if results:
                res_df = pd.DataFrame(results)
                res_df = res_df.sort_values(by='Date', ascending=False)
                
                st.subheader(f"Found {len(res_df)} Events (Last 10 Periods)")
                
                # Custom HTML table for badges
                html_rows = ['<table style="width:100%; border-collapse: collapse;"><thead><tr style="border-bottom: 2px solid #eee; text-align: left;"><th>Date</th><th>Ticker</th><th>Signal Type</th><th>RSI</th><th>Threshold</th><th>%ile</th></tr></thead><tbody>']
                
                for r in res_df.itertuples():
                    cls = "signal-bull" if "Bullish" in r.Type else "signal-bear"
                    row_html = f"""
                    <tr style="border-bottom: 1px solid #f0f0f0;">
                        <td style="padding: 10px;">{r.Date}</td>
                        <td style="padding: 10px; font-weight:bold;">{r.Ticker}</td>
                        <td style="padding: 10px;"><span class="signal-badge {cls}">{r.Type}</span></td>
                        <td style="padding: 10px;">{r.RSI:.1f}</td>
                        <td style="padding: 10px;">{r.Threshold:.1f}</td>
                        <td style="padding: 10px; color: #666;">{r.Percentile}</td>
                    </tr>
                    """
                    html_rows.append(row_html)
                
                html_rows.append("</tbody></table>")
                st.markdown("".join(html_rows), unsafe_allow_html=True)
                
            else:
                st.info("No tickers found crossing their 10th or 90th percentile thresholds in the last 10 periods.")
                
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
