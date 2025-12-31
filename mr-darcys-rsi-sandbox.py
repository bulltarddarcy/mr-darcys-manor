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
import time
from io import StringIO
import altair as alt
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 0. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

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
# MIN_N_THRESHOLD is now dynamic based on user input
URL_TICKER_MAP_DEFAULT = "https://drive.google.com/file/d/1MlVp6yF7FZjTdRFMpYCxgF-ezyKvO4gG/view?usp=sharing"

@st.cache_data(ttl=600)
def load_and_clean_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        want = ["Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"]
        existing_cols = set(df.columns)
        keep = [c for c in want if c in existing_cols]
        df = df[keep].copy()
        
        for c in ["Order Type", "Symbol", "Strike", "Expiry"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        
        if "Dollars" in df.columns:
            if df["Dollars"].dtype == 'object':
                df["Dollars"] = df["Dollars"].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)
            
        if "Trade Date" in df.columns:
            df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
            
        if "Expiry" in df.columns:
            df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
            
        if "Strike (Actual)" in df.columns:
            df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
            
        if "Error" in df.columns:
            mask = df["Error"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
            df = df[~mask]
            
        return df
    except Exception as e:
        st.error(f"Error loading global data: {e}")
        return pd.DataFrame()

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
                
        if confirm_token:
            response = session.get(download_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)
            
        if response.text.strip().startswith("<!DOCTYPE html>"):
            return "HTML_ERROR"
            
        return StringIO(response.text)
    except:
        return None

@st.cache_data(ttl=3600)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP", URL_TICKER_MAP_DEFAULT)
        buffer = get_confirmed_gdrive_data(url)
        if buffer and buffer != "HTML_ERROR":
            df = pd.read_csv(buffer)
            return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), df.iloc[:, 1].astype(str).str.strip()))
    except:
        pass
    return {}

@st.cache_data(ttl=300)
def get_ticker_technicals(ticker: str, mapping: dict):
    if not mapping or ticker not in mapping: return None
    file_id = mapping[ticker]
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    buffer = get_confirmed_gdrive_data(file_url)
    if buffer and buffer != "HTML_ERROR":
        try:
            df = pd.read_csv(buffer)
            df.columns = [c.strip().upper() for c in df.columns]
            return df
        except:
            return None
    return None

# --- 2. RSI CALCULATIONS & LOGIC ---

def calculate_optimal_signal_stats(history_indices, price_array, current_idx):
    """
    Scans multiple holding periods (10, 30, 60, 90, 180) for historical signal occurrences
    to find the holding duration that yields the highest Profit Factor.
    """
    valid_hist_indices = [idx for idx in history_indices if idx < current_idx]
    if not valid_hist_indices: return None
    
    periods = [10, 30, 60, 90, 180]
    best_pf = -1.0
    best_stats = None
    total_len = len(price_array)
    
    for p in periods:
        returns = []
        for idx in valid_hist_indices:
            exit_idx = idx + p
            if exit_idx < total_len:
                entry_p = price_array[idx]
                exit_p = price_array[exit_idx]
                if entry_p > 0:
                    returns.append((exit_p - entry_p) / entry_p)
        
        if not returns: continue
        
        returns_np = np.array(returns)
        wins = returns_np[returns_np > 0]
        losses = returns_np[returns_np < 0]
        
        gross_win = np.sum(wins)
        gross_loss = np.abs(np.sum(losses))
        
        if gross_loss == 0:
            pf = 999.0 if gross_win > 0 else 0.0
        else:
            pf = gross_win / gross_loss
            
        wr = (len(wins) / len(returns_np)) * 100
        ev = np.mean(returns_np) * 100
        
        if pf > best_pf:
            best_pf = pf
            best_stats = {
                "Best Period": f"{p}d",
                "Profit Factor": pf,
                "Win Rate": wr,
                "EV": ev,
                "N": len(returns_np)
            }
            
    return best_stats

def prepare_data(df):
    """
    Standardizes column names and separates data into Daily and Weekly formats.
    """
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    
    # Identify pre-populated technicals
    d_rsi = next((c for c in cols if 'RSI' in c and 'W_' not in c), 'RSI_14')
    d_ema8 = 'EMA_8' if 'EMA_8' in cols else None
    d_ema21 = 'EMA_21' if 'EMA_21' in cols else None
    
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    # Build Daily DF
    needed_cols = [close_col, vol_col, high_col, low_col]
    if d_rsi in df.columns: needed_cols.append(d_rsi)
    if d_ema8: needed_cols.append(d_ema8)
    if d_ema21: needed_cols.append(d_ema21)
    
    df_d = df[needed_cols].copy()
    rename_dict = {close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low'}
    if d_rsi in df_d.columns: rename_dict[d_rsi] = 'RSI'
    if d_ema8: rename_dict[d_ema8] = 'EMA8'
    if d_ema21: rename_dict[d_ema21] = 'EMA21'
    
    df_d.rename(columns=rename_dict, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    
    # Ensure RSI exists
    if 'RSI' not in df_d.columns:
        delta = df_d['Price'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        df_d['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    
    # Build Weekly DF
    w_close, w_vol, w_rsi = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_high, w_low, w_ema8, w_ema21 = 'W_HIGH', 'W_LOW', 'W_EMA_8', 'W_EMA_21'
    
    if all(c in df.columns for c in [w_close, w_vol, w_high, w_low, w_rsi]):
        cols_w = [w_close, w_vol, w_high, w_low, w_rsi]
        if w_ema8 in df.columns: cols_w.append(w_ema8)
        if w_ema21 in df.columns: cols_w.append(w_ema21)
        
        df_w = df[cols_w].copy()
        w_rename = {w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low', w_rsi: 'RSI'}
        if w_ema8 in df.columns: w_rename[w_ema8] = 'EMA8'
        if w_ema21 in df.columns: w_rename[w_ema21] = 'EMA21'
        
        df_w.rename(columns=w_rename, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else:
        df_w = None
        
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe, min_n=0):
    """
    Identifies Bullish and Bearish RSI Divergences based on pivots.
    """
    divergences = []
    n_rows = len(df_tf)
    if n_rows < DIVERGENCE_LOOKBACK + 1: return divergences
    
    rsi_vals = df_tf['RSI'].values
    low_vals = df_tf['Low'].values
    high_vals = df_tf['High'].values
    vol_vals = df_tf['Volume'].values
    vol_sma_vals = df_tf['VolSMA'].values
    close_vals = df_tf['Price'].values 
    
    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        if timeframe.lower() == 'weekly': 
             return df_tf.iloc[idx]['ChartDate'].strftime(fmt)
        return ts.strftime(fmt)
    
    bullish_indices = []
    bearish_indices = []
    potential_signals = [] 

    start_search = DIVERGENCE_LOOKBACK
    
    for i in range(start_search, n_rows):
        p2_rsi = rsi_vals[i]
        p2_low = low_vals[i]
        p2_high = high_vals[i]
        p2_vol = vol_vals[i]
        p2_volsma = vol_sma_vals[i]
        
        lb_start = i - DIVERGENCE_LOOKBACK
        lb_rsi = rsi_vals[lb_start:i]
        lb_low = low_vals[lb_start:i]
        lb_high = high_vals[lb_start:i]
        
        is_vol_high = int(p2_vol > (p2_volsma * 1.5)) if not np.isnan(p2_volsma) else 0
        
        # Bullish
        if p2_low < np.min(lb_low):
            p1_idx_rel = np.argmin(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            if p2_rsi > (p1_rsi + RSI_DIFF_THRESHOLD):
                idx_p1_abs = lb_start + p1_idx_rel
                if not np.any(rsi_vals[idx_p1_abs : i + 1] > 50): 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi <= p1_rsi): valid = False
                    if valid:
                        bullish_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bullish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})
        
        # Bearish
        elif p2_high > np.max(lb_high):
            p1_idx_rel = np.argmax(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            if p2_rsi < (p1_rsi - RSI_DIFF_THRESHOLD):
                idx_p1_abs = lb_start + p1_idx_rel
                if not np.any(rsi_vals[idx_p1_abs : i + 1] < 50): 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi >= p1_rsi): valid = False
                    if valid:
                        bearish_indices.append(i)
                        potential_signals.append({"index": i, "type": "Bearish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})

    display_threshold_idx = n_rows - SIGNAL_LOOKBACK_PERIOD
    
    for sig in potential_signals:
        i = sig["index"]
        if i < display_threshold_idx: continue

        s_type = sig["type"]
        idx_p1_abs = sig["p1_idx"]
        latest_p = df_tf.iloc[-1] 
        
        # Tags Generation
        tags = []
        row_at_sig = df_tf.iloc[i] 
        curr_price = row_at_sig['Price']
        
        # Look for the internal names mapped in prepare_data
        ema8_val = row_at_sig.get('EMA8') 
        ema21_val = row_at_sig.get('EMA21')

        if s_type == 'Bullish':
            if ema8_val is not None and curr_price >= ema8_val: tags.append(f"EMA{EMA8_PERIOD}")
            if ema21_val is not None and curr_price >= ema21_val: tags.append(f"EMA{EMA21_PERIOD}")
        else: # Bearish
            if ema8_val is not None and curr_price <= ema8_val: tags.append(f"EMA{EMA8_PERIOD}")
            if ema21_val is not None and curr_price <= ema21_val: tags.append(f"EMA{EMA21_PERIOD}")
        
        if sig["vol_high"]: tags.append("VOL_HIGH")
        if vol_vals[i] > vol_vals[idx_p1_abs]: tags.append("VOL_GROW")
        
        # Performance
        hist_list = bullish_indices if s_type == 'Bullish' else bearish_indices
        best_stats = calculate_optimal_signal_stats(hist_list, close_vals, i)
        if best_stats is None:
             best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
        
        if best_stats["N"] < min_n: continue
            
        p1_p = low_vals[idx_p1_abs] if s_type=='Bullish' else high_vals[idx_p1_abs]
        p2_p = low_vals[i] if s_type=='Bullish' else high_vals[i]
        
        divergences.append({
            'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 
            'Tags': tags, 
            'Date_Display': f"{get_date_str(idx_p1_abs, '%b %d')} → {get_date_str(i, '%b %d')}",
            'RSI_Display': f"{int(round(rsi_vals[idx_p1_abs]))} {'↗' if rsi_vals[i] > rsi_vals[idx_p1_abs] else '↘'} {int(round(rsi_vals[i]))}",
            'Price_Display': f"${p1_p:,.2f} {'↗' if p2_p > p1_p else '↘'} ${p2_p:,.2f}",
            'Last_Close': f"${latest_p['Price']:,.2f}", 
            'Best Period': best_stats['Best Period'],
            'Profit Factor': best_stats['Profit Factor'],
            'Win Rate': best_stats['Win Rate'],
            'EV': best_stats['EV'],
            'N': best_stats['N']
        })
            
    return divergences

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, min_n=1, filter_date=None):
    """
    Identifies signals when RSI leaves a historical percentile threshold.
    """
    if len(df) < 200: return []
    
    p10 = df['RSI'].quantile(pct_low)
    p90 = df['RSI'].quantile(pct_high)
    rsi_vals = df['RSI'].values
    price_vals = df['Price'].values
    
    bull_idx, bear_idx = [], []
    for i in range(1, len(df)):
        # Leaving Low (Bullish)
        if rsi_vals[i-1] < p10 and rsi_vals[i] >= (p10 + 1.0):
            bull_idx.append(i)
        # Leaving High (Bearish)
        elif rsi_vals[i-1] > p90 and rsi_vals[i] <= (p90 - 1.0):
            bear_idx.append(i)
            
    signals = []
    all_indices = sorted(bull_idx + bear_idx)
    
    for i in all_indices:
        dt = df.index[i].date()
        if filter_date and dt < filter_date: continue
        
        is_bull = i in bull_idx
        hist = bull_idx if is_bull else bear_idx
        
        stats = calculate_optimal_signal_stats(hist, price_vals, i)
        if stats is None:
            stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
            
        if stats["N"] < min_n: continue
        
        signals.append({
            'Ticker': ticker,
            'Date': df.index[i].strftime('%b %d'),
            'Date_Obj': dt,
            'Action': "Leaving Low" if is_bull else "Leaving High",
            'RSI_Display': f"{p10 if is_bull else p90:.0f} {'↗' if is_bull else '↘'} {rsi_vals[i]:.0f}",
            'Signal_Price': f"${df.iloc[i]['Price']:,.2f}",
            'Last_Close': f"${df.iloc[-1]['Price']:,.2f}",
            'Signal_Type': 'Bullish' if is_bull else 'Bearish',
            'Best Period': stats['Best Period'],
            'Profit Factor': stats['Profit Factor'],
            'Win Rate': stats['Win Rate'],
            'EV': stats['EV'],
            'N': stats['N']
        })
        
    return signals

@st.cache_data(ttl=3600)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        df = df.reset_index().rename(columns={
            df.columns[0]: "DATE", "Close": "CLOSE", "Volume": "VOLUME", 
            "High": "HIGH", "Low": "LOW", "Open": "OPEN"
        })
        df.columns = [c.upper() for c in df.columns]
        
        delta = df["CLOSE"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df
    except:
        return None

# --- 3. THE APP MODULES ---

def run_rsi_scanner_app(df_global):
    st.title("📈 RSI Scanner")
    
    dataset_map = {
        "Darcy Data": "URL_DARCY",
        "S&P 100 Data": "URL_SP100"
    }
    options = list(dataset_map.keys())
    
    tab_div, tab_pct, tab_bot = st.tabs(["📉 Divergences", "🔢 Percentiles", "🤖 Backtester"])

    with tab_bot:
        st.markdown('<div class="light-note">ℹ️ Use this tab to check any ticker against its full 10-year RSI history.</div>', unsafe_allow_html=True)
        with st.expander("ℹ️ Page Notes: Backtester Logic"):
            st.markdown("""
            * **Data Source**: Pulls complete 10-year history via Yahoo Finance.
            * **Methodology**: Identifies every historical date where the RSI was in the same range as today (+/- Tolerance).
            * **Statistics**: Calculates the forward returns for multiple holding periods (1d, 30d, 90d, etc.) for every one of those historical matches.
            """)
        
        c_l, c_r = st.columns([1, 6])
        with c_l:
            bt_ticker = st.text_input("Ticker", value="NFLX", key="rsi_bt_ticker_input").strip().upper()
            lookback_y = st.number_input("Lookback Years", 1, 10, 10)
            bt_tol = st.number_input("RSI Tolerance", 0.5, 5.0, 2.0, 0.5)
            
        if bt_ticker:
            with st.spinner(f"Backtesting {bt_ticker}..."):
                df_bt = fetch_yahoo_data(bt_ticker)
                if df_bt is not None:
                    # Filter history
                    cutoff = df_bt["DATE"].max() - timedelta(days=365*lookback_y)
                    df_bt = df_bt[df_bt["DATE"] >= cutoff].copy()
                    
                    curr_rsi = df_bt.iloc[-1]["RSI"]
                    st.sidebar.metric("Current RSI", f"{curr_rsi:.2f}")
                    
                    # Find historical matches
                    matches = df_bt.iloc[:-1][(df_bt["RSI"] >= curr_rsi - bt_tol) & (df_bt["RSI"] <= curr_rsi + bt_tol)].index.values
                    
                    results = []
                    periods = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
                    for p in periods:
                        valid_matches = matches[matches + p < len(df_bt)]
                        if len(valid_matches) == 0:
                            results.append({"Days": p, "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "Count": 0})
                            continue
                            
                        rets = (df_bt["CLOSE"].values[valid_matches + p] - df_bt["CLOSE"].values[valid_matches]) / df_bt["CLOSE"].values[valid_matches]
                        wins = rets[rets > 0]
                        losses = rets[rets < 0]
                        
                        gross_win = np.sum(wins)
                        gross_loss = np.abs(np.sum(losses))
                        
                        if gross_loss == 0:
                            pf = 999.0 if gross_win > 0 else 0.0
                        else:
                            pf = gross_win / gross_loss
                            
                        results.append({
                            "Days": p, 
                            "Profit Factor": pf, # MOVED: Profit Factor now between Days and Win Rate
                            "Win Rate": np.mean(rets > 0)*100, 
                            "EV": np.mean(rets)*100, 
                            "Count": len(valid_matches)
                        })
                    
                    st.dataframe(pd.DataFrame(results), hide_index=True, column_config={
                        "Days": st.column_config.NumberColumn(width=60),
                        "Profit Factor": st.column_config.NumberColumn(format="%.2f", width=100),
                        "Win Rate": st.column_config.NumberColumn(format="%.1f%%", width=80),
                        "EV": st.column_config.NumberColumn(format="%.2f%%", width=80),
                        "Count": st.column_config.NumberColumn(width=60)
                    })

    with tab_div:
        dataset_div = st.pills("Dataset", options=options, default=options[0], key="rsi_div_pills")
        with st.expander("ℹ️ Page Notes: Divergence Strategy Logic"):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('**📉 SIGNAL LOGIC**\n* **Bullish**: Price makes a Lower Low, but RSI makes a Higher Low.\n* **Bearish**: Price makes a Higher High, but RSI makes a Lower High.')
            with c2:
                # NEW COLUMN: COLUMNS EXPLAINED
                st.markdown('**📋 COLUMNS EXPLAINED**\n* **RSI/Price Δ**: Shows the change from Pivot 1 (Old) to Pivot 2 (Current).\n* **Best Period**: The historical holding time (10-180d) that yielded the best Profit Factor.')
            with c3:
                st.markdown('**🔮 SIGNAL-BASED OPTIMIZATION**\n* Finds every historical occurrence of this specific signal for the ticker.\n* Runs backtests on 5 holding periods to find the statistical "winner".')
            with c4:
                st.markdown('**🏷️ TAGS**\n* **EMA8/21**: Crossovers relative to signal price.\n* **VOL**: High relative volume (1.5x) or growing volume vs last pivot.')

        if dataset_div:
            try:
                # Use st.secrets for URLs
                master_url = st.secrets.get(dataset_map[dataset_div])
                if not master_url:
                    st.error(f"Secrets key '{dataset_map[dataset_div]}' not found.")
                else:
                    master_df = pd.read_csv(get_confirmed_gdrive_data(master_url))
                    t_col = next(c for c in master_df.columns if c.strip().upper() in ['TICKER', 'SYMBOL'])
                    
                    min_n_div = st.number_input("Minimum N", 0, 50, 0, key="rsi_div_min_n")
                    
                    raw_results_div = []
                    prog_div = st.progress(0, "Scanning for Divergences...")
                    
                    groups = list(master_df.groupby(t_col))
                    for i, (ticker, group) in enumerate(groups):
                        d_d, d_w = prepare_data(group.copy())
                        if d_d is not None:
                            raw_results_div.extend(find_divergences(d_d, ticker, 'Daily', min_n=min_n_div))
                        if d_w is not None:
                            raw_results_div.extend(find_divergences(d_w, ticker, 'Weekly', min_n=min_n_div))
                        prog_div.progress((i+1)/len(groups))
                    
                    prog_div.empty()
                    
                    if raw_results_div:
                        df_res_div = pd.DataFrame(raw_results_div)
                        for tf in ['Daily', 'Weekly']:
                            for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                                st.subheader(f"{emoji} {tf} {s_type} Divergences")
                                subset = df_res_div[(df_res_div['Type'] == s_type) & (df_res_div['Timeframe'] == tf)]
                                if not subset.empty:
                                    st.dataframe(subset, hide_index=True, use_container_width=True, column_config={
                                        "Tags": st.column_config.ListColumn(),
                                        "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                                        "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                                        "EV": st.column_config.NumberColumn(format="%.1f%%"),
                                        "Type": None, "Timeframe": None
                                    })
                                else:
                                    st.info(f"No {tf} {s_type} signals found.")
            except Exception as e:
                st.error(f"Scanner error: {e}")

    with tab_pct:
        dataset_pct = st.pills("Dataset", options=options, default=options[0], key="rsi_pct_pills")
        with st.expander("ℹ️ Page Notes: Percentile Strategy Logic"):
            cp1, cp2 = st.columns(2)
            with cp1:
                # NEW BULLET: PERCENTILES EXPLAINED
                st.markdown("""
                **🔢 PERCENTILES EXPLAINED**
                * **Low Percentile**: The RSI level that the stock is historically below only X% of the time (e.g., 10% = Extremely Oversold).
                * **High Percentile**: The RSI level that the stock is historically above only X% of the time (e.g., 90% = Extremely Overbought).
                """)
            with cp2:
                # NEW BULLET: COLUMNS EXPLAINED
                st.markdown("""
                **📋 COLUMNS EXPLAINED**
                * **Action**: "Leaving Low" means RSI was below the percentile and just crossed back up.
                * **RSI Δ**: Shows the move from the [Percentile Threshold] → [Current RSI Value].
                * **N**: Total historical times the stock has "left" this specific percentile level.
                """)
        
        if dataset_pct:
            master_url_pct = st.secrets.get(dataset_map[dataset_pct])
            master_df_pct = pd.read_csv(get_confirmed_gdrive_data(master_url_pct))
            t_col_pct = next(c for c in master_df_pct.columns if c.strip().upper() in ['TICKER', 'SYMBOL'])
            
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            with c_p1: in_low_pct = st.number_input("Low Percentile (%)", 1, 49, 10)
            with c_p2: in_high_pct = st.number_input("High Percentile (%)", 51, 99, 90)
            with c_p3: date_cutoff = st.date_input("Filter: Signals since", date.today() - timedelta(days=14))
            with c_p4: min_n_pct = st.number_input("Minimum N", 0, 50, 1, key="rsi_pct_min_n")
            
            raw_results_pct = []
            prog_pct = st.progress(0, "Scanning Percentiles...")
            
            groups_pct = list(master_df_pct.groupby(t_col_pct))
            for i, (ticker, group) in enumerate(groups_pct):
                df_d_p, _ = prepare_data(group.copy())
                if df_d_p is not None:
                    raw_results_pct.extend(find_rsi_percentile_signals(df_d_p, ticker, in_low_pct/100, in_high_pct/100, min_n_pct, date_cutoff))
                prog_pct.progress((i+1)/len(groups_pct))
                
            prog_pct.empty()
            
            if raw_results_pct:
                df_res_pct = pd.DataFrame(raw_results_pct).sort_values(by='Date_Obj', ascending=False)
                st.dataframe(df_res_pct, hide_index=True, use_container_width=True, column_config={
                    "Profit Factor": st.column_config.NumberColumn(format="%.2f"),
                    "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
                    "EV": st.column_config.NumberColumn(format="%.1f%%"),
                    "Date_Obj": None, "Signal_Type": None
                })
            else:
                st.info("No percentile signals found for this criteria.")

# --- 4. NAVIGATION & BOILERPLATE ---

def run_database_app(df_global):
    st.title("📂 Options Database")
    st.write("Database features would go here...")

def run_rankings_app(df_global):
    st.title("🏆 Rankings")
    st.write("Ranking features would go here...")

def run_pivot_tables_app(df_global):
    st.title("🎯 Pivot Tables")
    st.write("Pivot Table features would go here...")

def run_strike_zones_app(df_global):
    st.title("📊 Strike Zones")
    st.write("Strike Zone features would go here...")

# Inject custom CSS
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #ffffff; }
.main-header { font-size: 2.2rem; font-weight: 800; margin-bottom: 1rem; color: #66b7ff; }
.ticker-badge { background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 12px; font-weight:700 }
.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; }
[data-testid="stDataFrame"] th { font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)

try:
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)

    pg = st.navigation([
        st.Page(lambda: run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(lambda: run_rsi_scanner_app(df_global), title="RSI Sandbox", icon="📈", url_path="rsi_sandbox"),
    ])
    pg.run()
except Exception as e:
    st.error(f"Navigation Error: {e}")
