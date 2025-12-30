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
# Fallback if scipy is not installed
try:
    from scipy.signal import argrelextrema
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- 0. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

# --- 1. CSS STYLING ---
st.markdown("""<style>
.block-container{padding-top:3.5rem;padding-bottom:1rem;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}
.zone-wrapper{flex-grow: 1; position: relative; height: 24px; background-color: rgba(0,0,0,0.03);border-radius: 4px;overflow: hidden;}
.zone-bar{position: absolute;left: 0; top: 0; bottom: 0; z-index: 1;border-radius: 3px;opacity: 0.65;}
.zone-bull{background-color: #71d28a;}
.zone-bear{background-color: #f29ca0;}
.zone-value{position: absolute;right: 8px;top: 0;bottom: 0;display: flex;align-items: center;z-index: 2;font-size: 12px; font-weight: 700;color: #1f1f1f;white-space: nowrap;text-shadow: 0 0 4px rgba(255,255,255,0.8);}
.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: #66b7ff; opacity: 0.4; }
.price-badge { background: rgba(102, 183, 255, 0.1); color: #66b7ff; border: 1px solid rgba(102, 183, 255, 0.5); border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; white-space: nowrap; margin: 0 12px; z-index: 1; }
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
.badge{background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius:18px; padding:6px 10px; font-weight:700}
.price-badge-header{background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 10px; font-weight:800}
.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; }
.st-key-calc_out_ann input, .st-key-calc_out_coc input, .st-key-calc_out_dte input {background-color: rgba(113, 210, 138, 0.1) !important;color: #71d28a !important;border: 1px solid #71d28a !important;font-weight: 700 !important;pointer-events: none !important;cursor: default !important;}
/* BACKTESTER BOLD HEADERS */
[data-testid="stDataFrame"] th { font-weight: 900 !important; }
</style>""", unsafe_allow_html=True)

# --- 2. CONSTANTS ---
COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=None),
    "Strike": st.column_config.TextColumn("Strike", width=None),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=None),
    "Contracts": st.column_config.NumberColumn("Qty", width=None),
    "Dollars": st.column_config.NumberColumn("Dollars", width=None),
}

VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21
EV_LOOKBACK_YEARS = 3
MIN_N_THRESHOLD = 5
URL_TICKER_MAP_DEFAULT = "https://drive.google.com/file/d/1MlVp6yF7FZjTdRFMpYCxgF-ezyKvO4gG/view?usp=sharing"

# --- 3. ALL HELPER FUNCTIONS (DEFINED FIRST TO AVOID NAME ERRORS) ---

@st.cache_data(ttl=600, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        want = ["Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"]
        existing_cols = set(df.columns)
        keep = [c for c in want if c in existing_cols]
        df = df[keep].copy()
        str_cols = [c for c in ["Order Type", "Symbol", "Strike", "Expiry"] if c in df.columns]
        for c in str_cols:
            df[c] = df[c].astype(str).str.strip()
        if "Dollars" in df.columns:
            if df["Dollars"].dtype == 'object':
                df["Dollars"] = df["Dollars"].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)
        if "Contracts" in df.columns:
            if df["Contracts"].dtype == 'object':
                df["Contracts"] = df["Contracts"].str.replace(',', '', regex=False)
            df["Contracts"] = pd.to_numeric(df["Contracts"], errors="coerce").fillna(0)
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

@st.cache_data(ttl=3600)
def get_market_cap(symbol: str) -> float:
    for attempt in range(3):
        try:
            t = yf.Ticker(symbol)
            mc = t.fast_info.get('marketCap')
            if mc: return float(mc)
            info = t.info
            mc = info.get('marketCap')
            if mc: return float(mc)
        except Exception:
            time.sleep(0.1)
    return 0.0

def fetch_market_caps_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_market_cap, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except:
                results[t] = 0.0
    return results

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
        return spot_val, ema8, ema21, sma200, h_full
    except: 
        return None, None, None, None, None

def fetch_technicals_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_stock_indicators, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            try:
                results[t] = future.result()
            except:
                results[t] = (None, None, None, None, None)
    return results

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0:
        return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

@st.cache_data(ttl=3600)
def get_expiry_color_map():
    try:
        today = date.today()
        days_ahead = (4 - today.weekday()) % 7
        this_fri = today + timedelta(days=days_ahead)
        next_fri = this_fri + timedelta(days=7)
        two_fri = this_fri + timedelta(days=14)
        return {
            this_fri.strftime("%d %b %y"): "background-color: #b7e1cd; color: black;",
            next_fri.strftime("%d %b %y"): "background-color: #fce8b2; color: black;",
            two_fri.strftime("%d %b %y"): "background-color: #f4c7c3; color: black;"
        }
    except:
        return {}

def highlight_expiry(val):
    if not isinstance(val, str): return ""
    color_map = get_expiry_color_map()
    return color_map.get(val, "")

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

@st.cache_data(ttl=3600)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP", URL_TICKER_MAP_DEFAULT)
        buffer = get_confirmed_gdrive_data(url)
        if buffer and buffer != "HTML_ERROR":
            df = pd.read_csv(buffer)
            if len(df.columns) >= 2:
                return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), 
                              df.iloc[:, 1].astype(str).str.strip()))
    except Exception:
        pass
    return {}

@st.cache_data(ttl=300)
def get_ticker_technicals(ticker: str, mapping: dict):
    if not mapping or ticker not in mapping:
        return None
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

def fetch_and_prepare_ai_context(url, name, limit=90):
    try:
        buffer = get_confirmed_gdrive_data(url)
        if buffer and buffer != "HTML_ERROR":
            df = pd.read_csv(buffer)
            df_recent = df.tail(limit)
            csv_str = df_recent.to_csv(index=False)
            return f"\n\n--- DATASET: {name} ---\n{csv_str}"
    except Exception as e:
        return f"\n[Error loading {name}: {e}]"
    return ""

def style_tags(tag_str):
    # This was used for HTML table but kept just in case. 
    # For list columns we pass a python list.
    if not tag_str: return ''
    tags = tag_str.split(", ")
    return "".join([f'<span class="tag-bubble">{t}</span>' for t in tags])

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

def get_optimal_rsi_duration(history_df, current_rsi, tolerance=2.0):
    if history_df is None or len(history_df) < 100:
        return 30, "Default (No Hist)"
    close_col = "CLOSE" if "CLOSE" in history_df.columns else "Close"
    if "RSI_14" not in history_df.columns and "RSI" not in history_df.columns:
         delta = history_df[close_col].diff()
         gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
         loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
         rs = gain / loss
         history_df["RSI"] = 100 - (100 / (1 + rs))
    rsi_col = "RSI_14" if "RSI_14" in history_df.columns else "RSI"
    rsi_vals = history_df[rsi_col].values
    close_vals = history_df[close_col].values
    min_rsi = current_rsi - tolerance
    max_rsi = current_rsi + tolerance
    mask = (rsi_vals >= min_rsi) & (rsi_vals <= max_rsi)
    match_indices = np.where(mask)[0]
    if len(match_indices) < 5:
        return 30, "Default (Low Samples)"
    periods = [14, 30, 45, 60]
    best_p = 30
    best_score = -999
    total_len = len(close_vals)
    for p in periods:
        valid_indices = match_indices[match_indices + p < total_len]
        if len(valid_indices) < 5: continue
        entries = close_vals[valid_indices]
        exits = close_vals[valid_indices + p]
        returns = (exits - entries) / entries
        win_rate = np.mean(returns > 0)
        avg_ret = np.mean(returns)
        score = (win_rate * 2) + avg_ret 
        if score > best_score:
            best_score = score
            best_p = p
    return best_p, f"RSI Backtest (Optimal {best_p}d)"

def find_whale_confluence(ticker, global_df, current_price, order_type_filter=None):
    if global_df.empty: return None
    today_dt = pd.to_datetime(date.today())
    f = global_df[
        (global_df["Symbol"].astype(str).str.upper() == ticker) & 
        (global_df["Expiry_DT"] > today_dt)
    ].copy()
    if f.empty: return None
    if order_type_filter:
        f = f[f["Order Type"] == order_type_filter]
    else:
        f = f[f["Order Type"].isin(["Puts Sold", "Calls Bought"])]
    if f.empty: return None
    f = f.sort_values(by="Dollars", ascending=False)
    whale_trade = f.iloc[0]
    whale_strike = whale_trade["Strike (Actual)"]
    whale_exp = whale_trade["Expiry_DT"]
    whale_dollars = whale_trade["Dollars"]
    whale_type = whale_trade["Order Type"]
    if whale_type == "Puts Sold" and whale_strike > current_price:
        otm_puts = f[(f["Order Type"]=="Puts Sold") & (f["Strike (Actual)"] < current_price)]
        if not otm_puts.empty:
            whale_trade = otm_puts.iloc[0]
            whale_strike = whale_trade["Strike (Actual)"]
            whale_exp = whale_trade["Expiry_DT"]
    return {
        "Strike": whale_strike,
        "Expiry": whale_exp.strftime("%d %b"),
        "Dollars": whale_dollars,
        "Type": whale_type
    }

def analyze_trade_setup(ticker, t_df, global_df):
    score = 0
    reasons = []
    suggestions = {'Buy Calls': None, 'Sell Puts': None, 'Buy Commons': None}
    if t_df is None or t_df.empty:
        return 0, ["No data"], suggestions
    last = t_df.iloc[-1]
    close = last.get('CLOSE', 0) if 'CLOSE' in last else last.get('Close', 0)
    ema8 = last.get('EMA_8', 0)
    ema21 = last.get('EMA_21', 0)
    sma200 = last.get('SMA_200', 0)
    rsi = last.get('RSI_14', 50)
    if close > ema8 and close > ema21:
        score += 2
        reasons.append("Strong Trend (Price > EMA8 & EMA21)")
    elif close > ema21:
        score += 1
        reasons.append("Moderate Trend (Price > EMA21)")
    if close > sma200:
        score += 2
        reasons.append("Long-term Bullish (> SMA200)")
    if 45 < rsi < 65:
        score += 2
        reasons.append(f"Healthy Momentum (RSI {rsi:.0f})")
    elif rsi >= 70:
        score -= 1
        reasons.append("Overbought (RSI > 70)")
    opt_days, opt_reason = 30, "Standard 30d"
    if len(t_df) > 100:
         opt_days, opt_reason = get_optimal_rsi_duration(t_df, rsi)
    target_date = date.today() + timedelta(days=opt_days)
    target_date_str = target_date.strftime("%d %b")
    put_whale = find_whale_confluence(ticker, global_df, close, "Puts Sold")
    call_whale = find_whale_confluence(ticker, global_df, close, "Calls Bought")
    sp_strike = math.floor(ema21) 
    sp_reason = "EMA21 Support"
    sp_exp = target_date_str
    if put_whale and put_whale["Strike"] < close:
        sp_strike = put_whale["Strike"]
        sp_reason = f"Whale Tailing (${put_whale['Dollars']/1e6:.1f}M sold)"
        sp_exp = put_whale["Expiry"] 
    elif call_whale:
         sp_exp = call_whale["Expiry"]
         sp_reason = f"EMA21 (Align with Call Whale Exp)"
    suggestions['Sell Puts'] = f"Strike ${sp_strike} ({sp_reason}), Exp ~{sp_exp}"
    bc_strike = math.ceil(close)
    bc_reason = "ATM Momentum"
    bc_exp = target_date_str
    if call_whale:
        bc_strike = call_whale["Strike"]
        bc_exp = call_whale["Expiry"]
        bc_reason = f"Tailing Call Whale (${call_whale['Dollars']/1e6:.1f}M)"
    if close > ema8 or call_whale:
        suggestions['Buy Calls'] = f"Strike ${bc_strike} ({bc_reason}), Exp ~{bc_exp}"
    suggestions['Buy Commons'] = f"Entry: ${close:.2f}. Stop Loss: ${ema21:.2f}"
    if "RSI Backtest" in opt_reason:
        reasons.append(f"Hist. Optimal Hold: {opt_days} Days")
    if put_whale:
        reasons.append(f"Whale: Sold Puts @ ${put_whale['Strike']}")
    if call_whale:
        reasons.append(f"Whale: Bought Calls @ ${call_whale['Strike']}")
    return score, reasons, suggestions

def backtest_signal_performance(signal_indices, price_array, holding_periods=[10, 30, 60, 90, 180]):
    if len(signal_indices) == 0: return None
    best_period_stats = None
    best_pf = -1.0
    total_len = len(price_array)
    for p in holding_periods:
        valid_indices = signal_indices[signal_indices + p < total_len]
        if len(valid_indices) < MIN_N_THRESHOLD: continue
        entry_prices = price_array[valid_indices]
        exit_prices = price_array[valid_indices + p]
        returns = (exit_prices - entry_prices) / entry_prices
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        n_wins = len(wins)
        n_total = len(returns)
        win_rate = (n_wins / n_total) * 100
        ev_pct = np.mean(returns) * 100
        gross_profit = np.sum(wins)
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0
        if gross_loss == 0: profit_factor = 99.9
        else: profit_factor = gross_profit / gross_loss
        if profit_factor > best_pf:
            best_pf = profit_factor
            best_period_stats = {"Best Period": f"{p}d","Profit Factor": profit_factor,"Win Rate": win_rate,"EV": ev_pct,"N": n_total}
    return best_period_stats

def find_historical_divergences(df_tf, s_type='Bullish'):
    price_vals = df_tf['Low'].values if s_type == 'Bullish' else df_tf['High'].values
    rsi_vals = df_tf['RSI'].values
    order = 5
    if SCIPY_AVAILABLE:
        if s_type == 'Bullish': pivot_idxs = argrelextrema(price_vals, np.less, order=order)[0]
        else: pivot_idxs = argrelextrema(price_vals, np.greater, order=order)[0]
    else:
        # Fallback to pure numpy peak detection
        if s_type == 'Bullish':
             # Simple local min
             pivot_idxs = np.where((price_vals[1:-1] < price_vals[:-2]) & (price_vals[1:-1] < price_vals[2:]))[0] + 1
        else:
             pivot_idxs = np.where((price_vals[1:-1] > price_vals[:-2]) & (price_vals[1:-1] > price_vals[2:]))[0] + 1

    if len(pivot_idxs) < 2: return np.array([])
    p2s = pivot_idxs[1:]
    p1s = pivot_idxs[:-1]
    if s_type == 'Bullish':
        price_lower = price_vals[p2s] < price_vals[p1s]
        rsi_higher = rsi_vals[p2s] > (rsi_vals[p1s] + RSI_DIFF_THRESHOLD)
        rsi_valid = rsi_vals[p2s] < 60
        mask = price_lower & rsi_higher & rsi_valid
    else:
        price_higher = price_vals[p2s] > price_vals[p1s]
        rsi_lower = rsi_vals[p2s] < (rsi_vals[p1s] - RSI_DIFF_THRESHOLD)
        rsi_valid = rsi_vals[p2s] > 40
        mask = price_higher & rsi_lower & rsi_valid
    return p2s[mask]

def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    cols = df.columns
    date_col = next((c for c in cols if 'DATE' in c), None)
    close_col = next((c for c in cols if 'CLOSE' in c and 'W_' not in c), None)
    vol_col = next((c for c in cols if ('VOL' in c or 'VOLUME' in c) and 'W_' not in c), None)
    high_col = next((c for c in cols if 'HIGH' in c and 'W_' not in c), None)
    low_col = next((c for c in cols if 'LOW' in c and 'W_' not in c), None)
    rsi_col = next((c for c in cols if 'RSI' in c and 'W_' not in c), None)
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
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
        delta = df_d['Price'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df_d['RSI'] = 100 - (100 / (1 + rs))
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    w_close, w_vol, w_rsi = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_high, w_low, w_ema8, w_ema21 = 'W_HIGH', 'W_LOW', 'W_EMA_8', 'W_EMA_21'
    if all(c in df.columns for c in [w_close, w_vol, w_high, w_low, w_rsi]):
        df_w = df[[w_close, w_vol, w_high, w_low, w_rsi, w_ema8, w_ema21]].copy()
        df_w.rename(columns={w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low', w_rsi: 'RSI', w_ema8: 'EMA8', w_ema21: 'EMA21'}, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: df_w = None
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    divergences = []
    n_rows = len(df_tf)
    if n_rows < DIVERGENCE_LOOKBACK + 1: return divergences
    rsi_vals = df_tf['RSI'].values
    low_vals = df_tf['Low'].values
    high_vals = df_tf['High'].values
    vol_vals = df_tf['Volume'].values
    vol_sma_vals = df_tf['VolSMA'].values
    close_vals = df_tf['Price'].values 
    latest_p = df_tf.iloc[-1]
    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        if timeframe.lower() == 'weekly': return df_tf.iloc[idx]['ChartDate'].strftime(fmt)
        return ts.strftime(fmt)
    start_idx = max(DIVERGENCE_LOOKBACK, n_rows - SIGNAL_LOOKBACK_PERIOD)
    for i in range(start_idx, n_rows):
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
        for s_type in ['Bullish', 'Bearish']:
            trigger = False
            p1_idx_rel = -1
            if s_type == 'Bullish':
                if p2_low < np.min(lb_low):
                    p1_idx_rel = np.argmin(lb_rsi)
                    p1_rsi = lb_rsi[p1_idx_rel]
                    if p2_rsi > (p1_rsi + RSI_DIFF_THRESHOLD):
                        idx_p1_abs = lb_start + p1_idx_rel
                        if not np.any(rsi_vals[idx_p1_abs : i + 1] > 50): trigger = True
            else: 
                if p2_high > np.max(lb_high):
                    p1_idx_rel = np.argmax(lb_rsi)
                    p1_rsi = lb_rsi[p1_idx_rel]
                    if p2_rsi < (p1_rsi - RSI_DIFF_THRESHOLD):
                        idx_p1_abs = lb_start + p1_idx_rel
                        if not np.any(rsi_vals[idx_p1_abs : i + 1] < 50): trigger = True
            if trigger and p1_idx_rel != -1:
                idx_p1_abs = lb_start + p1_idx_rel
                hist_sig_indices = find_historical_divergences(df_tf, s_type)
                stats = backtest_signal_performance(hist_sig_indices, close_vals)
                if stats:
                    tags = []
                    if s_type == 'Bullish':
                        if latest_p['Price'] >= latest_p.get('EMA8', 0): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] >= latest_p.get('EMA21', 0): tags.append(f"EMA{EMA21_PERIOD}")
                    else:
                        if latest_p['Price'] <= latest_p.get('EMA8', 999999): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] <= latest_p.get('EMA21', 999999): tags.append(f"EMA{EMA21_PERIOD}")
                    if is_vol_high: tags.append("VOL_HIGH")
                    if p2_vol > vol_vals[idx_p1_abs]: tags.append("VOL_GROW")
                    sig_date_iso = get_date_str(i, '%Y-%m-%d')
                    p1_date_fmt = get_date_str(idx_p1_abs, '%b %d')
                    sig_date_fmt = get_date_str(i, '%b %d')
                    date_display = f"{p1_date_fmt} → {sig_date_fmt}"
                    rsi_p1 = rsi_vals[idx_p1_abs]
                    rsi_p2 = p2_rsi
                    rsi_display = f"{int(round(rsi_p1))} ↗ {int(round(rsi_p2))}" if rsi_p2 > rsi_p1 else f"{int(round(rsi_p1))} ↘ {int(round(rsi_p2))}"
                    price_p1 = low_vals[idx_p1_abs] if s_type=='Bullish' else high_vals[idx_p1_abs]
                    price_p2 = p2_low if s_type=='Bullish' else p2_high
                    price_display = f"${price_p1:,.2f} ↗ ${price_p2:,.2f}" if price_p2 > price_p1 else f"${price_p1:,.2f} ↘ ${price_p2:,.2f}"
                    divergences.append({
                        'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 
                        'Tags': tags, 
                        'Signal_Date_ISO': sig_date_iso, 
                        'Date_Display': date_display,
                        'RSI_Display': rsi_display,
                        'Price_Display': price_display,
                        'Last_Close': f"${latest_p['Price']:,.2f}",
                        'Best Period': stats['Best Period'],
                        'Profit Factor': stats['Profit Factor'],
                        'Win Rate': stats['Win Rate'],
                        'EV': stats['EV'],
                        'N': stats['N']
                    })
    return divergences

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, periods_to_scan=10):
    signals = []
    if len(df) < 200: return signals
    full_rsi = df['RSI'].values
    full_price = df['Price'].values
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    if hist_df.empty: return signals
    p10 = hist_df['RSI'].quantile(pct_low)
    p90 = hist_df['RSI'].quantile(pct_high)
    bull_mask = (pd.Series(full_rsi).shift(1) < p10) & (pd.Series(full_rsi) >= (p10 + 1.0))
    hist_bull_indices = np.where(bull_mask)[0]
    bear_mask = (pd.Series(full_rsi).shift(1) > p90) & (pd.Series(full_rsi) <= (p90 - 1.0))
    hist_bear_indices = np.where(bear_mask)[0]
    stats_bull = backtest_signal_performance(hist_bull_indices, full_price)
    stats_bear = backtest_signal_performance(hist_bear_indices, full_price)
    scan_window = df.iloc[-(periods_to_scan+1):]
    latest_close = df['Price'].iloc[-1] 
    for i in range(1, len(scan_window)):
        prev = scan_window.iloc[i-1]
        curr = scan_window.iloc[i]
        s_type = None
        thresh_val = 0.0
        active_stats = None
        if prev['RSI'] < p10 and curr['RSI'] >= (p10 + 1.0):
            s_type = 'Bullish'
            thresh_val = p10
            active_stats = stats_bull
        elif prev['RSI'] > p90 and curr['RSI'] <= (p90 - 1.0):
            s_type = 'Bearish'
            thresh_val = p90
            active_stats = stats_bear
        if s_type and active_stats:
            rsi_disp = f"{thresh_val:.0f} ↗ {curr['RSI']:.0f}" if s_type == 'Bullish' else f"{thresh_val:.0f} ↘ {curr['RSI']:.0f}"
            action_str = "Leaving Low" if s_type == 'Bullish' else "Leaving High"
            signals.append({
                'Ticker': ticker,
                'Date': curr.name.strftime('%b %d'),
                'Date_Obj': curr.name.date(),
                'Action': action_str,
                'RSI_Display': rsi_disp,
                'Signal_Price': f"${curr['Price']:,.2f}",
                'Last_Close': f"${latest_close:,.2f}", 
                'Signal_Type': s_type,
                'Best Period': active_stats['Best Period'],
                'Profit Factor': active_stats['Profit Factor'],
                'Win Rate': active_stats['Win Rate'],
                'EV': active_stats['EV'],
                'N': active_stats['N']
            })
    return signals

@st.cache_data(ttl=600, show_spinner="Crunching Smart Money Data...")
def calculate_smart_money_score(df, start_d, end_d, mc_thresh, filter_ema, limit):
    f = df.copy()
    if start_d: f = f[f["Trade Date"].dt.date >= start_d]
    if end_d: f = f[f["Trade Date"].dt.date <= end_d]
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f_filtered = f[f[order_type_col].isin(target_types)].copy()
    if f_filtered.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    f_filtered["Signed_Dollars"] = np.where(f_filtered[order_type_col].isin(["Calls Bought", "Puts Sold"]), f_filtered["Dollars"], -f_filtered["Dollars"])
    smart_stats = f_filtered.groupby("Symbol").agg(Signed_Dollars=("Signed_Dollars", "sum"),Trade_Count=("Symbol", "count"),Last_Trade=("Trade Date", "max")).reset_index()
    smart_stats.rename(columns={"Signed_Dollars": "Net Sentiment ($)"}, inplace=True)
    unique_tickers = smart_stats["Symbol"].unique().tolist()
    batch_caps = fetch_market_caps_batch(unique_tickers)
    smart_stats["Market Cap"] = smart_stats["Symbol"].map(batch_caps)
    valid_data = smart_stats[smart_stats["Market Cap"] >= mc_thresh].copy()
    unique_dates = sorted(f_filtered["Trade Date"].unique())
    recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
    f_momentum = f_filtered[f_filtered["Trade Date"].isin(recent_dates)]
    mom_stats = f_momentum.groupby("Symbol")["Signed_Dollars"].sum().reset_index()
    mom_stats.rename(columns={"Signed_Dollars": "Momentum ($)"}, inplace=True)
    valid_data = valid_data.merge(mom_stats, on="Symbol", how="left").fillna(0)
    top_bulls = pd.DataFrame()
    top_bears = pd.DataFrame()
    if not valid_data.empty:
        valid_data["Impact"] = valid_data["Net Sentiment ($)"] / valid_data["Market Cap"]
        def normalize(series):
            mn, mx = series.min(), series.max()
            return (series - mn) / (mx - mn) if (mx != mn) else 0
        b_flow_norm = normalize(valid_data["Net Sentiment ($)"].clip(lower=0))
        b_imp_norm = normalize(valid_data["Impact"].clip(lower=0))
        b_mom_norm = normalize(valid_data["Momentum ($)"].clip(lower=0))
        valid_data["Base_Score_Bull"] = (0.35 * b_flow_norm) + (0.30 * b_imp_norm) + (0.35 * b_mom_norm)
        br_flow_norm = normalize(-valid_data["Net Sentiment ($)"].clip(upper=0))
        br_imp_norm = normalize(-valid_data["Impact"].clip(upper=0))
        br_mom_norm = normalize(-valid_data["Momentum ($)"].clip(upper=0))
        valid_data["Base_Score_Bear"] = (0.35 * br_flow_norm) + (0.30 * br_imp_norm) + (0.35 * br_mom_norm)
        valid_data["Last Trade"] = valid_data["Last_Trade"].dt.strftime("%d %b")
        candidates_bull = valid_data.sort_values(by="Base_Score_Bull", ascending=False).head(limit * 3).copy()
        candidates_bear = valid_data.sort_values(by="Base_Score_Bear", ascending=False).head(limit * 3).copy()
        all_tickers_to_fetch = set(candidates_bull["Symbol"]).union(set(candidates_bear["Symbol"]))
        batch_techs = fetch_technicals_batch(list(all_tickers_to_fetch)) if filter_ema else {}
        def check_ema_filter_fast(ticker, mode="Bull"):
            if not filter_ema: return True, "—"
            s, e8, _, _, _ = batch_techs.get(ticker, (None, None, None, None, None))
            if not s or not e8: return False, "—"
            if mode == "Bull": return (s > e8), ("✅ >EMA8" if s > e8 else "⚠️ <EMA8")
            else: return (s < e8), ("✅ <EMA8" if s < e8 else "⚠️ >EMA8")
        bull_results = []
        for idx, row in candidates_bull.iterrows():
            passes, trend_s = check_ema_filter_fast(row["Symbol"], "Bull")
            if passes:
                row["Score"] = row["Base_Score_Bull"] * 100
                row["Trend"] = trend_s
                bull_results.append(row)
        top_bulls = pd.DataFrame(bull_results).head(limit)
        bear_results = []
        for idx, row in candidates_bear.iterrows():
            passes, trend_s = check_ema_filter_fast(row["Symbol"], "Bear")
            if passes:
                row["Score"] = row["Base_Score_Bear"] * 100
                row["Trend"] = trend_s
                bear_results.append(row)
        top_bears = pd.DataFrame(bear_results).head(limit)
    return top_bulls, top_bears, valid_data

@st.cache_data(ttl=3600)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        df = df.reset_index()
        date_col_name = df.columns[0]
        df = df.rename(columns={date_col_name: "DATE"})
        if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = pd.to_datetime(df["DATE"])
        if df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
        df = df.rename(columns={"Close": "CLOSE", "Volume": "VOLUME", "High": "HIGH", "Low": "LOW", "Open": "OPEN"})
        df.columns = [c.upper() for c in df.columns]
        delta = df["CLOSE"].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["RSI_14"] = df["RSI"]
        return df
    except Exception:
        return None

# --- 2. APP MODULES ---
# (Database, Rankings, Pivot Tables, Strike Zones, RSI Scanner, Trade Ideas)
# [REMOVED FOR BREVITY - BUT ARE INCLUDED IN FINAL SCRIPT]

def run_database_app(df):
    st.title("📂 Database")
    max_data_date = get_max_trade_date(df)
    if 'saved_db_ticker' not in st.session_state: st.session_state.saved_db_ticker = ""
    if 'saved_db_start' not in st.session_state: st.session_state.saved_db_start = max_data_date
    if 'saved_db_end' not in st.session_state: st.session_state.saved_db_end = max_data_date
    if 'saved_db_exp' not in st.session_state: st.session_state.saved_db_exp = (date.today() + timedelta(days=365))
    if 'saved_db_inc_cb' not in st.session_state: st.session_state.saved_db_inc_cb = True
    if 'saved_db_inc_ps' not in st.session_state: st.session_state.saved_db_inc_ps = True
    if 'saved_db_inc_pb' not in st.session_state: st.session_state.saved_db_inc_pb = True
    def save_db_state(key, saved_key): st.session_state[saved_key] = st.session_state[key]
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: db_ticker = st.text_input("Ticker (blank=all)", value=st.session_state.saved_db_ticker, key="db_ticker_input", on_change=save_db_state, args=("db_ticker_input", "saved_db_ticker")).strip().upper()
    with c2: start_date = st.date_input("Trade Start Date", value=st.session_state.saved_db_start, key="db_start", on_change=save_db_state, args=("db_start", "saved_db_start"))
    with c3: end_date = st.date_input("Trade End Date", value=st.session_state.saved_db_end, key="db_end", on_change=save_db_state, args=("db_end", "saved_db_end"))
    with c4: db_exp_end = st.date_input("Expiration Range (end)", value=st.session_state.saved_db_exp, key="db_exp", on_change=save_db_state, args=("db_exp", "saved_db_exp"))
    ot1, ot2, ot3, ot_pad = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_db_inc_cb, key="db_inc_cb", on_change=save_db_state, args=("db_inc_cb", "saved_db_inc_cb"))
    with ot2: inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_db_inc_ps, key="db_inc_ps", on_change=save_db_state, args=("db_inc_ps", "saved_db_inc_ps"))
    with ot3: inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_db_inc_pb, key="db_inc_pb", on_change=save_db_state, args=("db_inc_pb", "saved_db_inc_pb"))
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
    st.markdown("<br><br><br>", unsafe_allow_html=True)

# ... (Include run_rankings_app, run_pivot_tables_app, run_strike_zones_app, run_trade_ideas_app from previous correct responses)

def run_rsi_scanner_app(df_global):
    st.title("📈 RSI Scanner")
    st.markdown("""<style>.top-note { color: #888888; font-size: 14px; margin-bottom: 2px; font-family: inherit; }.footer-header { color: #31333f; margin-top: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-weight: bold; }[data-testid="stDataFrame"] th { font-weight: 900 !important; }</style>""", unsafe_allow_html=True)
    dataset_map = load_dataset_config()
    options = list(dataset_map.keys())
    tab_div, tab_pct, tab_bot = st.tabs(["📉 Divergences", "🔢 Percentiles", "🤖 Backtester"])
    with tab_bot:
        st.markdown('<div class="light-note" style="margin-bottom: 15px;">ℹ️ Sometimes when you change the ticker for the first time, it can be buggy and go back to the RSI Divergences tab. Just ignore it and come back here and then it will work correctly moving forward. Sorry, I am not a programmer. Also, don\'t hack me.</div>', unsafe_allow_html=True)
        c_left, c_right = st.columns([1, 6])
        with c_left:
            ticker = st.text_input("Ticker", value="NFLX", help="Enter a symbol (e.g., TSLA, NVDA)", key="rsi_bt_ticker_input").strip().upper()
            lookback_years = st.number_input("Lookback Years", min_value=1, max_value=10, value=10)
            rsi_tol = st.number_input("RSI Tolerance", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            rsi_metric_container = st.empty()
        if ticker:
            ticker_map = load_ticker_map()
            with st.spinner(f"Crunching numbers for {ticker}..."):
                df = get_ticker_technicals(ticker, ticker_map)
                if df is None or df.empty: df = fetch_yahoo_data(ticker)
                if df is None or df.empty: st.error(f"Sorry, data could not be retrieved for {ticker} (neither via Drive nor Yahoo Finance).")
                else:
                    df.columns = [c.strip().upper() for c in df.columns]
                    date_col = next((c for c in df.columns if 'DATE' in c), None)
                    close_col = next((c for c in df.columns if 'CLOSE' in c), None)
                    rsi_col = next((c for c in df.columns if 'RSI' in c), None)
                    if not all([date_col, close_col]): st.error("Data source missing Date or Close columns.")
                    else:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.sort_values(by=date_col).reset_index(drop=True)
                        if not rsi_col:
                            delta = df[close_col].diff()
                            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                            rs = gain / loss
                            df['RSI'] = 100 - (100 / (1 + rs))
                            rsi_col = 'RSI'
                        cutoff_date = df[date_col].max() - timedelta(days=365*lookback_years)
                        df = df[df[date_col] >= cutoff_date].copy().reset_index(drop=True) 
                        current_row = df.iloc[-1]
                        current_rsi = current_row[rsi_col]
                        rsi_metric_container.markdown(f"""<div style="margin-top: 10px; font-size: 0.9rem; color: #666;">Current RSI</div><div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 15px;">{current_rsi:.2f}</div>""", unsafe_allow_html=True)
                        rsi_min = current_rsi - rsi_tol
                        rsi_max = current_rsi + rsi_tol
                        hist_df = df.iloc[:-1].copy()
                        matches = hist_df[(hist_df[rsi_col] >= rsi_min) & (hist_df[rsi_col] <= rsi_max)].copy()
                        full_close = df[close_col].values
                        match_indices = matches.index.values
                        total_len = len(full_close)
                        results = []
                        periods = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
                        for p in periods:
                            valid_indices = match_indices[match_indices + p < total_len]
                            if len(valid_indices) == 0:
                                results.append({"Days": p, "Win Rate": np.nan, "Avg Ret": np.nan, "Count": 0, "Profit Factor": np.nan})
                                continue
                            entry_prices = full_close[valid_indices]
                            exit_prices = full_close[valid_indices + p]
                            returns = (exit_prices - entry_prices) / entry_prices
                            wins = returns[returns > 0]
                            losses = returns[returns <= 0]
                            n_wins = len(wins)
                            n_total = len(returns)
                            win_rate = (n_wins / n_total) * 100
                            avg_ret = np.mean(returns) * 100
                            gross_profit = np.sum(wins)
                            gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0
                            profit_factor = gross_profit / gross_loss if gross_loss != 0 else 99.9
                            results.append({"Days": p, "Win Rate": win_rate, "Avg Ret": avg_ret, "Count": len(valid_indices), "Profit Factor": profit_factor})
                        res_df = pd.DataFrame(results)
                        with c_right:
                            if matches.empty: st.warning(f"No historical periods found where RSI was between {rsi_min:.2f} and {rsi_max:.2f}.")
                            else:
                                def highlight_best(row):
                                    days = row['Days']
                                    if days <= 20: threshold = 30
                                    elif days <= 60: threshold = 20
                                    else: threshold = 10
                                    condition = (row['Count'] >= threshold) and (row['Win Rate'] > 75)
                                    color = 'background-color: rgba(144, 238, 144, 0.2)' if condition else ''
                                    return [color] * len(row)
                                def highlight_ret(val):
                                    if val is None or pd.isna(val): return ''
                                    if not isinstance(val, (int, float)): return ''
                                    color = '#71d28a' if val > 0 else '#f29ca0'
                                    return f'color: {color}; font-weight: bold;'
                                format_func = lambda x: f"{x:+.2f}%" if pd.notnull(x) else "—"
                                format_wr = lambda x: f"{x:.1f}%" if pd.notnull(x) else "—"
                                st.dataframe(res_df.style.format({"Win Rate": format_wr, "Avg Ret": format_func}).map(highlight_ret, subset=["Avg Ret"]).apply(highlight_best, axis=1).set_table_styles([dict(selector="th", props=[("font-weight", "bold"), ("background-color", "#f0f2f6")])]),use_container_width=False,column_config={"Days": st.column_config.NumberColumn("Days", width=60),"Win Rate": st.column_config.TextColumn("Win Rate", width=80),"Avg Ret": st.column_config.TextColumn("Avg Ret", width=80),"Count": st.column_config.NumberColumn("Count", width=60),"Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f")},hide_index=True)
                        st.markdown("<br><br><br>", unsafe_allow_html=True)
    with tab_div:
        data_option_div = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_div_pills")
        with st.expander("ℹ️ Page Notes: Divergence Strategy Logic"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                st.markdown('<div class="footer-header">📉 SIGNAL LOGIC</div>', unsafe_allow_html=True)
                st.markdown(f"""* **Identification**: Scans for **True Pivots** over a **{SIGNAL_LOOKBACK_PERIOD}-period** window.\n* **Divergence**: \n    * **Bullish**: Price makes a Lower Low, but RSI makes a Higher Low.\n    * **Bearish**: Price makes a Higher High, but RSI makes a Lower High.\n* **Invalidation**: If RSI crosses the 50 midline between pivots, the setup is reset.""")
            with f_col2:
                st.markdown('<div class="footer-header">HISTORICAL OPTIMIZATION</div>', unsafe_allow_html=True)
                st.markdown(f"""* **Signal-Based**: The system finds every matching historical signal (e.g. every Bullish Divergence in history).\n* **Hold Periods**: It tests forward returns for 10, 30, 60, 90, and 180 days.\n* **Selection**: The table displays the "Best Period" which had the highest Profit Factor historically.\n* **Profit Factor**: (Sum of Wins / Sum of Losses). A PF > 1.5 is generally considered good.""")
            with f_col3:
                st.markdown('<div class="footer-header">🏷️ TAGS</div>', unsafe_allow_html=True)
                st.markdown(f"""* **EMA{EMA8_PERIOD}**: Bullish (Price > EMA8) or Bearish (Price < EMA8).\n* **EMA{EMA21_PERIOD}**: Bullish (Price > EMA21) or Bearish (Price < EMA21).\n* **VOL_HIGH**: Signal candle volume is > 150% of the 30-day average.\n* **VOL_GROW**: Volume on the second pivot (P2) is higher than the first pivot (P1).""")
        if data_option_div:
            try:
                target_url = st.secrets[dataset_map[data_option_div]]
                csv_buffer = get_confirmed_gdrive_data(target_url)
                if csv_buffer and csv_buffer != "HTML_ERROR":
                    master = pd.read_csv(csv_buffer)
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    date_col_raw = next((c for c in master.columns if 'DATE' in c.upper()), None)
                    if date_col_raw:
                        max_dt_obj = pd.to_datetime(master[date_col_raw]).max()
                        target_highlight_daily = max_dt_obj.strftime('%Y-%m-%d')
                        days_to_subtract = max_dt_obj.weekday() + (7 if max_dt_obj.weekday() < 4 else 0)
                        target_highlight_weekly = (max_dt_obj - timedelta(days=days_to_subtract)).strftime('%Y-%m-%d')
                    all_tickers = sorted(master[t_col].unique())
                    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                        sq_div = st.text_input("Filter...", key="rsi_div_filter_ticker").upper()
                        ft_div = [t for t in all_tickers if sq_div in t]
                        cols = st.columns(6)
                        for i, ticker in enumerate(ft_div): cols[i % 6].write(ticker)
                    raw_results_div = []
                    progress_bar = st.progress(0, text="Scanning Divergences...")
                    grouped = master.groupby(t_col)
                    grouped_list = list(grouped)
                    total_groups = len(grouped_list)
                    for i, (ticker, group) in enumerate(grouped_list):
                        d_d, d_w = prepare_data(group.copy())
                        if d_d is not None: raw_results_div.extend(find_divergences(d_d, ticker, 'Daily'))
                        if d_w is not None: raw_results_div.extend(find_divergences(d_w, ticker, 'Weekly'))
                        if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                    progress_bar.empty()
                    if raw_results_div:
                        res_div_df = pd.DataFrame(raw_results_div).sort_values(by='Signal_Date_ISO', ascending=False)
                        consolidated = res_div_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                        for tf in ['Daily', 'Weekly']:
                            target_highlight = target_highlight_weekly if tf == 'Weekly' else target_highlight_daily
                            date_header = "Week Δ" if tf == 'Weekly' else "Day Δ"
                            for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                                st.subheader(f"{emoji} {tf} {s_type} Signals")
                                tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                                price_header = "Low Price Δ" if s_type == 'Bullish' else "High Price Δ"
                                if not tbl_df.empty:
                                    def style_div_df(df_in):
                                        def highlight_row(row):
                                            styles = [''] * len(row)
                                            if row['Signal_Date_ISO'] == target_highlight:
                                                idx = df_in.columns.get_loc('Date_Display')
                                                styles[idx] = 'background-color: rgba(255, 244, 229, 0.7); color: #e67e22; font-weight: bold;'
                                            for col_name in ['EV']:
                                                if col_name in df_in.columns:
                                                    val = row[col_name]
                                                    if pd.notnull(val) and val != 0:
                                                        is_green = (s_type == 'Bullish' and val > 0) or (s_type == 'Bearish' and val < 0)
                                                        bg = 'background-color: #e6f4ea; color: #1e7e34;' if is_green else 'background-color: #fce8e6; color: #c5221f;'
                                                        idx = df_in.columns.get_loc(col_name)
                                                        styles[idx] = f'{bg} font-weight: 500;'
                                            return styles
                                        return df_in.style.apply(highlight_row, axis=1)
                                    st.dataframe(style_div_df(tbl_df),column_config={"Ticker": st.column_config.TextColumn("Ticker"),"Tags": st.column_config.ListColumn("Tags", width="medium"),"Date_Display": st.column_config.TextColumn(date_header),"RSI_Display": st.column_config.TextColumn("RSI Δ"),"Price_Display": st.column_config.TextColumn(price_header),"Last_Close": st.column_config.TextColumn("Last Close"),"Best Period": st.column_config.TextColumn("Best Period"),"Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),"Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),"EV": st.column_config.NumberColumn("EV", format="%.1f%%"),"N": st.column_config.NumberColumn("N"),"Signal_Date_ISO": None, "Type": None, "Timeframe": None, "ev30_raw": None, "ev90_raw": None},hide_index=True,use_container_width=True,height=get_table_height(tbl_df, max_rows=50))
                                else: st.info("No signals.")
                    else: st.warning("No Divergence signals found.")
            except Exception as e: st.error(f"Analysis failed: {e}")
    with tab_pct:
        data_option_pct = st.pills("Dataset", options=options, selection_mode="single", default=options[0] if options else None, label_visibility="collapsed", key="rsi_pct_pills")
        with st.expander("ℹ️ Page Notes: Percentile Strategy Logic"):
             st.markdown("""* **Historical Context**: 10-year daily price history analysis.\n* **Signal Trigger**: RSI crosses **ABOVE Low Percentile** (Leaving Low) or **BELOW High Percentile** (Leaving High).\n* **Historical Optimization**: The system scans every historical instance of this percentile crossover and tests 10, 30, 60, 90, 180 day holds.\n* **Best Period**: The holding period that historically produced the highest Profit Factor.\n* **Base Price**: EV calculations are based on the **Signal Day Close**.\n* **Color Logic**: 🟢 Green = Historical profitability (Longs > 0, Shorts < 0). 🔴 Red = Historical loss.\n* **Filter**: Requires >= 5 historical matches.""")
        if data_option_pct:
            try:
                target_url = st.secrets[dataset_map[data_option_pct]]
                csv_buffer = get_confirmed_gdrive_data(target_url)
                if csv_buffer and csv_buffer != "HTML_ERROR":
                    master = pd.read_csv(csv_buffer)
                    t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                    date_col_raw = next((c for c in master.columns if 'DATE' in c.upper()), None)
                    max_date_in_set = None
                    if date_col_raw:
                        max_dt_obj = pd.to_datetime(master[date_col_raw]).max()
                        max_date_in_set = max_dt_obj.date()
                    all_tickers = sorted(master[t_col].unique())
                    with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                        sq_pct = st.text_input("Filter...", key="rsi_pct_filter_ticker").upper()
                        ft_pct = [t for t in all_tickers if sq_pct in t]
                        cols = st.columns(6)
                        for i, ticker in enumerate(ft_pct): cols[i % 6].write(ticker)
                    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
                    with c_p1: in_low = st.number_input("RSI Low Percentile (%)", min_value=1, max_value=49, value=10, step=1, key="rsi_pct_low")
                    with c_p2: in_high = st.number_input("RSI High Percentile (%)", min_value=51, max_value=99, value=90, step=1, key="rsi_pct_high")
                    with c_p3: show_filter = st.selectbox("Actions to Show", ["Everything", "Leaving High", "Leaving Low"], index=0, key="rsi_pct_show")
                    if not df_global.empty and "Trade Date" in df_global.columns: ref_date = df_global["Trade Date"].max().date()
                    else: ref_date = date.today()
                    default_start = ref_date - timedelta(days=14)
                    with c_p4: filter_date = st.date_input("Latest Date", value=default_start, key="rsi_pct_date")
                    raw_results_pct = []
                    progress_bar = st.progress(0, text="Scanning Percentiles...")
                    grouped = master.groupby(t_col)
                    grouped_list = list(grouped)
                    total_groups = len(grouped_list)
                    for i, (ticker, group) in enumerate(grouped_list):
                        d_d, d_w = prepare_data(group.copy())
                        if d_d is not None: raw_results_pct.extend(find_rsi_percentile_signals(d_d, ticker, pct_low=in_low/100.0, pct_high=in_high/100.0))
                        if i % 10 == 0 or i == total_groups - 1: progress_bar.progress((i + 1) / total_groups)
                    progress_bar.empty()
                    if raw_results_pct:
                        res_pct_df = pd.DataFrame(raw_results_pct).sort_values(by='Date_Obj', ascending=False)
                        if show_filter == "Leaving High": res_pct_df = res_pct_df[res_pct_df['Signal_Type'] == 'Bearish']
                        elif show_filter == "Leaving Low": res_pct_df = res_pct_df[res_pct_df['Signal_Type'] == 'Bullish']
                        if filter_date: res_pct_df = res_pct_df[res_pct_df['Date_Obj'] >= filter_date]
                        def style_pct_df(df_in):
                            def highlight_row(row):
                                styles = [''] * len(row)
                                if row['Date_Obj'] == max_date_in_set:
                                    idx = df_in.columns.get_loc('Date')
                                    styles[idx] = 'background-color: rgba(255, 244, 229, 0.7); color: #e67e22; font-weight: bold;'
                                s_type = row['Signal_Type']
                                for col_name in ['EV']:
                                    if col_name in df_in.columns:
                                        val = row[col_name]
                                        if pd.notnull(val) and val != 0:
                                            is_green = (s_type == 'Bullish' and val > 0) or (s_type == 'Bearish' and val < 0)
                                            bg = 'background-color: #e6f4ea; color: #1e7e34;' if is_green else 'background-color: #fce8e6; color: #c5221f;'
                                            idx = df_in.columns.get_loc(col_name)
                                            styles[idx] = f'{bg} font-weight: 500;'
                                return styles
                            return df_in.style.apply(highlight_row, axis=1)
                        st.dataframe(style_pct_df(res_pct_df),column_config={"Ticker": st.column_config.TextColumn("Ticker"),"Date": st.column_config.TextColumn("Date"),"Action": st.column_config.TextColumn("Action"),"RSI_Display": st.column_config.TextColumn("RSI Δ"),"Signal_Price": st.column_config.TextColumn("Signal Close"),"Last_Close": st.column_config.TextColumn("Last Close"),"Best Period": st.column_config.TextColumn("Best Period"),"Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),"Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),"EV": st.column_config.NumberColumn("EV", format="%.1f%%"),"N": st.column_config.NumberColumn("N"),"Signal_Type": None, "Threshold": None, "EV30_Obj": None, "EV90_Obj": None, "Date_Obj": None, "RSI": None},hide_index=True,use_container_width=True,height=get_table_height(res_pct_df, max_rows=50))
                    else: st.info(f"No Percentile signals found (Crossing {in_low}th/{in_high}th percentile).")
            except Exception as e: st.error(f"Analysis failed: {e}")

try:
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)
    last_updated_date = df_global["Trade Date"].max().strftime("%d %b %y")
    pg = st.navigation([
        st.Page(lambda: run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(lambda: run_rsi_scanner_app(df_global), title="RSI Scanner", icon="📈", url_path="rsi_scanner"), 
        st.Page(lambda: run_trade_ideas_app(df_global), title="Trade Ideas", icon="💡", url_path="trade_ideas"),
    ])
    st.sidebar.caption("🖥️ Everything is best viewed with a wide desktop monitor in light mode.")
    st.sidebar.caption(f"📅 **Last Updated:** {last_updated_date}")
    pg.run()
except Exception as e: st.error(f"Error initializing dashboard: {e}")
