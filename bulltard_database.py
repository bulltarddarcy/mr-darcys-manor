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
import altair as alt
import google.generativeai as genai

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
    try:
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
            mask = df["Error"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
            df = df[~mask]
            
        return df
    except Exception as e:
        st.error(f"Error loading global data: {e}")
        return pd.DataFrame()

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
        expiry_date = datetime.strptime(val, "%d %b %y").date()
        today = date.today()
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
        return None

def load_dataset_config():
    try:
        if "URL_CONFIG" not in st.secrets:
            # Fallback for dev environment
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
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_d['RSI'] = 100 - (100 / (1 + rs))
        
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    
    # Weekly Data
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

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, periods_to_scan=10):
    signals = []
    if len(df) < 200: return signals
    
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    
    if hist_df.empty: return signals
    
    p10 = hist_df['RSI'].quantile(pct_low)
    p90 = hist_df['RSI'].quantile(pct_high)
    
    if len(df) < periods_to_scan + 2: return signals
    
    rsi_vals = hist_df['RSI'].values
    price_vals = hist_df['Price'].values
    
    scan_window = df.iloc[-(periods_to_scan+1):]
    
    for i in range(1, len(scan_window)):
        prev = scan_window.iloc[i-1]
        curr = scan_window.iloc[i]
        
        s_type = None
        desc_str = ""
        thresh_val = 0.0
        
        if prev['RSI'] < p10 and curr['RSI'] >= p10:
            s_type = 'Bullish'
            desc_str = f"Leaving Low {int(pct_low*100)}%"
            thresh_val = p10
            
        elif prev['RSI'] > p90 and curr['RSI'] <= p90:
            s_type = 'Bearish'
            desc_str = f"Leaving High {int(pct_high*100)}%"
            thresh_val = p90
            
        if s_type:
            ev30 = calculate_ev_data_numpy(rsi_vals, price_vals, curr['RSI'], 30, curr['Price'])
            ev90 = calculate_ev_data_numpy(rsi_vals, price_vals, curr['RSI'], 90, curr['Price'])
            
            valid_30 = ev30 and ev30['n'] >= 5
            valid_90 = ev90 and ev90['n'] >= 5
            
            if valid_30 or valid_90:
                signals.append({
                    'Ticker': ticker,
                    'Date': curr.name.strftime('%Y-%m-%d'),
                    'RSI': curr['RSI'],
                    'Signal': desc_str,
                    'Signal_Type': s_type,
                    'Threshold': thresh_val,
                    'EV30_Obj': ev30 if valid_30 else None,
                    'EV90_Obj': ev90 if valid_90 else None,
                    'Date_Obj': curr.name.date()
                })
            
    return signals

# --- 4. NEW TRADE IDEAS MODULE (Updated for Macro Scanner) ---

@st.cache_data(ttl=3600)
def load_ticker_map():
    url = st.secrets.get("URL_TICKER_MAP", URL_TICKER_MAP_DEFAULT)
    try:
        csv_buffer = get_confirmed_gdrive_data(url)
        if csv_buffer and csv_buffer != "HTML_ERROR":
            df = pd.read_csv(csv_buffer, header=None)
            if len(df.columns) >= 2:
                return pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].str.upper().str.strip()).to_dict()
    except Exception as e:
        st.error(f"Failed to load Ticker Map: {e}")
    return {}

@st.cache_data(ttl=300)
def get_ticker_technicals(ticker, ticker_map):
    file_id = ticker_map.get(ticker.upper())
    if not file_id:
        return None
    
    url = f"https://docs.google.com/uc?export=download&id={file_id}"
    try:
        csv_buffer = get_confirmed_gdrive_data(url)
        if csv_buffer and csv_buffer != "HTML_ERROR":
            df = pd.read_csv(csv_buffer)
            df.columns = [c.strip().upper() for c in df.columns]
            
            date_col = next((c for c in df.columns if 'DATE' in c), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(date_col)
                return df
    except:
        pass
    return None

def analyze_trade_setup(ticker, trade_type, df_tech, df_flow):
    score = 5 
    reasons = []
    recommendation = "Neutral / Wait"
    
    last = df_tech.iloc[-1]
    price = last.get('CLOSE', 0)
    ema8 = last.get('EMA_8', last.get('EMA8', 0))
    ema21 = last.get('EMA_21', last.get('EMA21', 0))
    sma200 = last.get('SMA_200', last.get('SMA200', 0))
    rsi = last.get('RSI_14', last.get('RSI', 50))
    
    levels = [l for l in [ema21, sma200] if l > 0]
    nearest_support = max([l for l in levels if l < price], default=price*0.9)
    nearest_resistance = min([l for l in levels if l > price], default=price*1.1)

    is_bullish = False
    is_bearish = False

    if trade_type in ["Buy Calls", "Sell Puts", "Buy Shares", "Risk Reversal"]:
        if price > ema8 and price > ema21:
            score += 2
            reasons.append("✅ Price is strong (Above EMA 8 & 21)")
            is_bullish = True
        elif price < ema21:
            score -= 2
            reasons.append("⚠️ Price is weak (Below EMA 21)")
            
        if price > sma200:
            score += 1
            reasons.append("✅ In long-term uptrend (Above SMA 200)")
        else:
            reasons.append("⚠️ Below SMA 200 (Long-term downtrend)")
            
        if rsi < 40:
            score += 1
            reasons.append("✅ RSI is oversold (Potential bounce)")
        elif rsi > 70:
            score -= 1
            reasons.append("⚠️ RSI is overbought (Risk of pullback)")

    else: 
        if price < ema8 and price < ema21:
            score += 2
            reasons.append("✅ Price is weak (Below EMA 8 & 21)")
            is_bearish = True
        elif price > ema21:
            score -= 2
            reasons.append("⚠️ Price is strong (Above EMA 21)")
            
        if price < sma200:
            score += 1
            reasons.append("✅ In long-term downtrend (Below SMA 200)")
            
        if rsi > 60:
            score += 1
            reasons.append("✅ RSI is elevated (Room to fall)")
        elif rsi < 30:
            score -= 1
            reasons.append("⚠️ RSI is oversold (Risk of bounce)")

    if not df_flow.empty:
        flow_ticker = df_flow[df_flow['Symbol'] == ticker]
        if not flow_ticker.empty:
            calls = flow_ticker[flow_ticker['Order Type'] == "Calls Bought"]['Dollars'].sum()
            puts_sold = flow_ticker[flow_ticker['Order Type'] == "Puts Sold"]['Dollars'].sum()
            puts_bought = flow_ticker[flow_ticker['Order Type'] == "Puts Bought"]['Dollars'].sum()
            
            whales = flow_ticker[flow_ticker['Dollars'] >= 100000].copy()
            if not whales.empty:
                w_calls = whales[whales['Order Type'] == "Calls Bought"]['Dollars'].sum()
                w_puts_buy = whales[whales['Order Type'] == "Puts Bought"]['Dollars'].sum()
                
                if w_calls > w_puts_buy:
                    score += 1
                    reasons.append(f"🐋 Whale Alert: Significant large bullish trades found (${w_calls:,.0f}).")
                elif w_puts_buy > w_calls:
                    score -= 1
                    reasons.append(f"🐋 Whale Alert: Significant large bearish trades found (${w_puts_buy:,.0f}).")
            
            net_bullish = calls + puts_sold - puts_bought
            
            if trade_type in ["Buy Calls", "Sell Puts", "Buy Shares", "Risk Reversal"]:
                if net_bullish > 1_000_000:
                    score += 2
                    reasons.append(f"✅ Strong Bullish Flow (+${net_bullish/1e6:.1f}M)")
                elif net_bullish < -1_000_000:
                    score -= 2
                    reasons.append(f"⚠️ Bearish Flow detected (${net_bullish/1e6:.1f}M)")
            else:
                if net_bullish < -1_000_000:
                    score += 2
                    reasons.append(f"✅ Strong Bearish Flow ({net_bullish/1e6:.1f}M)")
                elif net_bullish > 1_000_000:
                    score -= 2
                    reasons.append(f"⚠️ Bullish Flow detected (+${net_bullish/1e6:.1f}M)")
    
    final_score = min(max(score, 0), 10)
    
    if final_score >= 6:
        if trade_type == "Risk Reversal":
            put_strike = math.floor(nearest_support)
            call_strike = math.ceil(price * 1.02)
            recommendation = f"Bullish Risk Reversal: Sell {put_strike} Put (Support), Buy {call_strike} Call."
        elif trade_type == "Sell Puts":
            strike = math.floor(nearest_support)
            recommendation = f"Sell Puts at ${strike} level (near support)."
        elif trade_type == "Buy Calls":
            strike = math.ceil(price)
            recommendation = f"Buy ATM/OTM Calls (Strike ${strike}+)."
        elif trade_type == "Buy Puts":
            strike = math.floor(price)
            recommendation = f"Buy ATM/OTM Puts (Strike ${strike}-)."
        elif trade_type == "Buy Shares":
            recommendation = "Accumulate Shares with stop below EMA21."
    else:
        recommendation = "Setup weak. Wait for better alignment."

    return final_score, reasons, recommendation

# Helper function to fetch and truncate CSV data for the AI
def fetch_and_prepare_ai_context(url, label, max_rows=90):
    try:
        csv_buffer = get_confirmed_gdrive_data(url)
        if csv_buffer and csv_buffer != "HTML_ERROR":
            df = pd.read_csv(csv_buffer)
            # Find the date column to sort
            date_col = next((c for c in df.columns if 'DATE' in c.upper()), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.sort_values(by=date_col)
            
            # If multiple tickers, we need to keep the structure but truncate per group
            t_col = next((c for c in df.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
            
            if t_col:
                # Group by ticker, take last N rows, then combine back to CSV string
                df_trunc = df.groupby(t_col).tail(max_rows)
                return f"\n=== {label} DATA (Last {max_rows} rows per ticker) ===\n" + df_trunc.to_csv(index=False)
            else:
                # If it's a macro file (no ticker column, just dates), just take tail
                df_trunc = df.tail(max_rows)
                return f"\n=== {label} DATA (Last {max_rows} rows) ===\n" + df_trunc.to_csv(index=False)
    except Exception as e:
        return f"\n=== {label} DATA ERROR: {str(e)} ===\n"
    return f"\n=== {label} DATA NOT FOUND ===\n"

def run_trade_ideas_app(df_global):
    st.title("💡 Trade Ideas Generator")
    st.caption("Combine Technicals, Flows, and AI for high-conviction setups.")
    
    ticker_map = load_ticker_map()
    
    # Renamed Tabs to reflect new structure
    tabs = st.tabs(["🔎 Analyze Ticker", "🤖 Macro Scanner (AI)", "🏆 Top 3 Scans"])
    
    # --- TAB 1: TICKER ANALYZER ---
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            default_t = "NVDA"
            t_input = st.text_input("Ticker Symbol", value=default_t).upper().strip()
        with c2:
            trade_type = st.selectbox("Trade Type", ["Buy Calls", "Sell Puts", "Buy Puts", "Buy Shares", "Risk Reversal"])
        with c3:
            duration = st.selectbox("Duration", ["Swing (1-4 Weeks)", "Leap (>1 Year)", "Scalp (Daily)"])
            
        if st.button("Analyze Trade", type="primary"):
            with st.spinner(f"Pulling data for {t_input}..."):
                tech_df = get_ticker_technicals(t_input, ticker_map)
                
                if tech_df is None:
                    st.error(f"Could not load technical data for {t_input}. Check if it exists in the Ticker Map.")
                else:
                    score, reasons, rec = analyze_trade_setup(t_input, trade_type, tech_df, df_global)
                    
                    st.markdown("---")
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        st.metric("Conviction Score", f"{score}/10")
                        if score >= 7: st.success("High Conviction Setup")
                        elif score <= 4: st.error("Low Conviction Setup")
                        else: st.warning("Neutral / Mixed Setup")
                            
                    with res_col2:
                        st.subheader("Observations")
                        for r in reasons: st.write(r)
                        st.info(f"**Recommendation:** {rec}")
                            
                    st.markdown("### 📉 Snapshot")
                    cols_to_plot = []
                    if 'CLOSE' in tech_df.columns: cols_to_plot.append('CLOSE')
                    if 'EMA_8' in tech_df.columns: cols_to_plot.append('EMA_8')
                    if 'EMA8' in tech_df.columns: cols_to_plot.append('EMA8') 
                    if 'EMA_21' in tech_df.columns: cols_to_plot.append('EMA_21')
                    if 'EMA21' in tech_df.columns: cols_to_plot.append('EMA21')
                    if 'SMA_200' in tech_df.columns: cols_to_plot.append('SMA_200')
                    if 'SMA200' in tech_df.columns: cols_to_plot.append('SMA200')
                    
                    if cols_to_plot:
                        chart_df = tech_df.tail(90).copy()
                        rename_map = {c: c.replace('_', '') for c in cols_to_plot}
                        chart_df = chart_df[cols_to_plot].rename(columns=rename_map)
                        
                        d_col = next((c for c in tech_df.columns if 'DATE' in c), None)
                        if d_col: 
                            chart_df.index = tech_df.tail(90)[d_col]
                            chart_df.index.name = "Date"
                        
                        chart_df = chart_df.reset_index()

                        melted_df = chart_df.melt('Date', var_name='Metric', value_name='Price')
                        
                        c = alt.Chart(melted_df).mark_line().encode(
                            x='Date:T',
                            y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
                            color='Metric:N',
                            tooltip=['Date', 'Metric', 'Price']
                        ).interactive()
                        
                        st.altair_chart(c, use_container_width=True)
                    else:
                        st.write("Chart data unavailable.")

    # --- TAB 2: MACRO SCANNER (AI) ---
    with tabs[1]:
        st.markdown("#### 🤖 AI Macro Portfolio Manager")
        st.info("This module ingests the Darcy, SP100, NQ100, and Macro datasets to generate a comprehensive strategy report.")
        
        if st.button("Run Global Macro Scan"):
            if "GOOGLE_API_KEY" not in st.secrets:
                st.error("Missing GOOGLE_API_KEY in secrets.toml")
            elif "URL_Prompt" not in st.secrets:
                st.error("Missing URL_Prompt in secrets.toml")
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                    
                    with st.spinner("Step 1/3: Fetching System Prompt..."):
                        prompt_url = st.secrets["URL_Prompt"]
                        prompt_buffer = get_confirmed_gdrive_data(prompt_url)
                        if not prompt_buffer or prompt_buffer == "HTML_ERROR":
                            st.error("Failed to load prompt file from URL.")
                            st.stop()
                        system_prompt = prompt_buffer.getvalue()

                    with st.spinner("Step 2/3: Ingesting & Pre-processing Datasets..."):
                        # Ingest all required datasets
                        # NOTE: Truncating to 90 rows to fit context window efficiently while keeping trend data
                        context_data = ""
                        context_data += fetch_and_prepare_ai_context(st.secrets["URL_DARCY"], "DARCY WATCHLIST", 90)
                        context_data += fetch_and_prepare_ai_context(st.secrets["URL_SP100"], "S&P 100", 90)
                        context_data += fetch_and_prepare_ai_context(st.secrets["URL_NQ100"], "NASDAQ 100", 90)
                        context_data += fetch_and_prepare_ai_context(st.secrets["URL_MACRO"], "MACRO INDICATORS", 90)
                        
                        full_prompt = f"{system_prompt}\n\n==========\nLIVE DATA CONTEXT:\n{context_data}\n=========="

                    with st.spinner("Step 3/3: AI Analysis in Progress (this may take 30-60s)..."):
                        # Use gemini-pro-1.5-flash for larger context window handling if available, else standard pro
                        # Fallback logic for safety
                        try:
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            response = model.generate_content(full_prompt)
                        except:
                            model = genai.GenerativeModel('gemini-pro') 
                            response = model.generate_content(full_prompt)
                            
                        st.success("Analysis Complete!")
                        st.markdown("---")
                        st.markdown(response.text)
                        
                except Exception as e:
                    st.error(f"AI Pipeline Failed: {e}")


    # --- TAB 3: TOP 3 SCANNER (Synced) ---
    with tabs[2]:
        st.markdown("#### 🏆 Automated Opportunity Scanner")
        st.write("Scans the Top 20 'Smart Money' tickers for the best technical alignment.")
        
        if st.button("Scan Top 20"):
            progress = st.progress(0, text="Initializing scan...")
            
            max_date = df_global["Trade Date"].max()
            start_date_filter = max_date - timedelta(days=14)
            
            mask = (df_global["Trade Date"] >= start_date_filter) & (df_global["Trade Date"] <= max_date)
            f_filtered = df_global[mask & df_global['Order Type'].isin(["Calls Bought", "Puts Sold", "Puts Bought"])].copy()
            
            f_filtered["Signed_Dollars"] = np.where(
                f_filtered['Order Type'].isin(["Calls Bought", "Puts Sold"]), 
                f_filtered["Dollars"], -f_filtered["Dollars"]
            )
            
            smart_stats = f_filtered.groupby("Symbol").agg(
                Signed_Dollars=("Signed_Dollars", "sum")
            ).reset_index()
            
            smart_stats["Market Cap"] = smart_stats["Symbol"].apply(lambda x: get_market_cap(x))
            
            # Synced 10B Market Cap Filter
            smart_stats = smart_stats[smart_stats["Market Cap"] >= 1e10]

            unique_dates = sorted(f_filtered["Trade Date"].unique())
            recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
            f_mom = f_filtered[f_filtered["Trade Date"].isin(recent_dates)]
            mom_stats = f_mom.groupby("Symbol")["Signed_Dollars"].sum().reset_index().rename(columns={"Signed_Dollars": "Momentum"})
            
            smart_stats = smart_stats.merge(mom_stats, on="Symbol", how="left").fillna(0)
            valid_data = smart_stats[smart_stats["Market Cap"] > 0].copy()
            
            if valid_data.empty:
                st.warning("Not enough data to run Smart Money logic.")
            else:
                valid_data["Impact"] = valid_data["Signed_Dollars"] / valid_data["Market Cap"]
                
                def normalize(series):
                    mn, mx = series.min(), series.max()
                    return (series - mn) / (mx - mn) if (mx != mn) else 0

                b_flow = valid_data["Signed_Dollars"].clip(lower=0)
                b_imp = valid_data["Impact"].clip(lower=0)
                b_mom = valid_data["Momentum"].clip(lower=0)
                
                valid_data["Score_Bull"] = (
                    (0.40 * normalize(b_flow)) + 
                    (0.30 * normalize(b_imp)) + 
                    (0.30 * normalize(b_mom))
                ) * 100
                
                top_tickers = valid_data.sort_values(by=["Score_Bull", "Signed_Dollars"], ascending=[False, False]).head(20)["Symbol"].tolist()
                
                with st.expander(f"View Scanned Tickers ({len(top_tickers)})"):
                    st.write(", ".join(top_tickers))
                
                candidates = []
                
                for i, t in enumerate(top_tickers):
                    progress.progress((i+1)/20, text=f"Analyzing {t}...")
                    t_df = get_ticker_technicals(t, ticker_map)
                    
                    if t_df is not None:
                        score, reasons, rec = analyze_trade_setup(t, "Buy Calls", t_df, df_global)
                        candidates.append({
                            "Ticker": t,
                            "Score": score,
                            "Price": t_df.iloc[-1].get('CLOSE'),
                            "Reasons": reasons,
                            "Rec": rec
                        })
                
                progress.empty()
                
                candidates = sorted(candidates, key=lambda x: x['Score'], reverse=True)[:3]
                
                st.success("Scan Complete!")
                
                cols = st.columns(3)
                for i, cand in enumerate(candidates):
                    with cols[i]:
                        with st.container(border=True):
                            st.markdown(f"### #{i+1} {cand['Ticker']}")
                            st.metric("Score", f"{cand['Score']}/10", f"${cand['Price']:.2f}")
                            st.caption(f"**Action:** {cand['Rec']}")
                            for r in cand['Reasons']:
                                st.caption(r)

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
        st.Page(lambda: run_trade_ideas_app(df_global), title="Trade Ideas", icon="💡", url_path="trade_ideas"),
    ])

    st.sidebar.caption("🖥️ Everything is best viewed with a wide desktop monitor in light mode.")
    st.sidebar.caption(f"📅 **Last Updated:** {last_updated_date}")
    
    pg.run()
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")
