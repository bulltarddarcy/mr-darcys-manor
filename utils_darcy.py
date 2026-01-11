# utils_darcy.py
# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
import requests
import re
from datetime import date, timedelta
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT SHARED UTILS ---
from utils_shared import get_gdrive_binary_data, get_table_height

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# --- CONSTANTS: DATABASE APP ---
DB_DEFAULT_EXPIRY_OFFSET = 365
DB_TABLE_MAX_ROWS = 30
DB_DATE_FMT = "%d %b %y"
# Styling
STYLE_BULL_CSS = 'background-color: rgba(113, 210, 138, 0.15); color: #71d28a; font-weight: 600;'
STYLE_BEAR_CSS = 'background-color: rgba(242, 156, 160, 0.15); color: #f29ca0; font-weight: 600;'

# --- CONSTANTS: RANKING APP ---
RANK_LOOKBACK_DAYS = 14
RANK_LIMIT_DEFAULT = 20
RANK_MC_THRESHOLDS = {"0B": 0, "2B": 2e9, "10B": 1e10, "50B": 5e10, "100B": 1e11}
RANK_SM_WEIGHTS = {'Sentiment': 0.35, 'Impact': 0.30, 'Momentum': 0.35}
RANK_CONVICTION_DIVISOR = 25.0
RANK_TOP_IDEAS_COUNT = 3

# --- CONSTANTS: PIVOT APP ---
PIVOT_NOTIONAL_MAP = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}
PIVOT_MC_MAP = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}
PIVOT_TABLE_FMT = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}

# --- CONSTANTS: STRIKE ZONES APP ---
SZ_DEFAULT_EXP_OFFSET = 365
SZ_DEFAULT_FIXED_SIZE = 10
SZ_AUTO_WIDTH_DENOM = 12.0
SZ_AUTO_STEPS = [1, 2, 5, 10, 25, 50, 100]
SZ_BUCKET_BINS = [0, 7, 30, 60, 90, 120, 180, 365, 10000]
SZ_BUCKET_LABELS = ["0-7d", "8-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-365d", ">365d"]

# REFRESH TIME: 600 seconds = 10 minutes
CACHE_TTL = 600 

# --- DATA LOADERS & GOOGLE DRIVE UTILS ---

@st.cache_data(ttl=CACHE_TTL)
def get_parquet_config():
    config = {}
    try:
        raw_config = st.secrets.get("PARQUET_CONFIG", "")
        if not raw_config:
            st.error("⛔ CRITICAL ERROR: 'PARQUET_CONFIG' not found in Secrets.")
            st.stop()
            
        lines = [line.strip() for line in raw_config.strip().split('\n') if line.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and parts[1] in st.secrets:
                config[parts[0]] = parts[1]

    except Exception as e:
        st.error(f"Failed to parse PARQUET_CONFIG: {e}")
        
    if not config:
        st.error("⛔ CRITICAL ERROR: No valid datasets mapped.")
        st.stop()
    return config


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading Dataset...")
def load_parquet_and_clean(key):
    """
    Optimized loader for Standardized Parquet Files.
    Expects columns: DATE, TICKER, CLOSE, etc.
    """
    clean_key = key.strip()
    if clean_key not in st.secrets: return None
        
    url = st.secrets[clean_key]
    
    try:
        buffer = get_gdrive_binary_data(url)
        if not buffer: return None
            
        content = buffer.getvalue()
        
        # 1. Try Parquet (Fastest)
        try:
            df = pd.read_parquet(BytesIO(content))
        except Exception:
            # 2. Fallback to CSV (Engine C for speed)
            try:
                df = pd.read_csv(BytesIO(content), engine='c')
            except Exception:
                return None

        # 3. Standardize Columns (Force Uppercase)
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        # 4. Standard Date Indexing
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df.rename(columns={'DATE': 'ChartDate'}, inplace=True)
            df.sort_values('ChartDate', inplace=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: 'ChartDate'}, inplace=True)
            df.sort_values('ChartDate', inplace=True)
        else:
            return None # Critical failure if no date

        # 5. Ensure Price Column Exists (for legacy app logic)
        if 'CLOSE' in df.columns:
            df['Price'] = df['CLOSE']
            
        return df

    except Exception as e:
        st.error(f"Error processing {clean_key}: {e}")
        return None


@st.cache_data(ttl=CACHE_TTL)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP")
        if not url: return {}

        buffer = get_gdrive_binary_data(url)
        if buffer:
            df = pd.read_csv(buffer, engine='c')
            if len(df.columns) >= 2:
                return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), 
                              df.iloc[:, 1].astype(str).str.strip()))
    except Exception:
        pass
    return {}

@st.cache_data(ttl=CACHE_TTL, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    # Kept largely same as it processes the Options Database (different format)
    try:
        df = pd.read_csv(url, engine='c')
        
        want = {"Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"}
        existing_cols = [c for c in df.columns if c in want]
        df = df[existing_cols]
        
        for c in ["Order Type", "Symbol", "Strike", "Expiry"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        
        if "Dollars" in df.columns and df["Dollars"].dtype == 'object':
             df["Dollars"] = pd.to_numeric(df["Dollars"].str.replace(r'[$,]', '', regex=True), errors="coerce").fillna(0.0)

        if "Contracts" in df.columns and df["Contracts"].dtype == 'object':
             df["Contracts"] = pd.to_numeric(df["Contracts"].str.replace(',', '', regex=False), errors="coerce").fillna(0)
        
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

# --- MATH & TECHNICAL ANALYSIS ---

def add_technicals(df):
    """
    Refactored to prefer PRE-CALCULATED standard columns.
    Only calculates if columns like RSI14/EMA8 are missing.
    """
    if df is None or df.empty: return df
    
    cols = df.columns
    
    # 1. Alias Standard Columns to App Internal Names if needed
    # (The app often looks for 'RSI', 'RSI_14', 'EMA_8' etc)
    
    if 'RSI14' in cols and 'RSI' not in cols:
        df['RSI'] = df['RSI14']
    elif 'RSI_14' in cols and 'RSI' not in cols:
        df['RSI'] = df['RSI_14']

    # 2. Identify missing Technicals
    has_rsi = 'RSI' in df.columns
    has_ema8 = ('EMA8' in cols) or ('EMA_8' in cols)
    has_ema21 = ('EMA21' in cols) or ('EMA_21' in cols)
    has_sma200 = ('SMA200' in cols) or ('SMA_200' in cols)

    if has_rsi and has_ema8 and has_ema21 and has_sma200:
        return df

    # 3. Calculate Missing (Fallback for Yahoo Data)
    close_col = next((c for c in ['CLOSE', 'Close', 'Price'] if c in cols), None)
    if not close_col: return df
    close_series = df[close_col]

    if not has_rsi:
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_14'] = df['RSI']
    
    if not has_ema8: df['EMA_8'] = close_series.ewm(span=8, adjust=False).mean()
    if not has_ema21: df['EMA_21'] = close_series.ewm(span=21, adjust=False).mean()
    if not has_sma200 and len(df) >= 200: df['SMA_200'] = close_series.rolling(window=200).mean()
            
    return df

@st.cache_data(ttl=CACHE_TTL)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        
        df = df.reset_index()
        # Standardize to our Format
        df.columns = [c.upper() for c in df.columns]
        if "DATE" in df.columns:
            if df["DATE"].dt.tz is not None:
                df["DATE"] = df["DATE"].dt.tz_localize(None)
        
        return add_technicals(df)
    except Exception:
        return None

# --- HELPERS ---

def parse_periods(periods_str):
    try:
        return sorted(list(set([int(x.strip()) for x in periods_str.split(',') if x.strip().isdigit()])))
    except:
        return [5, 21, 63, 126]

@st.cache_data(ttl=CACHE_TTL)
def get_expiry_color_map():
    try:
        today = date.today()
        days_ahead = (4 - today.weekday()) % 7
        this_fri = today + timedelta(days=days_ahead)
        return {
            this_fri.strftime("%d %b %y"): "background-color: #b7e1cd; color: black;",
            (this_fri + timedelta(days=7)).strftime("%d %b %y"): "background-color: #fce8b2; color: black;",
            (this_fri + timedelta(days=14)).strftime("%d %b %y"): "background-color: #f4c7c3; color: black;"
        }
    except:
        return {}

def highlight_expiry(val):
    if not isinstance(val, str): return ""
    return get_expiry_color_map().get(val, "")

def clean_strike_fmt(val):
    try:
        f = float(val)
        if f.is_integer(): return str(int(f))
        return str(f)
    except: return str(val)

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid = df["Trade Date"].dropna()
        if not valid.empty: return valid.max().date()
    return date.today() - timedelta(days=1)

@st.cache_data(ttl=CACHE_TTL)
def get_stock_indicators(sym: str):
    try:
        # Fast path via Yahoo if not in bulk
        ticker_obj = yf.Ticker(sym)
        h_full = ticker_obj.history(period="2y", interval="1d")
        
        if len(h_full) == 0: return None, None, None, None, None
        
        # Yahoo returns Title Case, we standardize
        h_full.columns = [c.upper() for c in h_full.columns]
        h_full = add_technicals(h_full)
        
        sma200_col = next((c for c in h_full.columns if c in ['SMA200','SMA_200']), None)
        sma200 = float(h_full[sma200_col].iloc[-1]) if sma200_col else None
        
        h_recent = h_full.iloc[-60:].copy() if len(h_full) > 60 else h_full.copy()
        if len(h_recent) == 0: return None, None, None, None, None
        
        spot_val = float(h_recent["CLOSE"].iloc[-1])
        
        ema8_col = next((c for c in h_recent.columns if c in ['EMA8','EMA_8']), None)
        ema21_col = next((c for c in h_recent.columns if c in ['EMA21','EMA_21']), None)
        
        ema8 = float(h_recent[ema8_col].iloc[-1]) if ema8_col else None
        ema21 = float(h_recent[ema21_col].iloc[-1]) if ema21_col else None
        
        return spot_val, ema8, ema21, sma200, h_full
    except: 
        return None, None, None, None, None


def find_divergences(df_tf, ticker, timeframe, min_n=0, periods_input=None, optimize_for='PF', lookback_period=90, price_source='High/Low', strict_validation=True, recent_days_filter=25, rsi_diff_threshold=2.0):
    # Logic unchanged, assumes df_tf has 'RSI', 'Price', 'Volume', 'VolSMA'
    divergences = []
    n_rows = len(df_tf)
    
    if n_rows < lookback_period + 1: return divergences
    
    rsi_vals = df_tf['RSI'].values
    vol_vals = df_tf['Volume'].values
    vol_sma_vals = df_tf['VolSMA'].values
    close_vals = df_tf['Price'].values 
    
    if price_source == 'Close':
        low_vals = close_vals
        high_vals = close_vals
    else:
        low_vals = df_tf['Low'].values
        high_vals = df_tf['High'].values
        
    def get_date_str(idx, fmt='%Y-%m-%d'): 
        ts = df_tf.index[idx]
        if timeframe.lower() == 'weekly': 
             return df_tf.iloc[idx]['ChartDate'].strftime(fmt)
        return ts.strftime(fmt)
    
    roll_low_min = pd.Series(low_vals).shift(1).rolling(window=lookback_period).min().values
    roll_high_max = pd.Series(high_vals).shift(1).rolling(window=lookback_period).max().values
    
    is_new_low = (low_vals < roll_low_min)
    is_new_high = (high_vals > roll_high_max)
    
    valid_mask = np.zeros(n_rows, dtype=bool)
    valid_mask[lookback_period:] = True
    candidate_indices = np.where(valid_mask & (is_new_low | is_new_high))[0]
    potential_signals = [] 

    for i in candidate_indices:
        p2_rsi = rsi_vals[i]
        p2_vol = vol_vals[i]
        p2_volsma = vol_sma_vals[i]
        lb_start = i - lookback_period
        lb_rsi = rsi_vals[lb_start:i]
        
        is_vol_high = int(p2_vol > (p2_volsma * 1.5)) if not np.isnan(p2_volsma) else 0
        
        if is_new_low[i]:
            p1_idx_rel = np.argmin(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            if p2_rsi > (p1_rsi + rsi_diff_threshold):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                is_valid_structure = True
                if strict_validation and np.any(subset_rsi > 50): is_valid_structure = False
                if is_valid_structure: 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi <= p1_rsi): valid = False
                    if valid:
                        potential_signals.append({"index": i, "type": "Bullish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})
        elif is_new_high[i]:
            p1_idx_rel = np.argmax(lb_rsi)
            p1_rsi = lb_rsi[p1_idx_rel]
            if p2_rsi < (p1_rsi - rsi_diff_threshold):
                idx_p1_abs = lb_start + p1_idx_rel
                subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                is_valid_structure = True
                if strict_validation and np.any(subset_rsi < 50): is_valid_structure = False
                if is_valid_structure: 
                    valid = True
                    if i < n_rows - 1:
                        post_rsi = rsi_vals[i+1:]
                        if np.any(post_rsi >= p1_rsi): valid = False
                    if valid:
                        potential_signals.append({"index": i, "type": "Bearish", "p1_idx": idx_p1_abs, "vol_high": is_vol_high})

    display_threshold_idx = n_rows - recent_days_filter
    bullish_indices = [x['index'] for x in potential_signals if x['type'] == 'Bullish']
    bearish_indices = [x['index'] for x in potential_signals if x['type'] == 'Bearish']

    for sig in potential_signals:
        i = sig["index"]
        s_type = sig["type"]
        idx_p1_abs = sig["p1_idx"]
        price_p1 = low_vals[idx_p1_abs] if s_type=='Bullish' else high_vals[idx_p1_abs]
        price_p2 = low_vals[i] if s_type=='Bullish' else high_vals[i]
        vol_p1 = vol_vals[idx_p1_abs]; vol_p2 = vol_vals[i]
        rsi_p1 = rsi_vals[idx_p1_abs]; rsi_p2 = rsi_vals[i]
        date_p1 = get_date_str(idx_p1_abs, '%Y-%m-%d'); date_p2 = get_date_str(i, '%Y-%m-%d')
        is_recent = (i >= display_threshold_idx)

        div_obj = {
            'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe,
            'Signal_Date_ISO': date_p2, 'P1_Date_ISO': date_p1,
            'RSI1': rsi_p1, 'RSI2': rsi_p2, 'Price1': price_p1, 'Price2': price_p2,
            'Day1_Volume': vol_p1, 'Day2_Volume': vol_p2, 'Is_Recent': is_recent
        }

        tags = []
        latest_row = df_tf.iloc[-1]
        last_price = latest_row['Price']
        
        # Check standard or alias columns
        last_ema8 = latest_row.get('EMA8', latest_row.get('EMA_8'))
        last_ema21 = latest_row.get('EMA21', latest_row.get('EMA_21'))
        
        def is_valid(val): return val is not None and not pd.isna(val)

        if s_type == 'Bullish':
            if is_valid(last_ema8) and last_price >= last_ema8: tags.append(f"EMA{EMA8_PERIOD}")
            if is_valid(last_ema21) and last_price >= last_ema21: tags.append(f"EMA{EMA21_PERIOD}")
        else: 
            if is_valid(last_ema8) and last_price <= last_ema8: tags.append(f"EMA{EMA8_PERIOD}")
            if is_valid(last_ema21) and last_price <= last_ema21: tags.append(f"EMA{EMA21_PERIOD}")
            
        if sig["vol_high"]: tags.append("V_HI")
        if vol_vals[i] > vol_vals[idx_p1_abs]: tags.append("V_GROW")
        
        date_display = f"{get_date_str(idx_p1_abs, '%b %d')} → {get_date_str(i, '%b %d')}"
        rsi_display = f"{int(round(rsi_p1))} {'↗' if rsi_p2 > rsi_p1 else '↘'} {int(round(rsi_p2))}"
        price_display = f"${price_p1:,.2f} ↗ ${price_p2:,.2f}" if price_p2 > price_p1 else f"${price_p1:,.2f} ↘ ${price_p2:,.2f}"

        hist_list = bullish_indices if s_type == 'Bullish' else bearish_indices
        best_stats = calculate_optimal_signal_stats(hist_list, close_vals, i, signal_type=s_type, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if best_stats is None: best_stats = {"Best Period": "—", "Profit Factor": 0.0, "Win Rate": 0.0, "EV": 0.0, "N": 0}
        if best_stats["N"] < min_n: continue

        div_obj.update({
            'Tags': tags, 'Date_Display': date_display,
            'RSI_Display': rsi_display, 'Price_Display': price_display, 
            'Last_Close': f"${latest_row['Price']:,.2f}", 'N': best_stats['N']
        })

        prefix = "Daily" if timeframe == "Daily" else "Weekly"
        if periods_input is not None:
            for p in periods_input:
                future_idx = i + p
                col_price = f"{prefix}_Price_After_{p}"
                col_vol = f"{prefix}_Volume_After_{p}"
                col_ret = f"Ret_{p}"
                if future_idx < n_rows:
                    f_price = close_vals[future_idx]
                    div_obj[col_price] = f_price
                    div_obj[col_vol] = vol_vals[future_idx]
                    entry = close_vals[i]
                    ret_pct = (f_price - entry) / entry if s_type == 'Bullish' else (entry - f_price) / entry 
                    div_obj[col_ret] = ret_pct * 100
                else:
                    div_obj[col_price] = "n/a"; div_obj[col_vol] = "n/a"; div_obj[col_ret] = np.nan
        
        divergences.append(div_obj)
            
    return divergences


def prepare_data(df):
    """
    HEAVILY REFACTORED for Standardized Format.
    Splits dataframe into Daily and Weekly based on explicit column names.
    Avoids re-calculation if standardized columns (RSI14, EMA8) are present.
    """
    # 1. Standardize Header
    df.columns = [col.strip().upper() for col in df.columns]
    
    # 2. Date Indexing
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE').sort_index()
    elif 'CHARTDATE' in df.columns:
        df['CHARTDATE'] = pd.to_datetime(df['CHARTDATE'])
        df = df.set_index('CHARTDATE').sort_index()
        
    # 3. Create DAILY DataFrame
    # Map Standard Cols -> Internal App Names
    daily_map = {
        'CLOSE': 'Price', 'VOLUME': 'Volume', 
        'HIGH': 'High', 'LOW': 'Low', 'OPEN': 'Open',
        'RSI': 'RSI', 'RSI14': 'RSI', 'RSI_14': 'RSI',
        'EMA8': 'EMA8', 'EMA_8': 'EMA8',
        'EMA21': 'EMA21', 'EMA_21': 'EMA21',
        'SMA50': 'SMA50', 'SMA_50': 'SMA50',
        'SMA100': 'SMA100', 'SMA_100': 'SMA100',
        'SMA200': 'SMA200', 'SMA_200': 'SMA200'
    }
    
    # Select available cols
    d_cols = [c for c in daily_map.keys() if c in df.columns]
    df_d = df[d_cols].rename(columns=daily_map).copy()
    
    # Calc VolSMA internally as it's rarely in source
    if 'Volume' in df_d.columns:
        df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        
    # Fallback: If source was missing techs (e.g. Yahoo), calc them
    df_d = add_technicals(df_d)
    
    # Validation
    if 'Price' not in df_d.columns or 'RSI' not in df_d.columns:
        # If still missing essential data, abort
        return None, None
        
    df_d.dropna(subset=['Price', 'RSI'], inplace=True)
    
    # 4. Create WEEKLY DataFrame
    # STRICT MAPPING based on User Requirement
    # Source Columns: W_OPEN, W_HIGH, W_LOW, W_CLOSE, W_VOLUME, W_EMA8, W_EMA21, W_SMA50, W_SMA100, W_SMA200, W_RSI14
    
    weekly_cols_map = {
        'W_CLOSE': 'Price', 
        'W_HIGH': 'High',
        'W_LOW': 'Low',
        'W_OPEN': 'Open',
        'W_VOLUME': 'Volume',
        'W_RSI14': 'RSI',
        'W_EMA8': 'EMA8',
        'W_EMA21': 'EMA21',
        'W_SMA50': 'SMA50',
        'W_SMA100': 'SMA100',
        'W_SMA200': 'SMA200'
    }
    
    available_w_cols = [c for c in weekly_cols_map.keys() if c in df.columns]
    
    if not available_w_cols:
        return df_d, None

    df_w = df[available_w_cols].rename(columns=weekly_cols_map).copy()
    
    # Set ChartDate for weekly (start of week)
    df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
    
    # CRITICAL FIX: Compress to 1 row per week.
    df_w = df_w.groupby('ChartDate').last().sort_index()
    
    # Ensure ChartDate is available as a column for display logic
    df_w['ChartDate'] = df_w.index
    
    if 'Volume' in df_w.columns:
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        
    df_w = add_technicals(df_w)
    
    # Only validate if mapped successfully
    if 'Price' in df_w.columns and 'RSI' in df_w.columns:
        df_w.dropna(subset=['Price', 'RSI'], inplace=True)
    else:
        return df_d, None
        
    return df_d, df_w

def calculate_optimal_signal_stats(history_indices, price_array, current_idx, signal_type='Bullish', timeframe='Daily', periods_input=None, optimize_for='PF'):
    hist_arr = np.array(history_indices)
    valid_indices = hist_arr[hist_arr < current_idx]
    
    if len(valid_indices) == 0: return None

    periods = np.array(periods_input) if periods_input else np.array([5, 21, 63, 126])
    total_len = len(price_array)
    unit = 'w' if timeframe.lower() == 'weekly' else 'd'

    exit_indices_matrix = valid_indices[:, None] + periods[None, :]
    valid_exits_mask = exit_indices_matrix < total_len
    safe_exit_indices = np.clip(exit_indices_matrix, 0, total_len - 1)
    
    entry_prices = price_array[valid_indices]
    exit_prices_matrix = price_array[safe_exit_indices]
    
    raw_returns = (exit_prices_matrix - entry_prices[:, None]) / entry_prices[:, None]
    if signal_type == 'Bearish': raw_returns = -raw_returns

    best_score = -999.0
    best_stats = None
    
    for i, p in enumerate(periods):
        col_mask = valid_exits_mask[:, i]
        p_rets = raw_returns[col_mask, i]
        if len(p_rets) == 0: continue
            
        wins = p_rets[p_rets > 0]
        gross_win = np.sum(wins)
        gross_loss = np.abs(np.sum(p_rets[p_rets < 0]))
        pf = 999.0 if gross_loss == 0 and gross_win > 0 else (gross_win / gross_loss if gross_loss > 0 else 0.0)
        
        n = len(p_rets)
        win_rate = (len(wins) / n) * 100
        avg_ret = np.mean(p_rets) * 100
        std_dev = np.std(p_rets)
        sqn = (np.mean(p_rets) / std_dev) * np.sqrt(n) if std_dev > 0 else 0.0
        
        current_score = pf if optimize_for == 'PF' else sqn
        
        if current_score > best_score:
            best_score = current_score
            best_stats = {
                "Best Period": f"{p}{unit}", "Profit Factor": pf,
                "Win Rate": win_rate, "EV": avg_ret, "N": n, "SQN": sqn
            }
            
    return best_stats

def get_optimal_rsi_duration(history_df, current_rsi, tolerance=2.0):
    if history_df is None or len(history_df) < 100:
        return 30, "Default (No Hist)"

    history_df = add_technicals(history_df)
    
    close_col = 'CLOSE' if 'CLOSE' in history_df.columns else 'Price'
    rsi_col = 'RSI_14' if 'RSI_14' in history_df.columns else 'RSI'
    
    if close_col not in history_df.columns or rsi_col not in history_df.columns:
        return 30, "Default (Missing Cols)"

    close_vals = history_df[close_col].values
    rsi_vals = history_df[rsi_col].values
    
    mask = (rsi_vals >= (current_rsi - tolerance)) & (rsi_vals <= (current_rsi + tolerance))
    match_indices = np.where(mask)[0]
    
    if len(match_indices) < 5:
        return 30, "Default (Low Samples)"
        
    periods = [14, 30, 45, 60]
    best_p = 30; best_score = -999
    total_len = len(close_vals)
    
    for p in periods:
        valid_indices = match_indices[match_indices + p < total_len]
        if len(valid_indices) < 5: continue
        entries = close_vals[valid_indices]
        exits = close_vals[valid_indices + p]
        returns = (exits - entries) / entries
        score = (np.mean(returns > 0) * 2) + np.mean(returns)
        if score > best_score:
            best_score = score
            best_p = p
            
    return best_p, f"RSI Backtest (Optimal {best_p}d)"

def find_whale_confluence(ticker, global_df, current_price, order_type_filter=None):
    if global_df.empty: return None

    f = global_df[global_df["Symbol"].astype(str).str.upper() == ticker].copy()
    if f.empty: return None

    today_dt = pd.Timestamp.now()
    f = f[f["Expiry_DT"] > today_dt]
    
    if order_type_filter:
        f = f[f["Order Type"] == order_type_filter]
    else:
        f = f[f["Order Type"].isin(["Puts Sold", "Calls Bought"])]
        
    if f.empty: return None
    
    f.sort_values(by="Dollars", ascending=False, inplace=True)
    whale = f.iloc[0]
    
    if whale["Order Type"] == "Puts Sold" and whale["Strike (Actual)"] > current_price:
        otm_puts = f[(f["Order Type"]=="Puts Sold") & (f["Strike (Actual)"] < current_price)]
        if not otm_puts.empty: whale = otm_puts.iloc[0]
    
    return {
        "Strike": whale["Strike (Actual)"],
        "Expiry": whale["Expiry_DT"].strftime("%d %b"),
        "Dollars": whale["Dollars"], "Type": whale["Order Type"]
    }

def analyze_trade_setup(ticker, t_df, global_df):
    score = 0; reasons = []
    suggestions = {'Buy Calls': None, 'Sell Puts': None, 'Buy Commons': None}
    
    if t_df is None or t_df.empty: return 0, ["No data"], suggestions

    last = t_df.iloc[-1]
    close = last.get('CLOSE', last.get('Price', 0))
    ema8 = last.get('EMA8', last.get('EMA_8', 0))
    ema21 = last.get('EMA21', last.get('EMA_21', 0))
    sma200 = last.get('SMA200', last.get('SMA_200', 0))
    rsi = last.get('RSI', last.get('RSI_14', 50))
    
    if close > ema8 and close > ema21:
        score += 2; reasons.append("Strong Trend (Price > EMA8 & EMA21)")
    elif close > ema21:
        score += 1; reasons.append("Moderate Trend (Price > EMA21)")
        
    if close > sma200:
        score += 2; reasons.append("Long-term Bullish (> SMA200)")
        
    if 45 < rsi < 65:
        score += 2; reasons.append(f"Healthy Momentum (RSI {rsi:.0f})")
    elif rsi >= 70:
        score -= 1; reasons.append("Overbought (RSI > 70)")
    
    opt_days, opt_reason = 30, "Standard 30d"
    if len(t_df) > 100: opt_days, opt_reason = get_optimal_rsi_duration(t_df, rsi)
    
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
    
    if "RSI Backtest" in opt_reason: reasons.append(f"Hist. Optimal Hold: {opt_days} Days")
    if put_whale: reasons.append(f"Whale: Sold Puts @ ${put_whale['Strike']}")
    if call_whale: reasons.append(f"Whale: Bought Calls @ ${call_whale['Strike']}")
    
    return score, reasons, suggestions

@st.cache_data(ttl=43200) # Cache for 12 hours
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info
        mc = fi.get('marketCap')
        if mc: return float(mc)
        shares = fi.get('shares'); price = fi.get('lastPrice')
        if shares and price: return float(shares * price)
    except: pass
    return 0.0

def fetch_market_caps_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_market_cap, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            results[future_to_ticker[future]] = future.result()
    return results

def fetch_technicals_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_stock_indicators, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            try: results[future_to_ticker[future]] = future.result()
            except: results[future_to_ticker[future]] = (None, None, None, None, None)
    return results

@st.cache_data(ttl=CACHE_TTL, show_spinner="Crunching Smart Money Data...")
def calculate_smart_money_score(df, start_d, end_d, mc_thresh, filter_ema, limit):
    f = df.copy()
    if start_d: f = f[f["Trade Date"].dt.date >= start_d]
    if end_d: f = f[f["Trade Date"].dt.date <= end_d]
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f = f[f[order_type_col].isin(target_types)].copy()
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    f["Signed_Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), f["Dollars"], -f["Dollars"])
    
    smart_stats = f.groupby("Symbol").agg(
        Signed_Dollars=("Signed_Dollars", "sum"),
        Trade_Count=("Symbol", "count"),
        Last_Trade=("Trade Date", "max")
    ).reset_index()
    
    smart_stats.rename(columns={"Signed_Dollars": "Net Sentiment ($)"}, inplace=True)
    unique_tickers = smart_stats["Symbol"].unique().tolist()
    batch_caps = fetch_market_caps_batch(unique_tickers)
    smart_stats["Market Cap"] = smart_stats["Symbol"].map(batch_caps)
    
    valid_data = smart_stats[smart_stats["Market Cap"] >= mc_thresh].copy()
    
    unique_dates = sorted(f["Trade Date"].unique())
    recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
    f_momentum = f[f["Trade Date"].isin(recent_dates)]
    mom_stats = f_momentum.groupby("Symbol")["Signed_Dollars"].sum().reset_index().rename(columns={"Signed_Dollars": "Momentum ($)"})
    
    valid_data = valid_data.merge(mom_stats, on="Symbol", how="left").fillna(0)
    top_bulls = pd.DataFrame(); top_bears = pd.DataFrame()

    if not valid_data.empty:
        valid_data["Impact"] = valid_data["Net Sentiment ($)"] / valid_data["Market Cap"]
        def normalize(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx != mn else 0

        valid_data["Base_Score_Bull"] = (0.35 * normalize(valid_data["Net Sentiment ($)"].clip(lower=0))) + \
                                        (0.30 * normalize(valid_data["Impact"].clip(lower=0))) + \
                                        (0.35 * normalize(valid_data["Momentum ($)"].clip(lower=0)))
        
        valid_data["Base_Score_Bear"] = (0.35 * normalize(-valid_data["Net Sentiment ($)"].clip(upper=0))) + \
                                        (0.30 * normalize(-valid_data["Impact"].clip(upper=0))) + \
                                        (0.35 * normalize(-valid_data["Momentum ($)"].clip(upper=0)))
        
        valid_data["Last Trade"] = valid_data["Last_Trade"].dt.strftime("%d %b")
        
        candidates_bull = valid_data.sort_values(by="Base_Score_Bull", ascending=False).head(limit * 3)
        candidates_bear = valid_data.sort_values(by="Base_Score_Bear", ascending=False).head(limit * 3)
        
        all_tickers = set(candidates_bull["Symbol"]).union(set(candidates_bear["Symbol"]))
        batch_techs = fetch_technicals_batch(list(all_tickers)) if filter_ema else {}

        def apply_ema_filter(df, mode="Bull"):
            if not filter_ema:
                df["Score"] = df[f"Base_Score_{mode}"] * 100
                df["Trend"] = "—"
                return df.head(limit)
            
            def check_row(t):
                s, e8, _, _, _ = batch_techs.get(t, (None, None, None, None, None))
                if not s or not e8: return False, "—"
                if mode == "Bull": return (s > e8), ("✅ >EMA8" if s > e8 else "⚠️ <EMA8")
                return (s < e8), ("✅ <EMA8" if s < e8 else "⚠️ >EMA8")
            
            results = [check_row(t) for t in df["Symbol"]]
            mask = [r[0] for r in results]; trends = [r[1] for r in results]
            
            filtered = df[mask].copy()
            filtered["Trend"] = [t for i, t in enumerate(trends) if mask[i]]
            filtered["Score"] = filtered[f"Base_Score_{mode}"] * 100
            return filtered.head(limit)

        top_bulls = apply_ema_filter(candidates_bull, "Bull")
        top_bears = apply_ema_filter(candidates_bear, "Bear")
        
    return top_bulls, top_bears, valid_data

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_history_optimized(ticker_sym, t_map):
    pq_key = f"{ticker_sym}_PARQUET"
    if pq_key in t_map:
        try:
            url = f"https://drive.google.com/uc?export=download&id={t_map[pq_key]}"
            buffer = get_gdrive_binary_data(url) 
            if buffer:
                # Optimized for Parquet
                df = pd.read_parquet(buffer, engine='pyarrow')
                # Ensure Date is loaded
                if 'DATE' in df.columns: df.rename(columns={'DATE': 'ChartDate'}, inplace=True)
                elif isinstance(df.index, pd.DatetimeIndex): 
                    df.reset_index(inplace=True)
                    df.rename(columns={df.columns[0]: 'ChartDate'}, inplace=True)
                return df
        except Exception: pass 

    if ticker_sym in t_map:
        try:
            df = get_ticker_technicals(ticker_sym, t_map)
            if df is not None and not df.empty: return df
        except Exception: pass

    try: return fetch_yahoo_data(ticker_sym)
    except Exception: return None

def find_rsi_percentile_signals(df, ticker, pct_low=0.10, pct_high=0.90, min_n=1, filter_date=None, timeframe='Daily', periods_input=None, optimize_for='SQN'):
    signals = []
    if len(df) < 200: return signals
    
    cutoff = df.index.max() - timedelta(days=365*10)
    hist_df = df[df.index >= cutoff].copy()
    if hist_df.empty: return signals
    
    rsi_vals = hist_df['RSI'].values 
    price_vals = hist_df['Price'].values
    p10 = np.quantile(rsi_vals, pct_low)
    p90 = np.quantile(rsi_vals, pct_high)
    
    prev_rsi = np.roll(rsi_vals, 1); prev_rsi[0] = rsi_vals[0]
    bull_mask = (prev_rsi < p10) & (rsi_vals >= (p10 + 1.0))
    bear_mask = (prev_rsi > p90) & (rsi_vals <= (p90 - 1.0))
    
    bull_indices = np.where(bull_mask)[0]; bear_indices = np.where(bear_mask)[0]
    all_indices = np.sort(np.concatenate((bull_indices, bear_indices)))
    latest_close = df['Price'].iloc[-1] 
    
    for i in all_indices:
        curr_date = hist_df.index[i].date()
        if filter_date and curr_date < filter_date: continue
            
        is_bullish = i in bull_indices
        s_type = 'Bullish' if is_bullish else 'Bearish'
        
        hist_list = bull_indices if is_bullish else bear_indices
        best_stats = calculate_optimal_signal_stats(hist_list, price_vals, i, signal_type=s_type, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if not best_stats or best_stats["N"] < min_n: continue
            
        ev_val = best_stats['EV']
        sig_close = price_vals[i]
        ev_price = sig_close * (1 + (ev_val / 100.0)) if is_bullish else sig_close * (1 - (ev_val / 100.0))
        thresh = p10 if is_bullish else p90
        curr_rsi = rsi_vals[i]

        signals.append({
            'Ticker': ticker, 'Date': curr_date.strftime('%b %d'), 'Date_Obj': curr_date,
            'Action': "Leaving Low" if is_bullish else "Leaving High",
            'RSI_Display': f"{thresh:.0f} {'↗' if is_bullish else '↘'} {curr_rsi:.0f}",
            'Signal_Price': f"${sig_close:,.2f}", 'Last_Close': f"${latest_close:,.2f}", 
            'Signal_Type': s_type, 'Best Period': best_stats['Best Period'],
            'Profit Factor': best_stats['Profit Factor'], 'Win Rate': best_stats['Win Rate'],
            'EV': best_stats['EV'], 'EV Target': ev_price, 'N': best_stats['N'], 'SQN': best_stats.get('SQN', 0.0)
        })
            
    return signals

@st.cache_data(ttl=CACHE_TTL)
def is_above_ema21(symbol: str) -> bool:
    try:
        ticker = yf.Ticker(symbol)
        h = ticker.history(period="60d")
        if len(h) < 21: return True 
        ema21 = h["Close"].ewm(span=21, adjust=False).mean().iloc[-1]
        return h["Close"].iloc[-1] > ema21
    except: return True

@st.cache_data(ttl=CACHE_TTL)
def get_ticker_technicals(ticker: str, mapping: dict):
    if not mapping or ticker not in mapping: return None
    file_url = f"https://drive.google.com/uc?export=download&id={mapping[ticker]}"
    buffer = get_gdrive_binary_data(file_url)
    
    if buffer:
        try:
            df = pd.read_csv(buffer, engine='c')
            df.columns = [c.strip().upper() for c in df.columns]
            if "DATE" not in df.columns: df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
            return add_technicals(df)
        except Exception: return None
    return None

# --- DATABASE APP HELPERS ---

def initialize_database_state(max_date):
    """Initializes session state variables for the Database app."""
    defaults = {
        'saved_db_ticker': "",
        'saved_db_start': max_date,
        'saved_db_end': max_date,
        'saved_db_exp': (date.today() + timedelta(days=DB_DEFAULT_EXPIRY_OFFSET)),
        'saved_db_inc_cb': True,
        'saved_db_inc_ps': True,
        'saved_db_inc_pb': True
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def _highlight_db_order_type(val):
    """Internal styling function for order types."""
    if val in ["Calls Bought", "Puts Sold"]: 
        return STYLE_BULL_CSS
    elif val == "Puts Bought": 
        return STYLE_BEAR_CSS
    return ''

def get_database_styled_view(df):
    """Prepares the Styler object for the database table."""
    if df.empty: return df
    
    order_type_col = "Order Type" if "Order Type" in df.columns else "Order type"
    display_cols = ["Trade Date", order_type_col, "Symbol", "Strike", "Expiry", "Contracts", "Dollars"]
    
    f_display = df[display_cols].copy()
    
    # Format Dates
    if not pd.api.types.is_string_dtype(f_display["Trade Date"]):
        f_display["Trade Date"] = f_display["Trade Date"].dt.strftime(DB_DATE_FMT)
    
    try:
        f_display["Expiry"] = pd.to_datetime(f_display["Expiry"]).dt.strftime(DB_DATE_FMT)
    except:
        pass 
        
    return f_display.style.format({
        "Dollars": "${:,.0f}", 
        "Contracts": "{:,.0f}"
    }).map(_highlight_db_order_type, subset=[order_type_col])

def filter_database_trades(df, ticker, start_date, end_date, exp_end, inc_cb, inc_ps, inc_pb):
    """Filters the global dataframe based on UI inputs."""
    if df.empty: return pd.DataFrame()

    f = df.copy()
    
    # Date & Ticker Filters
    if ticker: 
        f = f[f["Symbol"].astype(str).str.upper().eq(ticker)]
    if start_date: 
        f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: 
        f = f[f["Trade Date"].dt.date <= end_date]
    if exp_end: 
        f = f[f["Expiry_DT"].dt.date <= exp_end]
    
    # Order Type Logic
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed_types = []
    if inc_cb: allowed_types.append("Calls Bought")
    if inc_ps: allowed_types.append("Puts Sold")
    if inc_pb: allowed_types.append("Puts Bought")
    
    f = f[f[order_type_col].isin(allowed_types)]
    
    # Sort for display
    return f.sort_values(by=["Trade Date", "Symbol"], ascending=[False, True])
    
# --- RANKINGS APP HELPERS ---

def initialize_rankings_state(start_default, max_date):
    """Initializes session state for Rankings app."""
    defaults = {
        'saved_rank_start': start_default,
        'saved_rank_end': max_date,
        'saved_rank_limit': RANK_LIMIT_DEFAULT,
        'saved_rank_mc': "10B",
        'saved_rank_ema': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def generate_top_ideas(top_bulls_df, global_df):
    """Analyzes top bullish candidates using updated constants."""
    if top_bulls_df.empty: return []

    candidates = []
    bull_list = top_bulls_df["Symbol"].tolist()
    batch_results = fetch_technicals_batch(bull_list)
    
    prog_bar = st.progress(0, text="Analyzing technicals...")
    
    for i, t in enumerate(bull_list):
        prog_bar.progress((i + 1) / len(bull_list), text=f"Checking {t}...")
        
        data_tuple = batch_results.get(t)
        t_df = data_tuple[4] if data_tuple else None

        if t_df is None or t_df.empty:
            t_df = fetch_yahoo_data(t)

        if t_df is not None and not t_df.empty:
            sm_row = top_bulls_df[top_bulls_df["Symbol"] == t]
            sm_score = sm_row["Score"].iloc[0] if not sm_row.empty else 0
            
            tech_score, reasons, suggs = analyze_trade_setup(t, t_df, global_df)
            
            # Use Constant for Conviction Divisor
            final_conviction = (sm_score / RANK_CONVICTION_DIVISOR) + tech_score 
            
            price = t_df.iloc[-1].get('CLOSE') or t_df.iloc[-1].get('Close')
            
            candidates.append({
                "Ticker": t,
                "Score": final_conviction,
                "Price": price,
                "Reasons": reasons,
                "Suggestions": suggs
            })
    
    prog_bar.empty()
    
    # Use Constant for Top N
    return sorted(candidates, key=lambda x: x['Score'], reverse=True)[:RANK_TOP_IDEAS_COUNT]

def filter_rankings_data(df, start_date, end_date):
    """Filters data for the rankings period and valid order types."""
    if df.empty: return pd.DataFrame()
    
    f = df.copy()
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    
    # Filter for valid directional trades only
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f = f[f[order_type_col].isin(target_types)]
    
    return f

def calculate_volume_rankings(f_filtered, mc_thresh, filter_ema, limit):
    """
    Calculates the 'Bulltard' style volume rankings.
    Score = (Calls + Puts Sold) - (Puts Bought)
    """
    if f_filtered.empty: return pd.DataFrame(), pd.DataFrame()

    order_type_col = "Order Type" if "Order Type" in f_filtered.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]

    # 1. Pivot Count
    counts = f_filtered.groupby(["Symbol", order_type_col]).size().unstack(fill_value=0)
    for col in target_types:
        if col not in counts.columns: counts[col] = 0
        
    # 2. Calculate Score
    scores_df = pd.DataFrame(index=counts.index)
    scores_df["Score"] = counts["Calls Bought"] + counts["Puts Sold"] - counts["Puts Bought"]
    scores_df["Trade Count"] = counts.sum(axis=1)
    
    # 3. Add Last Trade Date
    last_trade_series = f_filtered.groupby("Symbol")["Trade Date"].max()
    scores_df["Last Trade"] = last_trade_series.dt.strftime(DB_DATE_FMT)
    
    res = scores_df.reset_index()
    
    # 4. Map Market Caps
    unique_ts = res["Symbol"].unique().tolist()
    caps = fetch_market_caps_batch(unique_ts)
    res["Market Cap"] = res["Symbol"].map(caps)
    res = res[res["Market Cap"] >= mc_thresh]

    # 5. Sort Lists
    pre_bull_df = res.sort_values(by=["Score", "Trade Count"], ascending=[False, False])
    pre_bear_df = res.sort_values(by=["Score", "Trade Count"], ascending=[True, False])

    # 6. Apply EMA Filter (Helper)
    def _apply_ema_limit(source_df, mode="Bull"):
        if not filter_ema:
            return source_df.head(limit)
        
        # Fetch extra candidates to account for filtering attrition
        candidates = source_df.head(limit * 3)
        final_list = []
        needed_tickers = candidates["Symbol"].tolist()
        
        # Small batch fetch for just these candidates
        mini_batch = fetch_technicals_batch(needed_tickers)
        
        for _, r in candidates.iterrows():
            try:
                # Tuple: (spot, ema8, ema21, sma200, history)
                t_data = mini_batch.get(r["Symbol"], (None, None, None, None, None))
                spot, ema8 = t_data[0], t_data[1]
                
                if spot and ema8:
                    if mode == "Bull" and spot > ema8: final_list.append(r)
                    elif mode == "Bear" and spot < ema8: final_list.append(r)
            except: 
                pass
            
            if len(final_list) >= limit: break
        
        return pd.DataFrame(final_list)

    bull_df = _apply_ema_limit(pre_bull_df, "Bull")
    bear_df = _apply_ema_limit(pre_bear_df, "Bear")
    
    return bull_df, bear_df

# --- PIVOT APP HELPERS ---

def initialize_pivot_state(start_default, max_date):
    """Initializes session state for Pivot app."""
    defaults = {
        'saved_pv_start': max_date,
        'saved_pv_end': max_date,
        'saved_pv_ticker': "",
        'saved_pv_notional': "0M",
        'saved_pv_mkt_cap': "0B",
        'saved_pv_ema': "All",
        'saved_calc_strike': 100.0,
        'saved_calc_premium': 2.50,
        'saved_calc_expiry': date.today() + timedelta(days=30)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def generate_pivot_pools(d_range):
    """
    Splits data into pools and identifies Risk Reversals (Matched Calls/Puts).
    Replicates original logic: Matches 'Calls Bought' with 'Puts Sold' by Symbol/Date/Expiry.
    """
    order_type_col = "Order Type" if "Order Type" in d_range.columns else "Order type"
    
    cb_pool = d_range[d_range[order_type_col] == "Calls Bought"].copy()
    ps_pool = d_range[d_range[order_type_col] == "Puts Sold"].copy()
    pb_pool = d_range[d_range[order_type_col] == "Puts Bought"].copy()
    
    # Risk Reversal Matching Logic
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    
    # Use groupby cumcount to handle multiple identical trades on same day
    cb_pool['occ'] = cb_pool.groupby(keys).cumcount()
    ps_pool['occ'] = ps_pool.groupby(keys).cumcount()
    
    rr_matches = pd.merge(cb_pool, ps_pool, on=keys + ['occ'], suffixes=('_c', '_p'))
    
    if not rr_matches.empty:
        # Construct RR DataFrame
        rr_c = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_c', 'Strike_c']].copy()
        rr_c.rename(columns={'Dollars_c': 'Dollars', 'Strike_c': 'Strike'}, inplace=True)
        rr_c['Pair_ID'] = rr_matches.index
        rr_c['Pair_Side'] = 0 # Call Side
        
        rr_p = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_p', 'Strike_p']].copy()
        rr_p.rename(columns={'Dollars_p': 'Dollars', 'Strike_p': 'Strike'}, inplace=True)
        rr_p['Pair_ID'] = rr_matches.index
        rr_p['Pair_Side'] = 1 # Put Side
        
        df_rr = pd.concat([rr_c, rr_p])
        df_rr['Strike'] = df_rr['Strike'].apply(clean_strike_fmt)
        
        # Filter matched rows out of the original pools
        match_keys = keys + ['occ']
        def _filter_out(pool, matches):
            temp = matches[match_keys].copy()
            temp['_remove'] = True
            merged = pool.merge(temp, on=match_keys, how='left')
            return merged[merged['_remove'].isna()].drop(columns=['_remove'])
            
        cb_pool = _filter_out(cb_pool, rr_matches)
        ps_pool = _filter_out(ps_pool, rr_matches)
    else:
        df_rr = pd.DataFrame(columns=['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars', 'Strike', 'Pair_ID', 'Pair_Side'])
        
    return cb_pool, ps_pool, pb_pool, df_rr

def filter_pivot_dataframe(data, ticker_filter, min_notional, min_mkt_cap, ema_filter):
    """Applies the Filters (Ticker, Notional, Market Cap, EMA) to a dataframe."""
    if data.empty: return data
    f = data.copy()
    
    if ticker_filter: 
        f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
    
    f = f[f["Dollars"] >= min_notional]
    
    # Advanced Filters (Market Cap & EMA)
    if not f.empty and (min_mkt_cap > 0 or ema_filter == "Yes"):
        unique_symbols = f["Symbol"].unique()
        valid_symbols = set(unique_symbols)
        
        if min_mkt_cap > 0:
            valid_symbols = {s for s in valid_symbols if get_market_cap(s) >= float(min_mkt_cap)}
        
        if ema_filter == "Yes":
            # Check EMA > SMA using batch fetch
            batch_results = fetch_technicals_batch(list(valid_symbols))
            # Keep symbol if data missing OR if Price > EMA21 (Tuple index 0 is Price, index 2 is EMA21)
            valid_symbols = {
                s for s in valid_symbols 
                if batch_results.get(s, (None, None))[2] is None or 
                (batch_results[s][0] is not None and batch_results[s][2] is not None and batch_results[s][0] > batch_results[s][2])
            }
        
        f = f[f["Symbol"].isin(valid_symbols)]
        
    return f

def get_pivot_styled_view(data, is_rr=False):
    """Aggregates data by Symbol/Strike/Expiry for display."""
    if data.empty: 
        return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
    
    sr = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
    
    if is_rr: 
        piv = data.merge(sr, on="Symbol").sort_values(by=["Total_Sym_Dollars", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
    else:
        piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().merge(sr, on="Symbol")
        piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
    
    piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
    
    # Hide repeated symbols for cleaner look
    piv["Symbol_Display"] = np.where(piv["Symbol"] == piv["Symbol"].shift(1), "", piv["Symbol"])
    
    return piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=None),
    "Strike": st.column_config.TextColumn("Strike", width=None),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=None),
    "Contracts": st.column_config.NumberColumn("Qty", width=None),
    "Dollars": st.column_config.NumberColumn("Dollars", width=None),
}

# --- STRIKE ZONES APP HELPERS ---

def initialize_strike_zone_state(exp_default):
    """Initializes session state for Strike Zones app."""
    defaults = {
        'saved_sz_ticker': "AMZN",
        'saved_sz_start': None,
        'saved_sz_end': None,
        'saved_sz_exp': exp_default,
        'saved_sz_view': "Price Zones",
        'saved_sz_width_mode': "Auto",
        'saved_sz_fixed': SZ_DEFAULT_FIXED_SIZE,
        'saved_sz_inc_cb': True,
        'saved_sz_inc_ps': True,
        'saved_sz_inc_pb': True
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def filter_strike_zone_data(df, ticker, start_date, end_date, exp_end, inc_cb, inc_ps, inc_pb):
    """Filters the dataframe for the Strike Zones editor."""
    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if f.empty: return f

    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    
    # Expiry Filter (Today to End Range)
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    
    # Order Types
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed = []
    if inc_cb: allowed.append("Calls Bought")
    if inc_ps: allowed.append("Puts Sold")
    if inc_pb: allowed.append("Puts Bought")
    
    f = f[f[order_type_col].isin(allowed)].copy()
    
    # Initialize Include col for editor
    if not f.empty and "Include" not in f.columns:
        f.insert(0, "Include", True)
        
    return f

def get_strike_zone_technicals(ticker):
    """Retrieves technicals, falling back to Yahoo calculation if needed."""
    spot, ema8, ema21, sma200, _ = get_stock_indicators(ticker)

    if spot is None:
        df_y = fetch_yahoo_data(ticker)
        if df_y is not None and not df_y.empty:
            try:
                spot = float(df_y["CLOSE"].iloc[-1])
                ema8 = float(df_y["CLOSE"].ewm(span=EMA8_PERIOD, adjust=False).mean().iloc[-1])
                ema21 = float(df_y["CLOSE"].ewm(span=EMA21_PERIOD, adjust=False).mean().iloc[-1])
                sma200 = float(df_y["CLOSE"].rolling(window=200).mean().iloc[-1]) if len(df_y) >= 200 else None
            except: 
                pass

    if spot is None: spot = 100.0
    return spot, ema8, ema21, sma200

def generate_price_zones_html(df, spot, width_mode, fixed_size, hide_empty=True):
    """Calculates price zones and returns the HTML string for the chart."""
    f = df.copy()
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    
    # Calculate Signed Dollars
    f["Signed Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0.0)
    
    # Determine Zone Width
    strike_vals = f["Strike (Actual)"].values
    strike_min, strike_max = float(np.nanmin(strike_vals)), float(np.nanmax(strike_vals))
    
    if width_mode == "Auto": 
        denom = SZ_AUTO_WIDTH_DENOM
        raw_w = max(1e-9, strike_max - strike_min) / denom
        # Find nearest step >= raw_w from constants
        zone_w = float(next((s for s in SZ_AUTO_STEPS if s >= raw_w), 100))
    else: 
        zone_w = float(fixed_size)
    
    # Calculate Buckets relative to Spot
    n_dn = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w))
    n_up = int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
    
    lower_edge = spot - n_dn * zone_w
    total = max(1, n_dn + n_up)
    
    f["ZoneIdx"] = np.clip(
        np.floor((f["Strike (Actual)"] - lower_edge) / zone_w).astype(int), 
        0, 
        total - 1
    )

    # Aggregation
    agg = f.groupby("ZoneIdx").agg(
        Net_Dollars=("Signed Dollars","sum"), 
        Trades=("Signed Dollars","count")
    ).reset_index()
    
    zone_df = pd.DataFrame([(z, lower_edge + z*zone_w, lower_edge + (z+1)*zone_w) for z in range(total)], columns=["ZoneIdx","Zone_Low","Zone_High"])
    zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
    
    if hide_empty: 
        zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
    
    # Build HTML
    html_out = ['<div class="zones-panel">']
    max_val = max(1.0, zs["Net_Dollars"].abs().max())
    sorted_zs = zs.sort_values("ZoneIdx", ascending=False)
    
    upper_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) > spot]
    lower_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) <= spot]
    
    def _fmt_neg(x): return f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"
    
    def _add_rows(rows):
        for _, r in rows.iterrows():
            color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
            pct = (abs(r['Net_Dollars']) / max_val) * 100
            val_str = _fmt_neg(r["Net_Dollars"])
            html_out.append(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')

    _add_rows(upper_zones)
    html_out.append(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>')
    _add_rows(lower_zones)
    
    html_out.append('</div>')
    return "".join(html_out)

def generate_expiry_buckets_html(df):
    """Calculates time buckets and returns the HTML string."""
    f = df.copy()
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    
    # Calculate Signed Dollars
    f["Signed Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0.0)
    
    days_diff = (pd.to_datetime(f["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days)
    
    f["Bucket"] = pd.cut(days_diff, bins=SZ_BUCKET_BINS, labels=SZ_BUCKET_LABELS, include_lowest=True)
    
    agg = f.groupby("Bucket").agg(
        Net_Dollars=("Signed Dollars","sum"), 
        Trades=("Signed Dollars","count")
    ).reset_index()
    
    max_val = max(1.0, agg["Net_Dollars"].abs().max())
    html_out = []
    
    def _fmt_neg(x): return f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

    for _, r in agg.iterrows():
        color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
        pct = (abs(r['Net_Dollars']) / max_val) * 100
        val_str = _fmt_neg(r["Net_Dollars"])
        html_out.append(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
    
    return "".join(html_out)



