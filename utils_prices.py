# utils_prices.py
# This file contains constants, technical analysis functions, and scanners for the Price Divergences, RSI Scanner, Seasonality, and EMA Distance apps.

# utils_prices.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
# Import core data functions from your main utils
# Adjust this import if you move these specific math functions later
from utils_darcy import fetch_technicals_batch, get_market_cap

# --- CONSTANTS: DIVERGENCES ---
DIV_CSV_PERIODS_DAYS = [5, 21, 63, 126, 252]
DIV_CSV_PERIODS_WEEKS = [4, 13, 26, 52]

# --- STATE INITIALIZATION HELPERS ---

def initialize_divergence_state():
    """Initializes session state for Price Divergences app."""
    defaults = {
        'saved_rsi_div_lookback': 90,
        'saved_rsi_div_source': "High/Low",
        'saved_rsi_div_strict': "Yes",
        'saved_rsi_div_days_since': 25,
        'saved_rsi_div_diff': 2.0,
        'rsi_hist_ticker': "AMZN",
        'rsi_hist_results': None,
        'rsi_hist_last_run_params': {},
        'rsi_hist_bulk_df': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def initialize_rsi_scanner_state():
    """Initializes session state for RSI Scanner app."""
    defaults = {
        'saved_rsi_pct_low': 10,
        'saved_rsi_pct_high': 90,
        'saved_rsi_pct_min_n': 1,
        'saved_rsi_pct_periods': "5, 21, 63, 126, 252"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def initialize_seasonality_state():
    """Initializes session state for Seasonality app."""
    defaults = {
        'seas_single_df': None,
        'seas_single_last_ticker': "",
        'seas_scan_results': None,
        'seas_scan_csvs': None,
        'seas_scan_active': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# --- HELPER FUNCTIONS ---

def inject_volume(results_list, data_df):
    """Looks up Volume for P1 and Signal dates and adds to results."""
    if not results_list or data_df is None or data_df.empty:
        return results_list
    
    # Identify Volume Column
    vol_col = next((c for c in data_df.columns if c.strip().upper() == 'VOLUME'), None)
    if not vol_col: return results_list

    # Identify Date Index/Column for Lookup
    lookup = {}
    try:
        temp_df = data_df.copy()
        date_col = next((c for c in temp_df.columns if 'DATE' in c.upper()), None)
        
        if date_col:
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            temp_df['__date_str'] = temp_df[date_col].dt.strftime('%Y-%m-%d')
            lookup = dict(zip(temp_df['__date_str'], temp_df[vol_col]))
        elif isinstance(temp_df.index, pd.DatetimeIndex):
            temp_df['__date_str'] = temp_df.index.strftime('%Y-%m-%d')
            lookup = dict(zip(temp_df['__date_str'], temp_df[vol_col]))
    except:
        return results_list
    
    # Inject
    for row in results_list:
        d1 = row.get('P1_Date_ISO')
        d2 = row.get('Signal_Date_ISO')
        row['Vol1'] = lookup.get(d1, np.nan)
        row['Vol2'] = lookup.get(d2, np.nan)
    
    return results_list

def process_export_columns(df_in):
    """Renames Ret_XX columns to D_Ret_XX or W_Ret_XX based on timeframe for CSV export."""
    if df_in.empty: return df_in
    out = df_in.copy()
    
    if 'Type' in out.columns:
        out['Divergence Type'] = out['Type']

    cols = out.columns
    ret_cols = [c for c in cols if c.startswith('Ret_')]
    
    for rc in ret_cols:
        d_col_name = f"D_{rc}"
        w_col_name = f"W_{rc}"
        out[d_col_name] = out.apply(lambda x: x[rc] if x.get('Timeframe') == 'Daily' else None, axis=1)
        out[w_col_name] = out.apply(lambda x: x[rc] if x.get('Timeframe') == 'Weekly' else None, axis=1)
    
    out = out.drop(columns=ret_cols)
    
    first_cols = ['Ticker', 'Divergence Type', 'Timeframe', 'Signal_Date_ISO', 'P1_Date_ISO', 'Price1', 'Price2', 'RSI1', 'RSI2', 'Vol1', 'Vol2']
    existing_first = [c for c in first_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in existing_first]
    
    return out[existing_first + other_cols]

def fmt_finance(val):
    """Formats a number as a finance percentage (e.g., (1.5%) for negatives)."""
    if pd.isna(val): return ""
    if isinstance(val, str): return val
    if val < 0: return f"({abs(val):.1f}%)"
    return f"{val:.1f}%"

def run_ema_backtest(signal_series, price_data, low_data, lookforward=30, drawdown_thresh=-0.08):
    """
    Backtests a specific signal series against price data.
    Used by EMA Distance App.
    """
    idxs = signal_series[signal_series].index
    if len(idxs) == 0: return 0, 0, 0
    hits = 0
    days_to_dd = []
    closes = price_data.values
    lows = low_data.values
    is_signal = signal_series.values
    n = len(closes)
    
    for i in range(n):
        if not is_signal[i]: continue
        if i + lookforward >= n: continue 
        
        entry_price = closes[i]
        future_window = lows[i+1 : i+1+lookforward]
        min_future = np.min(future_window)
        dd = (min_future - entry_price) / entry_price
        
        if dd <= drawdown_thresh:
            hits += 1
            target_price = entry_price * (1 + drawdown_thresh)
            hit_indices = np.where(future_window <= target_price)[0]
            if len(hit_indices) > 0:
                days_to_dd.append(hit_indices[0] + 1)
                
    hit_rate = (hits / len(idxs)) * 100 if len(idxs) > 0 else 0
    median_days = np.median(days_to_dd) if days_to_dd else 0
    return len(idxs), hit_rate, median_days