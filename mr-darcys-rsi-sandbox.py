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
/* Custom Note Box */
.info-box { background-color: #e8f4f8; border-left: 5px solid #2e86c1; padding: 15px; margin-bottom: 20px; border-radius: 4px; }
.info-title { font-weight: bold; color: #1a5276; font-size: 1.1em; margin-bottom: 5px; }
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
DIVERGENCE_LOOKBACK = 90      # Look for lows/highs in this window
SIGNAL_LOOKBACK_PERIOD = 25   # Show signals if they happened in last 25 days
RSI_DIFF_THRESHOLD = 2
MIN_N_THRESHOLD = 5

# --- 3. HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def load_and_clean_data(sheet_url):
    try:
        csv_url = sheet_url.replace('/edit?usp=sharing', '/export?format=csv')
        df = pd.read_csv(csv_url)
        # Normalize column names
        df.columns = df.columns.str.strip()
        
        # Try to find Date column
        date_col = None
        for c in df.columns:
            if 'date' in c.lower() or 'time' in c.lower():
                date_col = c
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.rename(columns={date_col: 'Trade Date'}, inplace=True)
            df = df.sort_values('Trade Date', ascending=False)
        
        # Convert numeric columns
        cols_to_num = ['Qty', 'Strike', 'Price', 'Mark', 'Delta', 'Gamma', 'Vega', 'Theta', 'IV', 'Vol', 'OI']
        for c in cols_to_num:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
        
        # Calculate DTE if Exp exists
        if 'Exp' in df.columns:
            df['Exp'] = pd.to_datetime(df['Exp'], errors='coerce')
            df['DTE'] = (df['Exp'] - pd.to_datetime(date.today())).dt.days
            
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

def get_market_cap(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info.get('marketCap', 0)
    except:
        return 0

@st.cache_data(ttl=86400)
def fetch_market_caps_batch(tickers):
    # This is a placeholder. In a real scenario you'd batch this or use a lightweight API.
    # For now we skip or do one by one if list is small.
    return {}

def get_stock_indicators(ticker):
    # Placeholder for single ticker fetch if needed
    return None

@st.cache_data(ttl=3600)
def fetch_technicals_batch(tickers):
    # Placeholder
    return pd.DataFrame()

def get_table_height(df, max_rows=15):
    return min((len(df) + 1) * 35 + 3, (max_rows + 1) * 35 + 3)

def get_expiry_color_map(dte_series):
    # Map DTE to colors
    return {}

def highlight_expiry(val):
    if not isinstance(val, (int, float)): return ''
    if val <= 0: return 'background-color: #ffcccc; color: #550000; font-weight: bold'
    elif val <= 7: return 'background-color: #ffeebb; color: #664400'
    elif val <= 30: return 'background-color: #eebbff; color: #440055'
    return ''

def clean_strike_fmt(val):
    return f"{val:.1f}"

def get_max_trade_date(df):
    if 'Trade Date' in df.columns and not df.empty:
        return df['Trade Date'].max()
    return pd.Timestamp.now()

def get_confirmed_gdrive_data():
    # Placeholder for drive logic
    return None

def load_dataset_config():
    return {}

def load_ticker_map():
    return {}

def get_ticker_technicals(ticker):
    try:
        # Avoid rate limits
        time.sleep(0.1)
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        return df.iloc[-1]
    except:
        return None

def fetch_and_prepare_ai_context(df_global):
    return "Context Placeholder"

def calculate_ev_data_numpy(rsi_hist, price_hist, current_rsi, lookahead, current_price):
    # Simplified version for speed
    return 0.0

def get_optimal_rsi_duration(df):
    return 14

def find_whale_confluence(df):
    return pd.DataFrame()

def analyze_trade_setup(ticker):
    return "Analysis Placeholder"

# --- CORE LOGIC UPDATE: BACKTESTER & DIVERGENCE ---

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
    # Scans using New Low vs Old RSI logic (Bulltard Database Method)
    rsi_vals = df_tf['RSI'].values
    low_vals = df_tf['Low'].values
    high_vals = df_tf['High'].values
    n_rows = len(df_tf)
    divergence_indices = []
    
    if n_rows < DIVERGENCE_LOOKBACK + 1: return np.array([])
        
    for i in range(DIVERGENCE_LOOKBACK, n_rows):
        p2_rsi = rsi_vals[i]
        lb_start = i - DIVERGENCE_LOOKBACK
        lb_rsi = rsi_vals[lb_start:i]
        trigger = False
        
        if s_type == 'Bullish':
            p2_low = low_vals[i]
            lb_low = low_vals[lb_start:i]
            # 1. Price New Low (Buying the drop)
            if p2_low < np.min(lb_low):
                # 2. RSI Higher Low
                p1_idx_rel = np.argmin(lb_rsi)
                p1_rsi = lb_rsi[p1_idx_rel]
                if p2_rsi > (p1_rsi + RSI_DIFF_THRESHOLD):
                    idx_p1_abs = lb_start + p1_idx_rel
                    # 3. No crossing 50 (The Reset Rule)
                    subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                    if not np.any(subset_rsi > 50): trigger = True
        else: 
            p2_high = high_vals[i]
            lb_high = high_vals[lb_start:i]
            # 1. Price New High
            if p2_high > np.max(lb_high):
                # 2. RSI Lower High
                p1_idx_rel = np.argmax(lb_rsi)
                p1_rsi = lb_rsi[p1_idx_rel]
                if p2_rsi < (p1_rsi - RSI_DIFF_THRESHOLD):
                    idx_p1_abs = lb_start + p1_idx_rel
                    # 3. No crossing 50
                    subset_rsi = rsi_vals[idx_p1_abs : i + 1]
                    if not np.any(subset_rsi < 50): trigger = True

        if trigger: divergence_indices.append(i)
    return np.array(divergence_indices)


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
        df_w.rename(columns={w_close: 'Price', w_vol: 'Volume', w_high: 'High', w_low: 'Low', w_rsi: 'RSI', w_ema8: 'EMA8', w_ema21: 'EMA21', w_high: 'High', w_low: 'Low'}, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: df_w = None
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    # This might be used by AI or legacy calls, but main scanner uses find_historical_divergences
    return []

def get_date_str(idx, df, timeframe):
    ts = df.index[idx]
    if timeframe.lower() == 'weekly': 
         return df.iloc[idx]['ChartDate'].strftime('%Y-%m-%d')
    return ts.strftime('%Y-%m-%d')

def find_rsi_percentile_signals(df_d, df_w):
    return "Signals Placeholder"

def fetch_yahoo_data(ticker_list):
    return None

# --- APP PAGES ---

def run_database_app(df):
    st.title("📂 Options Database")
    
    # --- Sidebar Filtering ---
    st.sidebar.header("Filter Database")
    
    # Ticker Filter
    all_tickers = sorted(df['Symbol'].unique().tolist()) if 'Symbol' in df.columns else []
    if not all_tickers and 'Ticker' in df.columns:
         all_tickers = sorted(df['Ticker'].unique().tolist())
         
    selected_tickers = st.sidebar.multiselect("Select Ticker(s)", all_tickers)
    
    # Date Filter
    if 'Trade Date' in df.columns:
        min_date = df['Trade Date'].min().date() if not df.empty else date.today()
        max_date = df['Trade Date'].max().date() if not df.empty else date.today()
        date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    else:
        date_range = []
    
    # Apply Filters
    df_filtered = df.copy()
    
    ticker_col = 'Symbol' if 'Symbol' in df.columns else 'Ticker'
    if selected_tickers and ticker_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[ticker_col].isin(selected_tickers)]
    
    if len(date_range) == 2 and 'Trade Date' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['Trade Date'].dt.date >= date_range[0]) & 
            (df_filtered['Trade Date'].dt.date <= date_range[1])
        ]
        
    st.dataframe(df_filtered, use_container_width=True, height=800)

def save_db_state(): pass
def highlight_db_order_type(val): return ''

def run_rankings_app(df):
    st.title("🏆 Rankings")
    
    if df.empty:
        st.warning("No data available.")
        return

    tab1, tab2, tab3 = st.tabs(["Most Active Tickers", "Largest Trades", "Unusual Volume"])
    
    with tab1:
        st.subheader("Top Tickers by Premium")
        if 'Symbol' in df.columns and 'Dollars' in df.columns:
            top_tickers = df.groupby('Symbol')['Dollars'].sum().sort_values(ascending=False).head(20)
            st.bar_chart(top_tickers)
        else:
            st.info("Required columns (Symbol, Dollars) not found.")
            
    with tab2:
        st.subheader("Top 20 Largest Individual Trades")
        if 'Dollars' in df.columns:
            cols = ['Trade Date', 'Symbol', 'Qty', 'Dollars', 'Exp', 'Strike']
            avail_cols = [c for c in cols if c in df.columns]
            largest = df.nlargest(20, 'Dollars')[avail_cols]
            st.dataframe(largest, use_container_width=True)
            
    with tab3:
        st.subheader("High Volume Strikes")
        if 'Symbol' in df.columns and 'Strike' in df.columns and 'Qty' in df.columns:
            # Group by Symbol + Strike
            df['StrikeStr'] = df['Strike'].astype(str)
            df['Key'] = df['Symbol'] + " $" + df['StrikeStr']
            
            vol_group = df.groupby('Key')['Qty'].sum().sort_values(ascending=False).head(20)
            st.bar_chart(vol_group)

def save_rank_state(): pass
def calculate_smart_money_score(row): return 0
def get_filtered_list(df): return df

def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    if df.empty: return
    
    c1, c2, c3 = st.columns(3)
    index_col = c1.selectbox("Rows (Index)", df.columns, index=0)
    columns_col = c2.selectbox("Columns", df.columns, index=1 if len(df.columns)>1 else 0)
    values_col = c3.selectbox("Values", df.columns, index=2 if len(df.columns)>2 else 0)
    agg_func = c3.selectbox("Aggregation", ['sum', 'mean', 'count', 'max'])
    
    try:
        pivot_df = pd.pivot_table(df, index=index_col, columns=columns_col, values=values_col, aggfunc=agg_func, fill_value=0)
        st.dataframe(pivot_df, use_container_width=True)
    except Exception as e:
        st.error(f"Pivot Error: {e}")

def save_pv_state(): pass
def apply_f(df): return df
def filter_out_matches(df): return df
def get_p(df): return df

def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    if df.empty: return
    
    ticker = st.text_input("Enter Ticker", "SPY").upper()
    if ticker and 'Symbol' in df.columns:
        df_t = df[df['Symbol'] == ticker]
        if not df_t.empty and 'Strike' in df_t.columns and 'Qty' in df_t.columns:
             strike_vol = df_t.groupby('Strike')['Qty'].sum().reset_index()
             chart = alt.Chart(strike_vol).mark_bar().encode(
                x=alt.X('Strike:Q', bin=False),
                y='Qty:Q'
            ).interactive()
             st.altair_chart(chart, use_container_width=True)

def save_sz_state(): pass
def pct_from_spot(val): return 0

# --- UPDATED SCANNER APP ---
def run_rsi_scanner_app(df_global):
    st.title("📈 RSI Scanner")
    
    # --- UPDATED PAGE NOTES WITH INVALIDATION LOGIC ---
    st.markdown("""
    <div class="info-box">
        <div class="info-title">ℹ️ HOW IT WORKS (Momentum Exhaustion)</div>
        <p>This scanner captures "Falling Knives" where selling pressure is high, but momentum is fading.</p>
        
        <p><b>✅ Signal Logic (Bullish):</b></p>
        <ul>
            <li><b>Price Action:</b> Price hits a <b>NEW 90-Day Low</b> (we are buying the drop).</li>
            <li><b>Divergence:</b> RSI is <b>HIGHER</b> than it was at the previous bottom.</li>
            <li><b>Recency:</b> Only shows signals detected in the <b>Last 25 Days</b>.</li>
        </ul>
        
        <p><b>⛔ Invalidation / Reset Rules:</b></p>
        <ul>
            <li><b>The "50" Reset:</b> If RSI crossed <b>above 50</b> between the previous low and now, the setup is <b>VOID</b> (momentum reset).</li>
            <li><b>Hard Stop:</b> If RSI drops <b>below</b> the previous RSI low, the divergence is broken. Exit immediately.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Input for Tickers
    default_tickers = "SPY, QQQ, IWM, NVDA, TSLA, AMD, AAPL, MSFT, GOOGL, AMZN, META, NFLX, RBLX"
    ticker_input = st.text_area("Enter Tickers (comma separated)", default_tickers)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    
    if st.button("Run Scanner"):
        results = []
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            df_ticker = get_ticker_data(ticker)
            if df_ticker is None: continue
            
            df_d, _ = prepare_data(df_ticker)
            if df_d is None or len(df_d) < 100: continue
            
            full_price = df_d['Price'].values
            
            # --- 1. Find ALL Historical Divergences (New Logic) ---
            hist_bull_indices = find_historical_divergences(df_d, 'Bullish')
            hist_bear_indices = find_historical_divergences(df_d, 'Bearish')
            
            # --- 2. Calculate Stats ---
            stats_bull = backtest_signal_performance(hist_bull_indices, full_price)
            stats_bear = backtest_signal_performance(hist_bear_indices, full_price)
            
            # --- 3. Check for RECENT Signals ---
            current_idx = len(df_d) - 1
            min_idx_to_show = len(df_d) - SIGNAL_LOOKBACK_PERIOD
            
            # Check Bullish
            active_bull = False
            bull_date = ""
            if len(hist_bull_indices) > 0:
                last_sig = hist_bull_indices[-1]
                if last_sig >= min_idx_to_show:
                    active_bull = True
                    bull_date = df_d.index[last_sig].strftime('%Y-%m-%d')

            # Check Bearish
            active_bear = False
            bear_date = ""
            if len(hist_bear_indices) > 0:
                last_sig = hist_bear_indices[-1]
                if last_sig >= min_idx_to_show:
                    active_bear = True
                    bear_date = df_d.index[last_sig].strftime('%Y-%m-%d')
            
            # --- 4. Add to Results ---
            if active_bull:
                row = {
                    "Ticker": ticker,
                    "Type": "Bullish",
                    "Signal Date": bull_date,
                    "Price": df_d['Price'].iloc[-1],
                    "RSI": f"{df_d['RSI'].iloc[-1]:.1f}",
                    "Best Period": stats_bull['Best Period'] if stats_bull else "N/A",
                    "Win Rate": f"{stats_bull['Win Rate']:.1f}%" if stats_bull else "N/A",
                    "PF": f"{stats_bull['Profit Factor']:.2f}" if stats_bull else "N/A",
                    "EV%": f"{stats_bull['EV']:.2f}%" if stats_bull else "N/A",
                    "N": stats_bull['N'] if stats_bull else 0
                }
                results.append(row)
                
            if active_bear:
                row = {
                    "Ticker": ticker,
                    "Type": "Bearish",
                    "Signal Date": bear_date,
                    "Price": df_d['Price'].iloc[-1],
                    "RSI": f"{df_d['RSI'].iloc[-1]:.1f}",
                    "Best Period": stats_bear['Best Period'] if stats_bear else "N/A",
                    "Win Rate": f"{stats_bear['Win Rate']:.1f}%" if stats_bear else "N/A",
                    "PF": f"{stats_bear['Profit Factor']:.2f}" if stats_bear else "N/A",
                    "EV%": f"{stats_bear['EV']:.2f}%" if stats_bear else "N/A",
                    "N": stats_bear['N'] if stats_bear else 0
                }
                results.append(row)
            
            progress_bar.progress((i + 1) / len(tickers))
            
        if results:
            st.success(f"Found {len(results)} active setups.")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True, column_config={
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Win Rate": st.column_config.ProgressColumn("Win Rate", format="%s", min_value=0, max_value=100),
            })
        else:
            st.warning("No divergences found in the last 25 days for these tickers.")


def highlight_best(s): return [''] * len(s)
def highlight_ret(s): return [''] * len(s)
def style_div_df(df): return df
def highlight_row(row): return [''] * len(row)
def style_pct_df(df): return df

def run_trade_ideas_app(df_global):
    st.title("💡 Trade Ideas AI")
    # Placeholder
    st.info("AI Analysis requires API configuration. (Placeholder)")
    
# --- MAIN EXECUTION ---
try:
    # Use secrets for sheet url if available
    sheet_url = st.secrets.get("GSHEET_URL", "")
    if sheet_url:
        df_global = load_and_clean_data(sheet_url)
    else:
        df_global = pd.DataFrame() 

    pg = st.navigation([
        st.Page(lambda: run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        st.Page(lambda: run_rsi_scanner_app(df_global), title="RSI Scanner", icon="📈", url_path="rsi_scanner"), 
        st.Page(lambda: run_trade_ideas_app(df_global), title="Trade Ideas", icon="💡", url_path="trade_ideas"),
    ])
    st.sidebar.caption("🖥️ RSI Sandbox v5.0 (Bulltard Logic)")
    pg.run()
    
except Exception as e:
    st.error(f"Application Error: {e}")
