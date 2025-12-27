import warnings
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import math
import os
import glob
import streamlit.components.v1 as components
import requests
import time
import re
from io import StringIO

# --- 1. GLOBAL DATA LOADING & UTILITIES ---
COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=65),
    "Strike": st.column_config.TextColumn("Strike", width=95),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=90),
    "Contracts": st.column_config.NumberColumn("Qty", width=60),
    "Dollars": st.column_config.NumberColumn("Dollars", width=110),
}

@st.cache_data(ttl=600, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    want = ["Trade Date","Order Type","Symbol","Strike (Actual)","Strike","Expiry","Contracts","Dollars","Error"]
    keep = [c for c in want if c in df.columns]
    df = df[keep].copy()
    
    for col in ["Order Type", "Symbol", "Strike", "Expiry"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    if "Dollars" in df.columns:
        df["Dollars"] = (df["Dollars"].astype(str)
                         .str.replace("$", "", regex=False)
                         .str.replace(",", "", regex=False))
        df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)

    if "Contracts" in df.columns:
        df["Contracts"] = (df["Contracts"].astype(str)
                           .str.replace(",", "", regex=False))
        df["Contracts"] = pd.to_numeric(df["Contracts"], errors="coerce").fillna(0)
    
    if "Trade Date" in df.columns:
        df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    
    if "Expiry" in df.columns:
        df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
        
    if "Strike (Actual)" in df.columns:
        df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
        
    if "Error" in df.columns:
        df = df[~df["Error"].astype(str).str.upper().isin(["TRUE","1","YES"])]
        
    return df

@st.cache_data(ttl=3600)
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        try:
            return float(t.fast_info['marketCap'])
        except:
            pass
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
        ema21 = h["Close"].ewm(span=21, adjust=False).mean()
        latest_price = h["Close"].iloc[-1]
        latest_ema = ema21.iloc[-1]
        return latest_price > latest_ema
    except:
        return True

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0:
        return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

def highlight_expiry(val):
    try:
        expiry_date = datetime.strptime(val, "%d %b %y").date()
        today = date.today()
        this_fri = today + timedelta(days=(4 - today.weekday()) % 7)
        next_fri = this_fri + timedelta(days=7)
        two_fri = this_fri + timedelta(days=14)
        if expiry_date < today: return "" 
        
        if expiry_date == this_fri: return "background-color: #b7e1cd; color: black;" 
        elif expiry_date == next_fri: return "background-color: #fce8b2; color: black;" 
        elif expiry_date == two_fri: return "background-color: #f4c7c3; color: black;" 
        return ""
    except: return ""

def clean_strike_fmt(val):
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except: return str(val)

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid_dates = df["Trade Date"].dropna()
        if not valid_dates.empty:
            return valid_dates.max().date()
    return date.today() - timedelta(days=1)

# --- RSI DIVERGENCE HELPERS & CONSTANTS ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21
EV_LOOKBACK_YEARS = 3
MIN_N_THRESHOLD = 5

def get_confirmed_gdrive_data(url):
    """Bypasses the 'File too large to scan' warning page using a token handshake."""
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
        st.error(f"Fetch Error: {e}")
        return None

def load_dataset_config():
    """Reads the TXT file from Drive and returns a dictionary {Name: SecretKey}"""
    try:
        if "URL_CONFIG" not in st.secrets:
            # Fallback if config url isn't set, but specific keys are
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
    colors = {f"EMA{EMA8_PERIOD}": "#4a90e2", f"EMA{EMA21_PERIOD}": "#9b59b6", "VOL_HIGH": "#e67e22", "V_GROW": "#27ae60"}
    html_str = ''
    for t in tags:
        color = colors.get(t, "#7f8c8d")
        html_str += f'<span class="tag-bubble" style="background-color: {color};">{t}</span>'
    return html_str

def calculate_ev_data(df, target_rsi, periods, current_price):
    if df.empty or pd.isna(target_rsi): return None
    cutoff_date = df.index.max() - timedelta(days=365 * EV_LOOKBACK_YEARS)
    hist_df = df[df.index >= cutoff_date].copy()
    mask = (hist_df['RSI'] >= target_rsi - 2) & (hist_df['RSI'] <= target_rsi + 2)
    indices = np.where(mask)[0]
    returns = []
    for idx in indices:
        if idx + periods < len(hist_df):
            entry_p = hist_df.iloc[idx]['Price']
            exit_p = hist_df.iloc[idx + periods]['Price']
            if entry_p > 0: returns.append((exit_p - entry_p) / entry_p)
    if not returns or len(returns) < MIN_N_THRESHOLD: return None
    avg_ret = np.mean(returns)
    ev_price = current_price * (1 + avg_ret)
    return {"price": ev_price, "n": len(returns), "return": avg_ret}

def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    w_close_col, w_vol_col, w_rsi_col = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'
    w_ema8_col, w_ema21_col = 'W_EMA_8', 'W_EMA_21'
    w_high_col, w_low_col = 'W_HIGH', 'W_LOW'
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low', d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'}, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    if all(c in df.columns for c in [w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col]):
        df_w = df[[w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col, w_ema8_col, w_ema21_col]].copy()
        df_w.rename(columns={w_close_col: 'Price', w_vol_col: 'Volume', w_high_col: 'High', w_low_col: 'Low', w_rsi_col: 'RSI', w_ema8_col: 'EMA8', w_ema21_col: 'EMA21'}, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: df_w = None
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences
    latest_p = df_tf.iloc[-1]
    ev30 = calculate_ev_data(df_tf, latest_p['RSI'], 30, latest_p['Price'])
    ev90 = calculate_ev_data(df_tf, latest_p['RSI'], 90, latest_p['Price'])
    def get_date_str(p): return df_tf.loc[p.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else p.name.strftime('%Y-%m-%d')
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        for s_type in ['Bullish', 'Bearish']:
            trigger = False
            if s_type == 'Bullish' and p2['Low'] < lookback['Low'].min():
                p1 = lookback.loc[lookback['RSI'].idxmin()]
                if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD) and not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any(): trigger = True
            elif s_type == 'Bearish' and p2['High'] > lookback['High'].max():
                p1 = lookback.loc[lookback['RSI'].idxmax()]
                if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD) and not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any(): trigger = True
            if trigger:
                post_df = df_tf.iloc[i + 1 :]
                valid = True
                if s_type == 'Bullish' and not post_df.empty and (post_df['RSI'] <= p1['RSI']).any(): valid = False
                if s_type == 'Bearish' and not post_df.empty and (post_df['RSI'] >= p1['RSI']).any(): valid = False
                if valid:
                    tags = []
                    if s_type == 'Bullish':
                        if latest_p['Price'] >= latest_p.get('EMA8', 0): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] >= latest_p.get('EMA21', 0): tags.append(f"EMA{EMA21_PERIOD}")
                    else:
                        if latest_p['Price'] <= latest_p.get('EMA8', 999999): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] <= latest_p.get('EMA21', 999999): tags.append(f"EMA{EMA21_PERIOD}")
                    if is_vol_high: tags.append("VOL_HIGH")
                    if p2['Volume'] > p1['Volume']: tags.append("V_GROW")
                    divergences.append({
                        'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                        'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                        'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                        'P1 Price': f"${p1['Low' if s_type=='Bullish' else 'High']:,.2f}", 
                        'P2 Price': f"${p2['Low' if s_type=='Bullish' else 'High']:,.2f}", 
                        'Last Close': f"${latest_p['Price']:,.2f}", # Renamed Column
                        'ev30_raw': ev30, 'ev90_raw': ev90
                    })
    return divergences

# --- 2. APP MODULES ---

def run_options_database_app(df):
    st.title("📂 Options Database")
    max_data_date = get_max_trade_date(df)
    
    # Removed the control-box wrapper
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
    
    # Removed the control-box wrapper
    c1, c2, c3, c_pad = st.columns([1.2, 1.2, 0.8, 3], gap="small")
    with c1: rank_start = st.date_input("Trade Start Date", value=start_default, key="rank_start")
    with c2: rank_end = st.date_input("Trade End Date", value=max_data_date, key="rank_end")
    with c3: limit = st.number_input("Limit", value=20, min_value=1, max_value=200, key="rank_limit")
    
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
        "Trade Count": st.column_config.NumberColumn("Qty", width=40),
        "Last Trade": st.column_config.TextColumn("Last", width=70),
        "Dollars": st.column_config.NumberColumn("Dollars", width=90),
        "Score": st.column_config.NumberColumn("Score", width=40),
    }
    fmt_currency = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"
    fmt_score = lambda x: f"({abs(int(x))})" if x < 0 else f"{int(x)}"
    
    bull_df = res[display_cols].sort_values(by=["Score", "Dollars"], ascending=[False, False]).head(limit)
    bear_df = res[display_cols].sort_values(by=["Score", "Dollars"], ascending=[True, True]).head(limit)
    
    st.caption("ℹ️ Ranking tables vary from Bulltard's as he includes expired trades and these do not.")
    st.caption("ℹ️ Tickers with the same score are sorted in descending order based on Dollars.")
    
    # --- TRENDING CALCULATION (MOVED UP FOR LAYOUT) ---
    trend_start_date = max_data_date - timedelta(days=14)
    df_trend = df[(df["Trade Date"].dt.date >= trend_start_date) & (df["Trade Date"].dt.date <= max_data_date)].copy()
    
    movers = pd.DataFrame() # Empty default
    if not df_trend.empty:
        df_trend_g = df_trend.groupby(["Trade Date", "Symbol", order_type_col]).size().unstack(fill_value=0)
        for t_type in target_types:
            if t_type not in df_trend_g.columns: df_trend_g[t_type] = 0
        
        df_trend_g["DailyScore"] = df_trend_g["Calls Bought"] + df_trend_g["Puts Sold"] - df_trend_g["Puts Bought"]
        daily_scores = df_trend_g["DailyScore"].reset_index()
        daily_scores["DayRank"] = daily_scores.groupby("Trade Date")["DailyScore"].rank(method="min", ascending=False)
        rank_matrix = daily_scores.pivot(index="Symbol", columns="Trade Date", values="DayRank")
        
        if len(rank_matrix.columns) >= 2:
            latest_date = rank_matrix.columns[-1]
            avg_rank = rank_matrix.mean(axis=1)
            current_rank = rank_matrix[latest_date]
            trend_df = pd.DataFrame({"Current Rank": current_rank, "Avg Rank (2w)": avg_rank})
            trend_df["Trend Score"] = trend_df["Avg Rank (2w)"] - trend_df["Current Rank"]
            movers = trend_df.dropna().sort_values("Trend Score", ascending=False).head(15).reset_index()

    # --- 3-COLUMN LAYOUT ---
    c_bull, c_bear, c_trend = st.columns(3, gap="medium")
    
    with c_bull:
        st.markdown("<h3 style='color: #71d28a; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;'>📈 Bullish Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(bull_df.style.format({"Dollars": fmt_currency, "Trade Count": "{:,.0f}", "Score": fmt_score}), use_container_width=True, hide_index=True, height=get_table_height(bull_df), column_config=rank_col_config)
    
    with c_bear:
        st.markdown("<h3 style='color: #f29ca0; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;'>📉 Bearish Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(bear_df.style.format({"Dollars": fmt_currency, "Trade Count": "{:,.0f}", "Score": fmt_score}), use_container_width=True, hide_index=True, height=get_table_height(bear_df), column_config=rank_col_config)

    with c_trend:
        st.markdown("<h3 style='font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;'>🔥 Trending Tickers</h3>", unsafe_allow_html=True)
        if not movers.empty:
            # Removed background_gradient to avoid matplotlib dependency
            # Using basic text color map instead
            def color_trend(val):
                if val > 0: return 'color: #71d28a; font-weight: bold;'
                if val < 0: return 'color: #f29ca0; font-weight: bold;'
                return ''
                
            st.dataframe(
                movers.style.format({"Current Rank": "{:.0f}", "Avg Rank (2w)": "{:.1f}", "Trend Score": "{:+.1f}"})
                .map(color_trend, subset=["Trend Score"]),
                use_container_width=True,
                hide_index=True,
                height=get_table_height(movers)
            )
        else:
            st.info("Insufficient data for trend analysis.")

def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    exp_range_default = (date.today() + timedelta(days=365))
    
    # --- LAYOUT: LEFT SIDEBAR (SETTINGS) | RIGHT COLUMN (GRAPHICS) ---
    col_settings, col_visuals = st.columns([1, 2.5], gap="large")
    
    with col_settings:
        # Removed "Settings" header here
        ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
        td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
        td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
        exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
        
        st.markdown("---")
        
        # Split remaining settings into two sub-columns
        # Left sub: View/Width | Right sub: Checkboxes
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
            
        # Hardcoded defaults since "Other Options" UI was removed
        hide_empty = True
        show_table = True
    
    # Use a container in the right column to act as a placeholder for the charts
    with col_visuals:
        chart_container = st.container()

    # --- DATA PROCESSING ---
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

    # --- TRACKING SELECTIONS ---
    if "Include" not in edit_pool_raw.columns:
        edit_pool_raw.insert(0, "Include", True)
    
    edit_pool_raw["Trade Date Str"] = edit_pool_raw["Trade Date"].dt.strftime("%d %b %y")
    edit_pool_raw["Expiry Str"] = edit_pool_raw["Expiry_DT"].dt.strftime("%d %b %y")

    # --- DATA TABLE (BOTTOM) ---
    if show_table:
        # Prepare display DF for editor: convert numbers to strings for formatting
        editor_input = edit_pool_raw[["Include", "Trade Date Str", order_type_col, "Symbol", "Strike", "Expiry Str", "Contracts", "Dollars"]].copy()
        
        # Format columns with commas
        editor_input["Dollars"] = editor_input["Dollars"].apply(lambda x: f"${x:,.0f}")
        editor_input["Contracts"] = editor_input["Contracts"].apply(lambda x: f"{x:,.0f}")

        st.markdown("---")
        st.subheader("Data Table & Selection")
        
        edited_df = st.data_editor(
            editor_input,
            column_config={
                "Include": st.column_config.CheckboxColumn("Include", default=True),
                # Changed to TextColumn to support the pre-formatted strings
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
        # Apply mask to original numeric data
        f = edit_pool_raw[edited_df["Include"]].copy()
    else:
        f = edit_pool_raw.copy()

    # --- VISUALS RENDERING (IN RIGHT COLUMN) ---
    with chart_container:
        if f.empty:
            st.info("No rows selected. Check the 'Include' boxes below.")
        else:
            @st.cache_data(ttl=300)
            def get_stock_indicators(sym: str):
                try:
                    ticker_obj = yf.Ticker(sym)
                    h = ticker_obj.history(period="60d", interval="1d")
                    if len(h) == 0: return None, None, None, None, None
                    close = h["Close"]
                    spot_val = float(close.iloc[-1])
                    ema8  = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
                    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
                    sma200_full = ticker_obj.history(period="2y")["Close"]
                    sma200 = float(sma200_full.rolling(window=200).mean().iloc[-1]) if len(sma200_full) >= 200 else None
                    return spot_val, ema8, ema21, sma200, h
                except: return None, None, None, None, None

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

            f["Signed Dollars"] = f.apply(lambda r: (1 if r[order_type_col] in ("Calls Bought","Puts Sold") else -1) * (r["Dollars"] or 0.0), axis=1)
            fmt_neg = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

            if view_mode == "Price Zones":
                strike_min, strike_max = float(np.nanmin(f["Strike (Actual)"].values)), float(np.nanmax(f["Strike (Actual)"].values))
                if width_mode == "Auto": zone_w = float(next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, strike_max - strike_min) / 12.0)), 100))
                else: zone_w = float(fixed_size_choice)
                
                n_dn, n_up = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w)), int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
                lower_edge = spot - n_dn * zone_w
                total = max(1, n_dn + n_up)
                f["ZoneIdx"] = f["Strike (Actual)"].apply(lambda x: min(total - 1, max(0, int(math.floor((x - lower_edge) / zone_w)))))
                agg = f.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                zone_df = pd.DataFrame([(z, lower_edge + z*zone_w, lower_edge + (z+1)*zone_w) for z in range(total)], columns=["ZoneIdx","Zone_Low","Zone_High"])
                zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
                if hide_empty: zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
                
                st.markdown('<div class="zones-panel">', unsafe_allow_html=True)
                for _, r in zs.sort_values("ZoneIdx", ascending=False).iterrows():
                    if r["Zone_Low"] + (zone_w/2) > spot:
                        color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, zs["Net_Dollars"].abs().max()))*420))
                        val_str = fmt_neg(r["Net_Dollars"])
                        st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>', unsafe_allow_html=True)
                for _, r in zs.sort_values("ZoneIdx", ascending=False).iterrows():
                    if r["Zone_Low"] + (zone_w/2) < spot:
                        color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, zs["Net_Dollars"].abs().max()))*420))
                        val_str = fmt_neg(r["Net_Dollars"])
                        st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                e = f.copy()
                e["Bucket"] = pd.cut((pd.to_datetime(e["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days), bins=[0, 7, 30, 90, 180, 10000], labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"], include_lowest=True)
                agg = e.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
                for _, r in agg.iterrows():
                    color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, agg["Net_Dollars"].abs().max()))*420))
                    val_str = fmt_neg(r["Net_Dollars"])
                    st.markdown(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
            
            st.caption("ℹ️ You can exclude individual trades from the graphic by unchecking them in the Data Tables box below.")

def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = get_max_trade_date(df)
            
    # --- SPLIT LAYOUT: FILTERS (LEFT) | CALCULATOR (RIGHT) ---
    col_filters, col_calculator = st.columns([1, 1], gap="medium")
    
    # --- MOVED STYLES HERE TO PREVENT LAYOUT SHIFT ---
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

    # --- LEFT COLUMN: FILTERS ---
    with col_filters:
        # Standardized Header
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>🔍 Filters</h4>", unsafe_allow_html=True)
        # Row 1 of Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1: td_start = st.date_input("Trade Start Date", value=max_data_date, key="pv_start")
        with fc2: td_end = st.date_input("Trade End Date", value=max_data_date, key="pv_end")
        with fc3: ticker_filter = st.text_input("Ticker (blank=all)", value="", key="pv_ticker").strip().upper()
        
        # Row 2 of Filters
        fc4, fc5, fc6 = st.columns(3)
        with fc4: min_notional = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}[st.selectbox("Min Dollars", options=["0M", "5M", "10M", "50M", "100M"], index=0, key="pv_notional")]
        with fc5: min_mkt_cap = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}[st.selectbox("Mkt Cap Min", options=["0B", "10B", "50B", "100B", "200B", "500B", "1T"], index=0, key="pv_mkt_cap")]
        with fc6: ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")
        
        st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Market Cap filtering can be buggy. If empty, reset \'Mkt Cap Min\' to 0B.</div>', unsafe_allow_html=True)
        st.markdown('<div class="light-note">ℹ️ Scroll to the bottom to see Risk Reversals.</div>', unsafe_allow_html=True)

    # --- RIGHT COLUMN: CALCULATOR ---
    with col_calculator:
        # Removed the wrapper box entirely as requested
        # Standardized Header to match Filters exactly
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
        
        # Row 1 of Calculator (Inputs)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: c_strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0, format="%.2f", key="calc_strike")
        with cc2: c_premium = st.number_input("Premium", min_value=0.00, value=2.50, step=0.05, format="%.2f", key="calc_premium")
        with cc3: c_expiry = st.date_input("Expiration", value=date.today() + timedelta(days=30), key="calc_expiry")
        
        dte = (c_expiry - date.today()).days
        coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
        annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0

        st.session_state["calc_out_ann"] = f"{annual_ret:.2f}%"
        st.session_state["calc_out_coc"] = f"{coc_ret:.2f}%"
        st.session_state["calc_out_dte"] = str(max(0, dte))

        # Row 2 of Calculator (Outputs)
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
    
    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    if d_range.empty: return

    order_type_col = "Order Type" if "Order Type" in d_range.columns else "Order type"
    cb_pool = d_range[d_range[order_type_col] == "Calls Bought"].copy()
    ps_pool = d_range[d_range[order_type_col] == "Puts Sold"].copy()
    pb_pool = d_range[d_range[order_type_col] == "Puts Bought"].copy()
    
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    cb_pool['occ'], ps_pool['occ'] = cb_pool.groupby(keys).cumcount(), ps_pool.groupby(keys).cumcount()
    rr_matches = pd.merge(cb_pool, ps_pool, on=keys + ['occ'], suffixes=('_c', '_p'))
    
    rr_rows = []
    for idx, row in rr_matches.iterrows():
        rr_rows.append({'Symbol': row['Symbol'], 'Trade Date': row['Trade Date'], 'Expiry_DT': row['Expiry_DT'], 'Contracts': row['Contracts'], 'Dollars': row['Dollars_c'], 'Strike': clean_strike_fmt(row['Strike_c']), 'Pair_ID': idx, 'Pair_Side': 0})
        rr_rows.append({'Symbol': row['Symbol'], 'Trade Date': row['Trade Date'], 'Expiry_DT': row['Expiry_DT'], 'Contracts': row['Contracts'], 'Dollars': row['Dollars_p'], 'Strike': clean_strike_fmt(row['Strike_p']), 'Pair_ID': idx, 'Pair_Side': 1})
    df_rr = pd.DataFrame(rr_rows)

    if not rr_matches.empty:
        match_keys = keys + ['occ']
        def filter_out_matches(pool, matches):
            temp_matches = matches[match_keys].copy()
            temp_matches['_remove'] = True
            merged = pool.merge(temp_matches, on=match_keys, how='left')
            return merged[merged['_remove'].isna()].drop(columns=['_remove'])
        cb_pool = filter_out_matches(cb_pool, rr_matches)
        ps_pool = filter_out_matches(ps_pool, rr_matches)

    def apply_f(data):
        if data.empty: return data
        f = data.copy()
        if ticker_filter: f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
        f = f[f["Dollars"] >= min_notional]
        if not f.empty and min_mkt_cap > 0:
            unique_symbols = f["Symbol"].unique()
            valid_symbols = [s for s in unique_symbols if get_market_cap(s) >= float(min_mkt_cap)]
            f = f[f["Symbol"].isin(valid_symbols)]
        if not f.empty and ema_filter == "Yes":
            unique_symbols = f["Symbol"].unique()
            valid_ema_symbols = [s for s in unique_symbols if is_above_ema21(s)]
            f = f[f["Symbol"].isin(valid_ema_symbols)]
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
        piv["Symbol_Display"] = piv["Symbol"]
        piv.loc[piv["Symbol"] == piv["Symbol"].shift(1), "Symbol_Display"] = ""
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
    st.title("📈 RSI Divergence Scanner")
    
    # --- Custom RSI-Specific Styles ---
    st.markdown("""
        <style>
        .top-note {
            color: #888888;
            font-size: 14px;
            margin-bottom: 2px;
            font-family: inherit;
        }
        
        /* Specific CSS for the RSI table visualization */
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
    data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

    if data_option:
        try:
            target_url = st.secrets[dataset_map[data_option]]
            csv_buffer = get_confirmed_gdrive_data(target_url)
            if csv_buffer and csv_buffer != "HTML_ERROR":
                master = pd.read_csv(csv_buffer)
                date_col = next((col for col in master.columns if 'DATE' in col.upper()), None)
                last_updated_str = pd.to_datetime(master[date_col]).max().strftime('%Y-%m-%d') if date_col else "Unknown"
                
                st.markdown('<div class="top-note">ℹ️ See bottom of page for strategy logic and tag explanations.</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="top-note">📅 Last Updated: {last_updated_str}</div>', unsafe_allow_html=True)
                
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
                # Convert grouping to list to get length for progress bar
                grouped_list = list(grouped)
                total_groups = len(grouped_list)
                
                for i, (ticker, group) in enumerate(grouped_list):
                    d_d, d_w = prepare_data(group.copy())
                    if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
                    if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
                    progress_bar.progress((i + 1) / total_groups)
                
                if raw_results:
                    res_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
                    consolidated = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                    for tf in ['Daily', 'Weekly']:
                        st.divider()
                        st.header(f"📅 {tf} Divergence Analysis")
                        for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                            st.subheader(f"{emoji} {s_type} Signals")
                            tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                            if not tbl_df.empty:
                                # Optimized HTML Table construction
                                html = '<table class="rsi-table"><thead><tr><th style="width:7%">Ticker</th><th style="width:25%">Tags</th><th style="width:8%">P1 Date</th><th style="width:8%">Signal Date</th><th style="width:8%">RSI</th><th style="width:8%">P1 Price</th><th style="width:8%">P2 Price</th><th style="width:8%">Last Close</th><th style="width:10%">EV 30p</th><th style="width:10%">EV 90p</th></tr></thead><tbody>'
                                for _, row in tbl_df.iterrows():
                                    html += '<tr>'
                                    html += f'<td style="text-align:left"><b>{row["Ticker"]}</b></td>'
                                    html += f'<td style="text-align:left">{style_tags(row["Tags"])}</td>'
                                    html += f'<td style="text-align:center">{row["P1 Date"]}</td>'
                                    html += f'<td style="text-align:center">{row["Signal Date"]}</td>'
                                    html += f'<td style="text-align:center">{row["RSI"]}</td>'
                                    html += f'<td style="text-align:left">{row["P1 Price"]}</td>'
                                    html += f'<td style="text-align:left">{row["P2 Price"]}</td>'
                                    html += f'<td style="text-align:left">{row["Last Close"]}</td>'
                                    for ev_key in ['ev30_raw', 'ev90_raw']:
                                        data = row[ev_key]
                                        if data:
                                            is_pos = data['return'] > 0
                                            cls = ("ev-positive" if is_pos else "ev-negative") if s_type == 'Bullish' else ("ev-positive" if not is_pos else "ev-negative")
                                            html += f'<td class="{cls}">{data["return"]*100:+.1f}% <br><small>(${data["price"]:,.2f}, N={data["n"]})</small></td>'
                                        else: html += '<td class="ev-neutral">N/A</td>'
                                    html += '</tr>'
                                html += '</tbody></table>'
                                st.markdown(html, unsafe_allow_html=True)
                            else: st.write("No signals.")
                else: st.warning("No signals.")
                
                # --- Updated Robust Footer ---
                st.divider()
                f_col1, f_col2, f_col3 = st.columns(3)
                
                with f_col1:
                    st.markdown('<div class="footer-header">📉 SIGNAL LOGIC</div>', unsafe_allow_html=True)
                    st.markdown(f"""
                    * **Signal Identification**: Scans for price extremes (New Low for Bullish, New High for Bearish) within a **{SIGNAL_LOOKBACK_PERIOD}-period window**.
                    * **Divergence Mechanism**: Compares the RSI at a new price extreme to a previous RSI extreme found within the **{DIVERGENCE_LOOKBACK}-period lookback**.
                    * **Bullish Standards**: Price hits a new low while RSI is at least **{RSI_DIFF_THRESHOLD} points higher** than at the previous low. RSI must remain below 50 between points.
                    * **Bearish Standards**: Price hits a new high while RSI is at least **{RSI_DIFF_THRESHOLD} points lower** than at the previous high. RSI must remain above 50 between points.
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
                    * **V_GROW**: Triggered if volume at the current signal point (P2) is higher than the volume at the previous extreme (P1).
                    """)
        except Exception as e: st.error(f"Error: {e}")

# --- 3. MAIN EXECUTION ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

# Adaptive styles that respect Light/Dark mode
st.markdown("""<style>
/* Adaptive variables using Streamlit native theme hooks */
.block-container{padding-top:1.2rem;padding-bottom:1rem;}

/* Removed control-box CSS entirely */

.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
.zone-label{width:100px;font-weight:700; text-align: right;}
.zone-bar{height:22px;border-radius:6px;min-width:6px}

/* Consistent semantic colors for charts */
.zone-bull{background: linear-gradient(90deg, #71d28a, #60c57b)}
.zone-bear{background: linear-gradient(90deg, #f29ca0, #e4878d)}
.zone-value{min-width:220px;font-variant-numeric:tabular-nums}

.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: #66b7ff; opacity: 0.4; }
.price-badge { background: rgba(102, 183, 255, 0.1); color: #66b7ff; border: 1px solid rgba(102, 183, 255, 0.5); border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; white-space: nowrap; margin: 0 12px; z-index: 1; }

.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
.badge{background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius:18px; padding:6px 10px; font-weight:700}
.price-badge-header{background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 10px; font-weight:800}

.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; }

/* Hide the native "Sidebar" button if you want it truly fixed, but st.navigation manages this well */
</style>""", unsafe_allow_html=True)

try:
    # Load data
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)
    last_updated_date = df_global["Trade Date"].max().strftime("%d %b %y")

    # --- NAVIGATION SETUP ---
    # Using st.navigation instead of custom buttons. 
    # To fix the "Multiple Pages specified with URL pathname <lambda>" error,
    # we provide a unique url_path for each page that uses a lambda.
    pg = st.navigation({
        "Tools": [
            st.Page(
                lambda: run_options_database_app(df_global), 
                title="Options Database", 
                icon="📂", 
                url_path="options_db", 
                default=True
            ),
            st.Page(
                lambda: run_rankings_app(df_global), 
                title="Rankings", 
                icon="🏆", 
                url_path="rankings"
            ),
            st.Page(
                lambda: run_pivot_tables_app(df_global), 
                title="Pivot Tables", 
                icon="🎯", 
                url_path="pivot_tables"
            ),
            st.Page(
                lambda: run_strike_zones_app(df_global), 
                title="Strike Zones", 
                icon="📊", 
                url_path="strike_zones"
            ),
            st.Page(
                run_rsi_divergences_app, 
                title="RSI Divergences", 
                icon="📈", 
                url_path="rsi_divergences"
            ),
        ]
    })

    # Add extra info to the sidebar footer
    # Removed the st.markdown("---") to avoid double lines.
    st.sidebar.caption(f"📅 **Last Updated:** {last_updated_date}")
    
    # Execution
    pg.run()
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")
