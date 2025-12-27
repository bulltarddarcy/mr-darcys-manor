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

def get_table_height(df, max_rows=30):
    row_count = len(df)
    if row_count == 0:
        return 100
    display_rows = min(row_count, max_rows)
    return (display_rows + 1) * 35 + 5

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid_dates = df["Trade Date"].dropna()
        if not valid_dates.empty:
            return valid_dates.max().date()
    return date.today() - timedelta(days=1)

def render_page_header(title, df):
    st.markdown(f"<h1 style='margin-bottom: 0px;'>{title}</h1>", unsafe_allow_html=True)
    last_updated = get_max_trade_date(df).strftime("%d %b %y")
    st.markdown(f"<p style='color: #808495; margin-top: 0px; margin-bottom: 25px; font-size: 0.9rem;'>Last Updated: {last_updated}</p>", unsafe_allow_html=True)

# --- 2. APP MODULES ---

def run_options_database_app(df):
    render_page_header("📂 Options Database", df)
    max_data_date = get_max_trade_date(df)
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
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
    
    check_cols = st.columns([0.15, 0.15, 0.15, 0.55])
    with check_cols[0]: inc_cb = st.checkbox("Calls Bought", value=True, key="db_inc_cb")
    with check_cols[1]: inc_ps = st.checkbox("Puts Sold", value=True, key="db_inc_ps")
    with check_cols[2]: inc_pb = st.checkbox("Puts Bought", value=True, key="db_inc_pb")
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    render_page_header("🏆 Rankings", df)
    max_data_date = get_max_trade_date(df)
    start_default = max_data_date - timedelta(days=14)
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c_pad = st.columns([1.2, 1.2, 0.8, 3], gap="small")
    with c1: rank_start = st.date_input("Trade Start Date", value=start_default, key="rank_start")
    with c2: rank_end = st.date_input("Trade End Date", value=max_data_date, key="rank_end")
    with c3: limit = st.number_input("Limit", value=20, min_value=1, max_value=200, key="rank_limit")
    
    st.caption("ℹ️ Ranking tables vary from Bulltard's as he includes expired trades and these do not.")
    st.caption("ℹ️ Tickers with the same score are sorted in descending order based on Dollars.")
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown("<h3 style='color: #71d28a; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;'>Bullish Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(bull_df.style.format({"Dollars": fmt_currency, "Trade Count": "{:,.0f}", "Score": fmt_score}), use_container_width=True, hide_index=True, height=get_table_height(bull_df), column_config=rank_col_config)
    with col_right:
        st.markdown("<h3 style='color: #f29ca0; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;'>Bearish Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(bear_df.style.format({"Dollars": fmt_currency, "Trade Count": "{:,.0f}", "Score": fmt_score}), use_container_width=True, hide_index=True, height=get_table_height(bear_df), column_config=rank_col_config)

def run_pivot_tables_app(df):
    render_page_header("📊 Pivot Tables", df)
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: ticker = st.text_input("Ticker", value="TSLA", key="pv_ticker").strip().upper()
    with c2: start_date = st.date_input("Trade Date (start)", value=None, key="pv_start")
    with c3: end_date = st.date_input("Trade Date (end)", value=None, key="pv_end")
    st.markdown('</div>', unsafe_allow_html=True)

    f = df[df["Symbol"].astype(str).str.upper() == ticker].copy()
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]

    if f.empty:
        st.warning(f"No data found for {ticker} in selected range.")
        return

    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    f["Expiry_Table"] = f["Expiry_DT"].dt.strftime("%d %b %y")

    col_a, col_b = st.columns(2, gap="large")
    
    with col_a:
        st.markdown("<h3 style='color: #71d28a; font-size: 1.1rem;'>Calls Bought</h3>", unsafe_allow_html=True)
        cb = f[f[order_type_col] == "Calls Bought"].copy()
        if not cb.empty:
            cb_pivot = cb.groupby(["Symbol", "Strike", "Expiry_Table"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().sort_values("Dollars", ascending=False)
            st.dataframe(cb_pivot, column_config=COLUMN_CONFIG_PIVOT, use_container_width=True, hide_index=True)
        else: st.info("No Calls Bought found.")

    with col_b:
        st.markdown("<h3 style='color: #f29ca0; font-size: 1.1rem;'>Puts Bought</h3>", unsafe_allow_html=True)
        pb = f[f[order_type_col] == "Puts Bought"].copy()
        if not pb.empty:
            pb_pivot = pb.groupby(["Symbol", "Strike", "Expiry_Table"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().sort_values("Dollars", ascending=False)
            st.dataframe(pb_pivot, column_config=COLUMN_CONFIG_PIVOT, use_container_width=True, hide_index=True)
        else: st.info("No Puts Bought found.")

def run_strike_zones_app(df):
    render_page_header("📊 Strike Zones", df)
    exp_range_default = (date.today() + timedelta(days=365))
    
    input_col, toggle_col = st.columns([1.1, 1], gap="large")
    
    with input_col:
        st.markdown('<div class="control-box">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 12px; font-size: 1rem;'>🔍 Data Filters</h4>", unsafe_allow_html=True)
        f_row1_c1, f_row1_c2 = st.columns(2)
        with f_row1_c1: ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
        with f_row1_c2: exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
        f_row2_c1, f_row2_c2 = st.columns(2)
        with f_row2_c1: td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
        with f_row2_c2: td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
        st.markdown('</div>', unsafe_allow_html=True)

    with toggle_col:
        st.markdown('<div class="calc-box">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 12px; font-size: 1rem; color: #66b7ff;'>⚙️ Visual Controls</h4>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**View Mode**")
            view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], label_visibility="collapsed")
        with c2:
            st.markdown("**Zone Width**")
            width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], label_visibility="collapsed")
            fixed_size_choice = st.select_slider("Size ($)", options=[1, 5, 10, 25, 50, 100], value=10) if width_mode == "Fixed" else 10
        with c3:
            st.markdown("**Include**")
            inc_cb = st.checkbox("Calls Bought", value=True)
            inc_ps = st.checkbox("Puts Sold", value=True)
            inc_pb = st.checkbox("Puts Bought", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

    f_base = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start: f_base = f_base[f_base["Trade Date"].dt.date >= td_start]
    if td_end: f_base = f_base[f_base["Trade Date"].dt.date <= td_end]
    f_base = f_base[(f_base["Expiry_DT"].dt.date >= date.today()) & (f_base["Expiry_DT"].dt.date <= exp_end)]
    
    order_type_col = "Order Type" if "Order Type" in f_base.columns else "Order type"
    allowed_sz_types = [t for t, inc in [("Calls Bought", inc_cb), ("Puts Sold", inc_ps), ("Puts Bought", inc_pb)] if inc]
    edit_pool_raw = f_base[f_base[order_type_col].isin(allowed_sz_types)].copy()
    
    if edit_pool_raw.empty:
        st.warning("No trades match current filters.")
        return

    if "Include" not in edit_pool_raw.columns: edit_pool_raw.insert(0, "Include", True)
    edit_pool_raw["Trade Date Str"] = edit_pool_raw["Trade Date"].dt.strftime("%d %b %y")
    edit_pool_raw["Expiry Str"] = edit_pool_raw["Expiry_DT"].dt.strftime("%d %b %y")

    st.markdown('<div style="color: #808495; font-size: 0.85rem; margin-bottom: 15px;">ℹ️ Remove outliers by unchecking them below.</div>', unsafe_allow_html=True)
    
    edited_df = st.data_editor(edit_pool_raw[["Include", "Trade Date Str", order_type_col, "Symbol", "Strike", "Expiry Str", "Contracts", "Dollars"]], hide_index=True, use_container_width=True, key="sz_editor")
    f = edit_pool_raw[edited_df["Include"]].copy()

    spot, ema8, ema21, sma200, _ = get_stock_indicators(ticker)
    if spot:
        pct = lambda x: f"{(x/spot-1)*100:+.1f}%" if x else "—"
        st.markdown(f'<div class="metric-row"><span class="price-badge-header">Price: ${spot:,.2f}</span><span class="badge">EMA(8): ${ema8:,.2f} ({pct(ema8)})</span><span class="badge">EMA(21): ${ema21:,.2f} ({pct(ema21)})</span></div>', unsafe_allow_html=True)

    f["Signed Dollars"] = f.apply(lambda r: (1 if r[order_type_col] in ("Calls Bought","Puts Sold") else -1) * (r["Dollars"] or 0.0), axis=1)
    
    if view_mode == "Price Zones":
        strike_min, strike_max = float(np.nanmin(f["Strike (Actual)"].values)), float(np.nanmax(f["Strike (Actual)"].values))
        zone_w = float(next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, strike_max - strike_min) / 12.0)), 100)) if width_mode == "Auto" else float(fixed_size_choice)
        # Visual rendering logic here...
        st.info("Price Zone Graphic Placeholder")
    else:
        st.info("Expiry Buckets Graphic Placeholder")

def run_rsi_divergences_app(df):
    render_page_header("📈 RSI Divergences", df)
    st.markdown("""<style>div.stLinkButton > a { background: linear-gradient(45deg, #ff00ff, #00ffff, #ff0000, #ffff00, #00ff00); background-size: 400% 400%; animation: tie-dye 10s ease infinite; border: none; color: white !important; font-weight: bold; padding: 15px 30px; border-radius: 10px; text-decoration: none; display: inline-block; transition: transform 0.2s; } div.stLinkButton > a:hover { transform: scale(1.05); } @keyframes tie-dye { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }</style>""", unsafe_allow_html=True)
    st.link_button("🌈 🚀 Click to travel to the new RSI Divergence website 🚀 🌈", "https://mr-darcys-rsi-divergence.streamlit.app/")

# --- 3. CSS & PAGE SETUP ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")
st.markdown("""<style>
:root{--bg:#1f1f22; --panel:#2a2d31; --panel2:#24272b; --text:#e7e7ea; --green:#71d28a; --red:#f29ca0; --line:#66b7ff;}
.block-container{padding-top: 2rem !important;}
header[data-testid="stHeader"] {background: transparent !important;}

/* Fix for the black/green boxes */
.nav-container {padding: 0; margin-bottom: 0px;}
.control-box{padding:15px; border-radius:10px; background-color: var(--panel2); border: 1px solid #3a3f45; margin-bottom: 20px;}
.calc-box {padding: 15px; border-radius: 10px; background-color: transparent; border: 1px solid rgba(113, 210, 138, 0.2); margin-bottom: 20px;}

div.stButton > button.nav-btn {background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; border-radius: 0px !important; color: #808495 !important; font-weight: 600 !important; padding-bottom: 8px !important; height: auto !important;}
div.stButton > button.nav-btn:hover {color: #ffffff !important; border-bottom: 2px solid #555 !important;}
div.stButton > button.nav-active {color: #66b7ff !important; border-bottom: 2px solid #66b7ff !important;}

.zones-panel{padding:14px 0; border-radius:10px;} .zone-row{display:flex;align-items:center;gap:12px;margin:10px 0;} .zone-label{width:100px;font-weight:700; text-align: right;} .zone-bar{height:22px;border-radius:6px;min-width:6px} .zone-bull{background:linear-gradient(90deg,var(--green),#60c57b)} .zone-bear{background:linear-gradient(90deg,var(--red),#e4878d)} .zone-value{min-width:220px;font-variant-numeric:tabular-nums}
.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; } .price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: var(--line); opacity: 0.6; } .price-badge { background: #2b3a45; color: #bfe7ff; border: 1px solid #56b6ff; border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; box-shadow: 0 2px 8px rgba(0,0,0,0.35); white-space: nowrap; margin: 0 12px; z-index: 1; }
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0} .badge{background:#2b3a45;border:1px solid #3b5566;color:#cde8ff;border-radius:18px;padding:6px 10px;font-weight:700} .price-badge-header{background:#2b3a45;border:1px solid #56b6ff;color:#bfe7ff;border-radius:18px;padding:6px 10px;font-weight:800}
th,td{border:1px solid #3a3f45;padding:8px} th{background:#343a40;text-align:left}</style>""", unsafe_allow_html=True)

try:
    sheet_url = st.secrets["GSHEET_URL"]; df_global = load_and_clean_data(sheet_url)
    if "app_choice" not in st.session_state: st.session_state["app_choice"] = "Options Database"
    nav_items = ["Options Database", "Rankings", "Pivot Tables", "Strike Zones", "RSI Divergences"]
    
    # Navigation Row
    cols = st.columns([1.1, 0.8, 1, 1, 1.2, 3])
    for i, item in enumerate(nav_items):
        is_active = st.session_state["app_choice"] == item
        btn_class = "nav-active" if is_active else "nav-btn"
        if cols[i].button(item, key=f"nav_btn_{item}", use_container_width=True):
            st.session_state["app_choice"] = item; st.rerun()
    st.markdown("<hr style='margin-top: -15px; margin-bottom: 25px; opacity: 0.15; height: 1px; border: none; background-color: #555;'>", unsafe_allow_html=True)

    current_choice = st.session_state["app_choice"]
    if current_choice == "Options Database": run_options_database_app(df_global)
    elif current_choice == "Rankings": run_rankings_app(df_global)
    elif current_choice == "Pivot Tables": run_pivot_tables_app(df_global)
    elif current_choice == "Strike Zones": run_strike_zones_app(df_global)
    elif current_choice == "RSI Divergences": run_rsi_divergences_app(df_global)
except Exception as e: st.error(f"Error: {e}")
