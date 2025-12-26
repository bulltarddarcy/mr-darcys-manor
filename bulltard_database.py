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

# --- 2. APP MODULES ---

def run_options_database_app(df):
    st.title("📂 Options Database")
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
    
    ot1, ot2, ot3, ot_pad = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: inc_cb = st.checkbox("Calls Bought", value=True, key="db_inc_cb")
    with ot2: inc_ps = st.checkbox("Puts Sold", value=True, key="db_inc_ps")
    with ot3: inc_pb = st.checkbox("Puts Bought", value=True, key="db_inc_pb")
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
    st.title("🏆 Rankings")
    max_data_date = get_max_trade_date(df)
    start_default = max_data_date - timedelta(days=14)
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c_pad = st.columns([1.2, 1.2, 0.8, 3], gap="small")
    with c1: rank_start = st.date_input("Trade Start Date", value=start_default, key="rank_start")
    with c2: rank_end = st.date_input("Trade End Date", value=max_data_date, key="rank_end")
    with c3: limit = st.number_input("Limit", value=20, min_value=1, max_value=200, key="rank_limit")
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
    
    st.caption("ℹ️ Ranking tables vary from Bulltard's as he includes expired trades and these do not.")
    st.caption("ℹ️ Tickers with the same score are sorted in descending order based on Dollars.")
    
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown("<h3 style='color: #71d28a; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;'>Bullish Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(bull_df.style.format({"Dollars": fmt_currency, "Trade Count": "{:,.0f}", "Score": fmt_score}), use_container_width=True, hide_index=True, height=get_table_height(bull_df), column_config=rank_col_config)
    with col_right:
        st.markdown("<h3 style='color: #f29ca0; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0;'>Bearish Rankings</h3>", unsafe_allow_html=True)
        st.dataframe(bear_df.style.format({"Dollars": fmt_currency, "Trade Count": "{:,.0f}", "Score": fmt_score}), use_container_width=True, hide_index=True, height=get_table_height(bear_df), column_config=rank_col_config)

def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    exp_range_default = (date.today() + timedelta(days=365))
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
    with c2: td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
    with c3: td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
    with c4: exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
    
    sc1, sc2, sc3, sc4 = st.columns(4, gap="medium")
    with sc1:
        st.markdown("**View Mode**")
        view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], label_visibility="collapsed")
    with sc2:
        st.markdown("**Zone Width**")
        width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], label_visibility="collapsed")
        if width_mode == "Fixed": 
            fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=10)
        else: fixed_size_choice = 10
    with sc3:
        st.markdown("**Include Order Type**")
        inc_cb = st.checkbox("Calls Bought", value=True)
        inc_ps = st.checkbox("Puts Sold", value=True)
        inc_pb = st.checkbox("Puts Bought", value=True)
    with sc4:
        st.markdown("**Other Options**")
        hide_empty      = st.checkbox("Hide Empty Zones", value=True)
        show_table       = st.checkbox("Show Data Table", value=True)
    
    st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)
        
    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start: f = f[f["Trade Date"].dt.date >= td_start]
    if td_end: f = f[f["Trade Date"].dt.date <= td_end]
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    
    allowed_sz_types = []
    if inc_cb: allowed_sz_types.append("Calls Bought")
    if inc_ps: allowed_sz_types.append("Puts Sold")
    if inc_pb: allowed_sz_types.append("Puts Bought")
    f = f[f[order_type_col].isin(allowed_sz_types)].copy()
    
    if f.empty:
        st.warning("No trades match current filters.")
        return

    # Initialize checkbox state
    if "sz_include_list" not in st.session_state:
        st.session_state["sz_include_list"] = [True] * len(f)
    
    # Sync state if length changed (new ticker/date range)
    if len(st.session_state["sz_include_list"]) != len(f):
        st.session_state["sz_include_list"] = [True] * len(f)

    # Prepare data for the editor
    f_display = f.copy()
    f_display.insert(0, "Include", st.session_state["sz_include_list"])
    f_display["Trade Date"] = f_display["Trade Date"].dt.strftime("%d %b %y")
    f_display["Expiry"] = f_display["Expiry_DT"].dt.strftime("%d %b %y")

    if show_table:
        st.subheader("Data Table (Uncheck to exclude from graphic)")
        # Use data_editor to allow checking/unchecking
        edited_f = st.data_editor(
            f_display[["Include", "Trade Date", order_type_col, "Symbol", "Strike", "Expiry", "Contracts", "Dollars"]],
            column_config={
                "Include": st.column_config.CheckboxColumn("Incl", default=True, width="small"),
                "Dollars": st.column_config.NumberColumn("Dollars", format="$%d")
            },
            hide_index=True,
            use_container_width=True,
            key="sz_editor"
        )
        # Update session state
        st.session_state["sz_include_list"] = edited_f["Include"].tolist()
        # Filter chart data based on checkboxes
        f_chart = f[edited_f["Include"].values].copy()
    else:
        f_chart = f[st.session_state["sz_include_list"]].copy()

    if f_chart.empty:
        st.info("No trades selected. Check rows in the table to display them.")
        return

    # --- Graphic Logic ---
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

    f_chart["Signed Dollars"] = f_chart.apply(lambda r: (1 if r[order_type_col] in ("Calls Bought","Puts Sold") else -1) * (r["Dollars"] or 0.0), axis=1)
    fmt_neg = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

    if view_mode == "Price Zones":
        strike_min, strike_max = float(np.nanmin(f_chart["Strike (Actual)"].values)), float(np.nanmax(f_chart["Strike (Actual)"].values))
        if width_mode == "Auto": zone_w = float(next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, strike_max - strike_min) / 12.0)), 100))
        else: zone_w = float(fixed_size_choice)
        
        n_dn, n_up = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w)), int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
        lower_edge = spot - n_dn * zone_w
        total = max(1, n_dn + n_up)
        f_chart["ZoneIdx"] = f_chart["Strike (Actual)"].apply(lambda x: min(total - 1, max(0, int(math.floor((x - lower_edge) / zone_w)))))
        agg = f_chart.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
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
        e = f_chart.copy()
        e["Bucket"] = pd.cut((pd.to_datetime(e["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days), bins=[0, 7, 30, 90, 180, 10000], labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"], include_lowest=True)
        agg = e.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
        for _, r in agg.iterrows():
            color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, agg["Net_Dollars"].abs().max()))*420))
            val_str = fmt_neg(r["Net_Dollars"])
            st.markdown(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)

def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = get_max_trade_date(df)
            
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1: td_start = st.date_input("Trade Start Date", value=max_data_date, key="pv_start")
    with c2: td_end = st.date_input("Trade End Date", value=max_data_date, key="pv_end")
    with c3: ticker_filter = st.text_input("Ticker (blank=all)", value="", key="pv_ticker").strip().upper()
    with c4: min_notional = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}[st.selectbox("Min Dollars", options=["0M", "5M", "10M", "50M", "100M"], index=0, key="pv_notional")]
    with c5: min_mkt_cap = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}[st.selectbox("Mkt Cap Min", options=["0B", "10B", "50B", "100B", "200B", "500B", "1T"], index=0, key="pv_mkt_cap")]
    with c6: ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")
    
    st.markdown('<div class="light-note">ℹ️ Market Cap filtering can occasionally be buggy. If the tables are not populating, reset \'Mkt Cap Min\' to 0B and then try again.</div>', unsafe_allow_html=True)
    st.markdown('<div class="light-note">ℹ️ If tables appear overlapped, try using a wider monitor or reducing your browser zoom level for an optimal view.</div>', unsafe_allow_html=True)

    # --- Puts Sold Calculator ---
    st.markdown("<hr style='margin: 15px 0; opacity: 0.2;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom: 12px; font-size: 1rem;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
    
    calc_cols = st.columns(6)
    with calc_cols[0]: c_strike = st.number_input("Strike Price", min_value=0.01, value=100.0, step=1.0, format="%.2f", key="calc_strike")
    with calc_cols[1]: c_premium = st.number_input("Premium", min_value=0.00, value=2.50, step=0.05, format="%.2f", key="calc_premium")
    with calc_cols[2]: c_expiry = st.date_input("Expiration", value=date.today() + timedelta(days=30), key="calc_expiry")
    
    dte = (c_expiry - date.today()).days
    coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
    annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0

    st.session_state["calc_out_ann"] = f"{annual_ret:.2f}%"
    st.session_state["calc_out_coc"] = f"{coc_ret:.2f}%"
    st.session_state["calc_out_dte"] = str(max(0, dte))
        
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

    with calc_cols[3]: st.text_input("Annualised Return", key="calc_out_ann")
    with calc_cols[4]: st.text_input("Cash on Cash Return", key="calc_out_coc")
    with calc_cols[5]: st.text_input("Days to Expiration", key="calc_out_dte")

    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 20px; margin-bottom: 20px; align-items: center;">
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#b7e1cd"></div> This Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#fce8b2"></div> Next Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#f4c7c3"></div> Two Fridays</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    st.title("📈 RSI Divergences")
    st.markdown("""<style>div.stLinkButton > a { background: linear-gradient(45deg, #ff00ff, #00ffff, #ff0000, #ffff00, #00ff00); background-size: 400% 400%; animation: tie-dye 10s ease infinite; border: none; color: white !important; font-weight: bold; padding: 15px 30px; border-radius: 10px; text-decoration: none; display: inline-block; transition: transform 0.2s; } div.stLinkButton > a:hover { transform: scale(1.05); } @keyframes tie-dye { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }</style>""", unsafe_allow_html=True)
    st.link_button("🌈 🚀 Click to travel to the new RSI Divergence website 🚀 🌈", "https://mr-darcys-rsi-divergence.streamlit.app/")

# --- 3. MAIN EXECUTION ---
if "tool" in st.query_params or "ticker" in st.query_params:
    st.session_state["app_choice_internal"] = st.query_params.get("tool")
    st.session_state["db_ticker"] = st.query_params.get("ticker")
    st.query_params.clear()

st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")
st.markdown("""<style>:root{--bg:#1f1f22; --panel:#2a2d31; --panel2:#24272b; --text:#e7e7ea; --green:#71d28a; --red:#f29ca0; --line:#66b7ff; --ema8:#b689ff; --ema21:#ffb86b; --sma200:#ffffff; --price:#bfe7ff;}
html,body,[class*=\"css\"]{color:var(--text)!important;background-color:var(--bg)!important;}
.block-container{padding-top:1.2rem;padding-bottom:1rem;}
.control-box{padding:14px 0; border-radius:10px;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
.zone-label{width:100px;font-weight:700; text-align: right;}
.zone-bar{height:22px;border-radius:6px;min-width:6px}
.zone-bull{background:linear-gradient(90deg,var(--green),#60c57b)}
.zone-bear{background:linear-gradient(90deg,var(--red),#e4878d)}
.zone-value{min-width:220px;font-variant-numeric:tabular-nums}
.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: var(--line); opacity: 0.6; }
.price-badge { background: #2b3a45; color: #bfe7ff; border: 1px solid #56b6ff; border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; letter-spacing: 0.5px; box-shadow: 0 2px 8px rgba(0,0,0,0.35); white-space: nowrap; margin: 0 12px; z-index: 1; }
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
.badge{background:#2b3a45;border:1px solid #3b5566;color:#cde8ff;border-radius:18px;padding:6px 10px;font-weight:700}
.price-badge-header{background:#2b3a45;border:1px solid #56b6ff;color:#bfe7ff;border-radius:18px;padding:6px 10px;font-weight:800}
th,td{border:1px solid #3a3f45;padding:8px} th{background:#343a40;text-align:left}
.light-note { color: #a1a1a1; font-size: 14px; margin-bottom: 10px; }
</style>""", unsafe_allow_html=True)

try:
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = load_and_clean_data(sheet_url)
    last_updated_date = df_global["Trade Date"].max().strftime("%d %b %y")
    with st.sidebar:
        st.markdown(f"**Last Updated:** {last_updated_date}")
        st.header("Select Tool")
        tools = ["Options Database", "Rankings", "Pivot Tables", "Strike Zones", "RSI Divergences"]
        default_tool_idx = 0
        if "app_choice_internal" in st.session_state:
            try: default_tool_idx = tools.index(st.session_state["app_choice_internal"])
            except: pass
            del st.session_state["app_choice_internal"]
        app_choice = st.selectbox("Select Tool", tools, index=default_tool_idx, label_visibility="collapsed")
    
    if app_choice == "Options Database": run_options_database_app(df_global)
    elif app_choice == "Rankings": run_rankings_app(df_global)
    elif app_choice == "Pivot Tables": run_pivot_tables_app(df_global)
    elif app_choice == "RSI Divergences": run_rsi_divergences_app()
    else: run_strike_zones_app(df_global)
    
except Exception as e: st.error(f"Error: {e}")
