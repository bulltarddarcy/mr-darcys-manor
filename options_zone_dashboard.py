import warnings
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import yfinance as yf
import math
import streamlit_authenticator as stauth

# --- 1. AUTHENTICATION SETUP ---
credentials = {
    "usernames": {
        "admin": {
            "name": "Admin",
            "password": "tape-curtain-phone" 
        },
        "mister": {
            "name": "Mister", 
            "password": "darcy"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials, 
    "options_dashboard_cookie", 
    "abcdef", 
    cookie_expiry_days=30
)

authenticator.login(location='main')

# --- 2. GLOBAL DATA LOADING & UTILITIES ---
@st.cache_data(show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    want = ["Trade Date","Order Type","Symbol","Strike (Actual)","Strike","Expiry","Contracts","Dollars","Error"]
    keep = [c for c in want if c in df.columns]
    df = df[keep].copy()
    
    # Clean whitespace to ensure matches are clean
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
        mc = t.info.get('marketCap', 0)
        return float(mc) if mc else 0.0
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
    """Highlights Expiry based on which Friday it falls on."""
    try:
        expiry_date = datetime.strptime(val, "%d %b %y").date()
        today = date.today()
        
        # Calculate reference Fridays
        this_fri = today + timedelta(days=(4 - today.weekday()) % 7)
        next_fri = this_fri + timedelta(days=7)
        two_fri = this_fri + timedelta(days=14)
        
        if expiry_date < today:
            return "" 
        
        if expiry_date == this_fri:
            return "background-color: #2d5a27; color: white;" 
        elif expiry_date == next_fri:
            return "background-color: #8c5e03; color: white;" 
        elif expiry_date == two_fri:
            return "background-color: #7d3c3c; color: white;" 
        return ""
    except:
        return ""

def clean_strike_fmt(val):
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except:
        return str(val)

# Shrunk column widths to fit 3 tables side-by-side without horizontal scrolling
COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=65),
    "Strike": st.column_config.TextColumn("Strike", width=95),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=90),
    "Contracts": st.column_config.NumberColumn("Qty", width=60),
    "Dollars": st.column_config.NumberColumn("Dollars", width=110),
}

# --- 3. APP MODULES ---

def run_strike_zones_app(df):
    st.title("📊 Options Strike Zones Dashboard")
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
    with c2:
        td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
    with c3:
        td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
    with c4:
        exp_range_default = (date.today() + timedelta(days=365))
        exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
    st.markdown('</div>', unsafe_allow_html=True)

    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start:
        f = f[f["Trade Date"].dt.date >= td_start]
    if td_end:
        f = f[f["Trade Date"].dt.date <= td_end]
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    
    used = f[f["Order Type"].isin(["Calls Bought","Puts Sold","Puts Bought"])].copy()
    if used.empty:
        st.warning("No trades match filters.")
        return

    st.subheader("Data Table")
    display_used = used.copy()
    display_used["Trade Date"] = display_used["Trade Date"].dt.strftime("%d %b %y")
    display_used["Expiry"] = pd.to_datetime(display_used["Expiry"]).dt.strftime("%d %b %y")
    # Using format "${:,.0f}" to ensure commas are added to Dollars
    st.dataframe(display_used.style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}), use_container_width=True, hide_index=True, height=get_table_height(display_used, max_rows=30))


def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    yesterday = date.today() - timedelta(days=1)

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1:
        td_start = st.date_input("Trade Start Date", value=yesterday, key="pv_start")
    with c2:
        td_end = st.date_input("Trade End Date", value=yesterday, key="pv_end")
    with c3:
        ticker_filter = st.text_input("Ticker (blank=all)", value="", key="pv_ticker").strip().upper()
    with c4:
        notional_choices = {"0M": 0, "5M": 5_000_000, "10M": 10_000_000, "50M": 50_000_000, "100M": 100_000_000}
        min_notional = notional_choices[st.selectbox("Min Dollars", options=list(notional_choices.keys()), index=1, key="pv_notional")]
    with c5:
        mkt_cap_choices = {"0B": 0, "100B": 100e9, "200B": 200e9, "500B": 500e9, "1T": 1e12}
        min_mkt_cap = mkt_cap_choices[st.selectbox("Mkt Cap Min", options=list(mkt_cap_choices.keys()), index=1, key="pv_mkt_cap")]
    with c6:
        ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- RR Pairing Engine ---
    # Processing the entire date range to find pairs
    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    d_range['_original_idx'] = d_range.index
    
    cb_pool = d_range[d_range["Order Type"] == "Calls Bought"].copy()
    ps_pool = d_range[d_range["Order Type"] == "Puts Sold"].copy()
    
    # Strictly matching on Trade Date (Col A), Symbol (Col C), Expiry_DT (Col E), and Contracts (Col F)
    match_keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    
    # Use sequence grouping to pair identical trades 1:1
    cb_pool['occ'] = cb_pool.groupby(match_keys).cumcount()
    ps_pool['occ'] = ps_pool.groupby(match_keys).cumcount()
    
    rr_matches = pd.merge(cb_pool, ps_pool, on=match_keys + ['occ'], suffixes=('_c', '_p'))
    
    used_cb_ids = rr_matches['_original_idx_c'].tolist()
    used_ps_ids = rr_matches['_original_idx_p'].tolist()
    
    # Solo trade pools excluding those used in Risk Reversals
    df_cb_solo = cb_pool[~cb_pool['_original_idx'].isin(used_cb_ids)].copy()
    df_ps_solo = ps_pool[~ps_pool['_original_idx'].isin(used_ps_ids)].copy()
    
    # Constructing the Risk Reversal table
    df_rr = pd.DataFrame(columns=['Symbol', 'Strike', 'Expiry_DT', 'Contracts', 'Dollars'])
    if not rr_matches.empty:
        df_rr_matched = pd.DataFrame()
        df_rr_matched['Symbol'] = rr_matches['Symbol']
        df_rr_matched['Trade Date'] = rr_matches['Trade Date']
        df_rr_matched['Expiry_DT'] = rr_matches['Expiry_DT']
        df_rr_matched['Contracts'] = rr_matches['Contracts']
        df_rr_matched['Dollars'] = rr_matches['Dollars_c'] + rr_matches['Dollars_p']
        # Combine strikes - removed the "c" and "p" suffixes and used " & " as separator
        df_rr_matched['Strike'] = rr_matches['Strike_c'].apply(clean_strike_fmt) + " & " + rr_matches['Strike_p'].apply(clean_strike_fmt)
        df_rr = df_rr_matched

    def apply_filters(data, exclude_filters=False):
        if data.empty: return data
        f_data = data.copy()
        
        # 1. Ticker Filter (Always applies)
        if ticker_filter: 
            f_data = f_data[f_data["Symbol"].astype(str).str.upper() == ticker_filter]
        
        # Size & EMA Filters (Conditional)
        if not exclude_filters:
            # Min Dollars
            f_data = f_data[f_data["Dollars"] >= min_notional]
            
            # Market Cap
            if not f_data.empty and min_mkt_cap > 0:
                unique_syms = f_data["Symbol"].unique()
                f_data = f_data[f_data["Symbol"].isin([s for s in unique_syms if get_market_cap(s) >= min_mkt_cap])]
                
            # 21-day EMA
            if not f_data.empty and ema_filter == "Yes":
                unique_syms = f_data["Symbol"].unique()
                f_data = f_data[f_data["Symbol"].isin([s for s in unique_syms if is_above_ema21(s)])]
        
        return f_data

    # Solo tables respect all filters; RR table ignores Min Dollars, Mkt Cap Min, and EMA
    df_cb_f = apply_filters(df_cb_solo, exclude_filters=False)
    df_ps_f = apply_filters(df_ps_solo, exclude_filters=False)
    df_rr_f = apply_filters(df_rr, exclude_filters=True)

    def get_ranked_pivot(data):
        if data.empty: return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
        sym_rank = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index()
        piv = piv.merge(sym_rank, on="Symbol")
        piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
        piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        
        # Display Symbol once per group to keep it clean
        piv["Symbol_Display"] = piv["Symbol"]
        piv.loc[piv["Symbol"] == piv["Symbol"].shift(1), "Symbol_Display"] = ""
        
        res = piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})
        return res[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    col1, col2, col3 = st.columns(3)
    # Using format string "${:,.0f}" to ensure Dollars include commas
    currency_format = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}

    with col1:
        st.subheader("Calls Bought")
        tbl = get_ranked_pivot(df_cb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(currency_format).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
        else: st.info("None.")
    with col2:
        st.subheader("Puts Sold")
        tbl = get_ranked_pivot(df_ps_f)
        if not tbl.empty: st.dataframe(tbl.style.format(currency_format).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
        else: st.info("None.")
    with col3:
        st.subheader("Risk Reversals")
        tbl = get_ranked_pivot(df_rr_f)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(currency_format).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
            st.caption("⚠️ RR Table reflects date range only (ignores Ticker, Min Dollars, Mkt Cap, and EMA filters).")
        else: st.info("None.")

# --- 4. MAIN EXECUTION ---
if st.session_state["authentication_status"]:
    st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")
    st.markdown("""<style>:root{--bg:#1f1f22; --panel:#2a2d31; --panel2:#24272b; --text:#e7e7ea; --green:#71d28a; --red:#f29ca0; --line:#66b7ff; --ema8:#b689ff; --ema21:#ffb86b; --sma200:#ffffff; --price:#bfe7ff;}
    html,body,[class*="css"]{color:var(--text)!important;background-color:var(--bg)!important;}
    .block-container{padding-top:1.2rem;padding-bottom:1rem;}
    .control-box{padding:14px 0; border-radius:10px;}
    .price-badge-header{background:#2b3a45;border:1px solid #56b6ff;color:#bfe7ff;border-radius:18px;padding:6px 10px;font-weight:800}
    th,td{border:1px solid #3a3f45;padding:8px} th{background:#343a40;text-align:left}
    .legend-box { padding: 10px; border: 1px solid #3a3f45; border-radius: 8px; margin-top: 20px; font-size: 13px; }
    .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
    .color-dot { width: 12px; height: 12px; border-radius: 50%; }
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.header("Navigation")
        app_choice = st.selectbox("Select Tool", ["Strike Zones", "Pivot Tables"])
        
        st.markdown('<div class="legend-box"><strong>Expiry Legend</strong>', unsafe_allow_html=True)
        st.markdown('<div class="legend-item"><div class="color-dot" style="background:#2d5a27"></div> This Friday</div>', unsafe_allow_html=True)
        st.markdown('<div class="legend-item"><div class="color-dot" style="background:#8c5e03"></div> Next Friday</div>', unsafe_allow_html=True)
        st.markdown('<div class="legend-item"><div class="color-dot" style="background:#7d3c3c"></div> Two Fridays from now</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        authenticator.logout('Logout', 'sidebar')
        
    try:
        sheet_url = st.secrets["GSHEET_URL"]
        df_global = load_and_clean_data(sheet_url)
        if app_choice == "Strike Zones": run_strike_zones_app(df_global)
        else: run_pivot_tables_app(df_global)
    except Exception as e:
        st.error(f"Error: {e}")
elif st.session_state["authentication_status"] is False: st.error('Incorrect password')
elif st.session_state["authentication_status"] is None: st.warning('Please login')
