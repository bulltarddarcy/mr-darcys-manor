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
    
    # Strip whitespace from categorical columns
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
    """Highlights Expiry based on proximity to upcoming Fridays."""
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
    """Removes .0 from strikes for cleaner pairing display."""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except:
        return str(val)

# Optimized column widths
COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=65),
    "Strike": st.column_config.TextColumn("Strike", width=95),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=90),
    "Contracts": st.column_config.NumberColumn("Qty", width=60),
    "Dollars": st.column_config.NumberColumn("Dollars", width=90, format="$%d"),
}

# --- 3. APP MODULES ---

def run_strike_zones_app(df):
    """Logic for the original Strike Zones Dashboard"""
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

    with st.sidebar:
        st.header("Display Settings")
        def compact_divider():
            st.markdown('<hr style="margin: 1.0em 0; opacity: 0.15;">', unsafe_allow_html=True)
        st.markdown("**View Mode**")
        view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], label_visibility="collapsed")
        compact_divider()
        st.markdown("**Zone Width**")
        width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], label_visibility="collapsed")
        fixed_size_choice = 10
        if width_mode == "Fixed":
            fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=10)
        compact_divider()
        st.markdown("**Include Order Types**")
        inc_calls_bought = st.checkbox("Calls Bought", value=True)
        inc_puts_sold    = st.checkbox("Puts Sold", value=True)
        inc_puts_bought  = st.checkbox("Puts Bought", value=True)
        compact_divider()
        st.markdown("**Other Options**")
        hide_empty      = st.checkbox("Hide Empty Zones", value=True)
        show_table       = st.checkbox("Show Strike Zone Table", value=True)
        show_raw         = st.checkbox("Show Trades Used Table", value=True)

    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    
    if td_start:
        f = f[f["Trade Date"].dt.date >= td_start]
    if td_end:
        f = f[f["Trade Date"].dt.date <= td_end]
        
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    
    f["Included"] = (
        (f["Order Type"].eq("Calls Bought") & inc_calls_bought) |
        (f["Order Type"].eq("Puts Sold") & inc_puts_sold) |
        (f["Order Type"].eq("Puts Bought") & inc_puts_bought)
    )
    used = f[f["Included"] & f["Order Type"].isin(["Calls Bought","Puts Sold","Puts Bought"])].copy()
    
    if used.empty:
        st.warning("No trades match current filters.")
        return

    @st.cache_data(ttl=300)
    def get_stock_indicators(sym: str):
        try:
            h = yf.Ticker(sym).history(period="2y", interval="1d")
            if len(h) == 0: return None, None, None, None
            close = h["Close"]
            spot_val = float(close.iloc[-1])
            ema8  = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
            ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            sma200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else None
            return spot_val, ema8, ema21, sma200
        except: return None, None, None, None

    spot, ema8, ema21, sma200 = get_stock_indicators(ticker)
    if spot is None:
        spot = st.number_input("Manual Current Price", value=100.0)

    def pct_from_spot(x):
        if x is None or np.isnan(x): return "—"
        return f"{(x/spot-1)*100:+.1f}%"

    badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
    if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct_from_spot(ema8)})</span>')
    if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct_from_spot(ema21)})</span>')
    if sma200: badges.append(f'<span class="badge">SMA(200): ${sma200:,.2f} ({pct_from_spot(sma200)})</span>')
    st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

    def sign_for(order_type: str) -> int:
        if order_type in ("Calls Bought","Puts Sold"): return +1
        if order_type == "Puts Bought": return -1
        return 0
    used["Signed Dollars"] = used.apply(lambda r: sign_for(r["Order Type"]) * (r["Dollars"] or 0.0), axis=1)

    if view_mode == "Price Zones":
        strike_min = float(np.nanmin(used["Strike (Actual)"].values))
        strike_max = float(np.nanmax(used["Strike (Actual)"].values))
        if width_mode == "Auto":
            rng = max(1e-9, strike_max - strike_min)
            target_bucket = rng / 12.0
            steps = [1, 2, 5, 10, 25, 50, 100]
            zone_w = float(next((s for s in steps if s >= target_bucket), 100))
        else:
            zone_w = float(fixed_size_choice)
        
        n_dn = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w))
        n_up = int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
        lower_edge = spot - n_dn * zone_w
        upper_edge = spot + n_up * zone_w
        total = max(1, n_dn + n_up)
        
        def zone_index(x: float) -> int:
            if x <= lower_edge: return 0
            if x >= upper_edge: return total - 1
            return int(math.floor((x - lower_edge) / zone_w))
            
        used["ZoneIdx"] = used["Strike (Actual)"].apply(zone_index)
        zs_list = []
        for z in range(total):
            zl = lower_edge + z*zone_w
            zh = zl + zone_w
            zs_list.append((z, zl, zh, (zl+zh)/2.0))
        zone_df = pd.DataFrame(zs_list, columns=["ZoneIdx","Zone_Low","Zone_High","Zone_Center"])
        agg = used.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
        zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
        
        if hide_empty: zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
        
        st.subheader("Strike Zones")
        st.markdown('<div class="zones-panel">', unsafe_allow_html=True)
        above = zs[zs["Zone_Center"] > spot].sort_values("Zone_Center", ascending=False)
        below = zs[zs["Zone_Center"] < spot].sort_values("Zone_Center", ascending=False)
        max_abs = float(np.abs(zs["Net_Dollars"]).max()) if not zs.empty else 1.0
        
        for _, r in above.iterrows():
            color = "zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"
            w = max(6, int((abs(r['Net_Dollars'])/max_abs)*420))
            st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{r["Net_Dollars"]:,.0f} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
            
        st.markdown(f'<div class="price-divider"><div class="line"></div><div class="price-badge">SPOT: ${spot:,.2f}</div></div>', unsafe_allow_html=True)
        
        for _, r in below.iterrows():
            color = "zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"
            w = max(6, int((abs(r['Net_Dollars'])/max_abs)*420))
            st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{r["Net_Dollars"]:,.0f} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        e = used.copy()
        e["DTE"] = (pd.to_datetime(e["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days)
        bins = [0, 7, 30, 90, 180, 10000]
        labels = ["0-7d", "8-30d", "31-90d", "91-180d", ">180d"]
        e["Bucket"] = pd.cut(e["DTE"], bins=bins, labels=labels, include_lowest=True)
        agg = e.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
        st.subheader("Expiry Buckets")
        max_abs_exp = float(agg["Net_Dollars"].abs().max()) if not agg.empty else 1.0
        for _, r in agg.iterrows():
            color = "zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"
            w = max(6, int((abs(r['Net_Dollars'])/max_abs_exp)*420))
            st.markdown(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{r["Net_Dollars"]:,.0f} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)

    if show_table:
        st.subheader("Data Table")
        display_used = used.copy()
        display_used["Trade Date"] = display_used["Trade Date"].dt.strftime("%d %b %y")
        display_used["Expiry"] = pd.to_datetime(display_used["Expiry"]).dt.strftime("%d %b %y")
        
        st.dataframe(
            display_used, 
            use_container_width=True, 
            hide_index=True, 
            height=get_table_height(display_used, max_rows=30)
        )


def run_pivot_tables_app(df):
    """Analyzes exposure using Pivot Tables with specific Risk Reversal pairing logic"""
    st.title("🎯 Pivot Tables")
    
    yesterday = date.today() - timedelta(days=1)

    # ---------- Inputs ----------
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
        min_notional_label = st.selectbox("Min Dollars", options=list(notional_choices.keys()), index=1, key="pv_notional")
        min_notional = notional_choices[min_notional_label]
    with c5:
        mkt_cap_choices = {"0B": 0, "100B": 100e9, "200B": 200e9, "500B": 500e9, "1T": 1e12}
        min_mkt_cap_label = st.selectbox("Mkt Cap Min", options=list(mkt_cap_choices.keys()), index=1, key="pv_mkt_cap")
        min_mkt_cap = mkt_cap_choices[min_mkt_cap_label]
    with c6:
        ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 1. Identify Risk Reversals ---
    # We use all data in the date range to find pairs
    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    d_range['_original_idx'] = d_range.index
    
    cb_pool = d_range[d_range["Order Type"] == "Calls Bought"].copy()
    ps_pool = d_range[d_range["Order Type"] == "Puts Sold"].copy()
    
    # Matching logic: same date, symbol, expiry date, contracts
    # Use Expiry_DT for consistency against formatting differences in the CSV
    match_keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    
    cb_pool['occ'] = cb_pool.groupby(match_keys).cumcount()
    ps_pool['occ'] = ps_pool.groupby(match_keys).cumcount()
    
    rr_matches = pd.merge(
        cb_pool, ps_pool, 
        on=match_keys + ['occ'], 
        suffixes=('_c', '_p')
    )
    
    # Row IDs used in RR to exclude from solo tables
    used_cb_ids = rr_matches['_original_idx_c'].tolist()
    used_ps_ids = rr_matches['_original_idx_p'].tolist()
    
    # 3 Distinct Pools
    df_cb_solo = cb_pool[~cb_pool['_original_idx'].isin(used_cb_ids)].copy()
    df_ps_solo = ps_pool[~ps_pool['_original_idx'].isin(used_ps_ids)].copy()
    
    df_rr = pd.DataFrame()
    if not rr_matches.empty:
        df_rr['Symbol'] = rr_matches['Symbol']
        df_rr['Trade Date'] = rr_matches['Trade Date']
        df_rr['Expiry'] = rr_matches['Expiry_c'] # Use the string for grouping
        df_rr['Expiry_DT'] = rr_matches['Expiry_DT']
        df_rr['Contracts'] = rr_matches['Contracts']
        df_rr['Dollars'] = rr_matches['Dollars_c'] + rr_matches['Dollars_p']
        
        # Format strikes to remove .0 decimals
        s_c = rr_matches['Strike_c'].apply(clean_strike_fmt)
        s_p = rr_matches['Strike_p'].apply(clean_strike_fmt)
        df_rr['Strike'] = s_c + "c/" + s_p + "p"
        df_rr['Order Type'] = "Risk Reversal"

    def apply_filters(data):
        if data.empty: return data
        f_data = data.copy()
        if ticker_filter:
            f_data = f_data[f_data["Symbol"].astype(str).str.upper() == ticker_filter]
        f_data = f_data[f_data["Dollars"] >= min_notional]
        if not f_data.empty:
            unique_syms = f_data["Symbol"].unique()
            if min_mkt_cap > 0:
                valid_mc = [s for s in unique_syms if get_market_cap(s) >= min_mkt_cap]
                f_data = f_data[f_data["Symbol"].isin(valid_mc)]
                unique_syms = f_data["Symbol"].unique()
            if ema_filter == "Yes":
                valid_ema = [s for s in unique_syms if is_above_ema21(s)]
                f_data = f_data[f_data["Symbol"].isin(valid_ema)]
        return f_data

    df_cb_filtered = apply_filters(df_cb_solo)
    df_ps_filtered = apply_filters(df_ps_solo)
    df_rr_filtered = apply_filters(df_rr)

    def get_ranked_pivot(data):
        if data.empty: 
            return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
            
        sym_rank = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        # Ensure Expiry grouping handles different formatting by using standardized DT internally
        piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index()
        piv = piv.merge(sym_rank, on="Symbol")
        piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
        piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        
        piv["Symbol_Display"] = piv["Symbol"]
        piv.loc[piv["Symbol"] == piv["Symbol"].shift(1), "Symbol_Display"] = ""
        
        res = piv.drop(columns=["Symbol"]).rename(columns={
            "Symbol_Display": "Symbol", 
            "Expiry_Fmt": "Expiry_Table"
        })
        return res[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    # Side-by-side layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Calls Bought")
        tbl_cb = get_ranked_pivot(df_cb_filtered)
        if not tbl_cb.empty:
            st.dataframe(tbl_cb.style.format({"Dollars": "{:,.0f}", "Contracts": "{:,.0f}"}).map(highlight_expiry, subset=["Expiry_Table"]),
                         use_container_width=True, hide_index=True, height=get_table_height(tbl_cb),
                         column_config=COLUMN_CONFIG_PIVOT)
        else:
            st.info("None matching.")

    with col2:
        st.subheader("Puts Sold")
        tbl_ps = get_ranked_pivot(df_ps_filtered)
        if not tbl_ps.empty:
            st.dataframe(tbl_ps.style.format({"Dollars": "{:,.0f}", "Contracts": "{:,.0f}"}).map(highlight_expiry, subset=["Expiry_Table"]),
                         use_container_width=True, hide_index=True, height=get_table_height(tbl_ps),
                         column_config=COLUMN_CONFIG_PIVOT)
        else:
            st.info("None matching.")

    with col3:
        st.subheader("Risk Reversals")
        tbl_rr = get_ranked_pivot(df_rr_filtered)
        if not tbl_rr.empty:
            st.dataframe(tbl_rr.style.format({"Dollars": "{:,.0f}", "Contracts": "{:,.0f}"}).map(highlight_expiry, subset=["Expiry_Table"]),
                         use_container_width=True, hide_index=True, height=get_table_height(tbl_rr),
                         column_config=COLUMN_CONFIG_PIVOT)
        else:
            st.info("None matching.")


# --- 4. MAIN EXECUTION ---
if st.session_state["authentication_status"]:
    st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")
    
    st.markdown("""
    <style>
    :root{
      --bg:#1f1f22; --panel:#2a2d31; --panel2:#24272b; --text:#e7e7ea;
      --green:#71d28a; --red:#f29ca0; --line:#66b7ff; --ema8:#b689ff; --ema21:#ffb86b; --sma200:#ffffff; --price:#bfe7ff;
    }
    html,body,[class*="css"]{color:var(--text)!important;background-color:var(--bg)!important;}
    .block-container{padding-top:1.2rem;padding-bottom:1rem;}
    .control-box{padding:14px 0; border-radius:10px;}
    .zones-panel{padding:14px 0; border-radius:10px;}
    .zone-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
    .zone-label{width:220px;font-weight:700;color:#fff}
    .zone-bar{height:22px;border-radius:6px;min-width:6px}
    .zone-bull{background:linear-gradient(90deg,var(--green),#60c57b)}
    .zone-bear{background:linear-gradient(90deg,var(--red),#e4878d)}
    .zone-value{min-width:220px;font-variant-numeric:tabular-nums}
    .price-divider{position:relative;margin:16px 0 12px 0;text-align:center}
    .price-divider .line{height:2px;background:var(--line);opacity:.9}
    .price-badge{position:absolute;left:50%;transform:translate(-50%,-50%);top:0;background:#2b3a45;color:#bfe7ff;
      border:1px solid #56b6ff;border-radius:16px;padding:6px 12px;font-weight:800;font-size:12px;letter-spacing:.3px;
      box-shadow:0 2px 8px rgba(0,0,0,.35)}
    .metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
    .badge{background:#2b3a45;border:1px solid #3b5566;color:#cde8ff;border-radius:18px;padding:6px 10px;font-weight:700}
    .price-badge-header{background:#2b3a45;border:1px solid #56b6ff;color:#bfe7ff;border-radius:18px;padding:6px 10px;font-weight:800}
    th,td{border:1px solid #3a3f45;padding:8px} th{background:#343a40;text-align:left}
    [data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0px; }
    [data-testid="stSidebar"] .stCheckbox { margin-bottom: -10px; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Navigation")
        app_choice = st.selectbox("Select Tool", ["Strike Zones", "Pivot Tables"])
        st.markdown("---")
        authenticator.logout('Logout', 'sidebar')

    try:
        sheet_url = st.secrets["GSHEET_URL"]
        df_global = load_and_clean_data(sheet_url)
    except Exception as e:
        st.error(f"Error initializing data: {e}")
        st.stop()

    if app_choice == "Strike Zones":
        run_strike_zones_app(df_global)
    elif app_choice == "Pivot Tables":
        run_pivot_tables_app(df_global)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
