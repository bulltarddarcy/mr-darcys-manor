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
        this_fri = today + timedelta(days=(4 - today.weekday()) % 7)
        next_fri = this_fri + timedelta(days=7)
        two_fri = this_fri + timedelta(days=14)
        
        if expiry_date < today: return "" 
        if expiry_date == this_fri: return "background-color: #2d5a27; color: white;" 
        elif expiry_date == next_fri: return "background-color: #8c5e03; color: white;" 
        elif expiry_date == two_fri: return "background-color: #7d3c3c; color: white;" 
        return ""
    except:
        return ""

def clean_strike_fmt(val):
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except:
        return str(val)

COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=65),
    "Strike": st.column_config.TextColumn("Strike", width=95),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=90),
    "Contracts": st.column_config.NumberColumn("Qty", width=60),
    "Dollars": st.column_config.NumberColumn("Dollars", width=110),
}

# --- 3. APP MODULES ---

def run_options_database_app(df):
    """Simple list view of all trade data with custom highlighting and top-level sorting"""
    st.title("📂 Options Database")

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        db_ticker = st.text_input("Ticker", value="", key="db_ticker").strip().upper()
    with c2:
        start_date = st.date_input("Trade Start Date", value=None, key="db_start")
    with c3:
        end_date = st.date_input("Trade End Date", value=None, key="db_end")
    with c4:
        exp_range_default = (date.today() + timedelta(days=365))
        db_exp_end = st.date_input("Expiration Range (end)", value=exp_range_default, key="db_exp")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("**Include Order Type**")
        inc_cb = st.checkbox("Calls Bought", value=True, key="db_inc_cb")
        inc_pb = st.checkbox("Puts Bought", value=True, key="db_inc_pb")
        inc_ps = st.checkbox("Puts Sold", value=True, key="db_inc_ps")

    f = df.copy()
    if db_ticker:
        f = f[f["Symbol"].astype(str).str.upper().eq(db_ticker)]
    if start_date:
        f = f[f["Trade Date"].dt.date >= start_date]
    if end_date:
        f = f[f["Trade Date"].dt.date <= end_date]
    if db_exp_end:
        f = f[f["Expiry_DT"].dt.date <= db_exp_end]

    allowed_types = []
    if inc_cb: allowed_types.append("Calls Bought")
    if inc_pb: allowed_types.append("Puts Bought")
    if inc_ps: allowed_types.append("Puts Sold")
    f = f[f["Order Type"].isin(allowed_types)]

    if f.empty:
        st.warning("No data found matching these filters.")
        return

    # Sort by most recent date then symbol A-Z
    f = f.sort_values(by=["Trade Date", "Symbol"], ascending=[False, True])

    display_cols = ["Trade Date", "Order Type", "Symbol", "Strike", "Expiry", "Contracts", "Dollars"]
    f_display = f[display_cols].copy()
    f_display["Trade Date"] = f_display["Trade Date"].dt.strftime("%d %b %y")
    f_display["Expiry"] = pd.to_datetime(f_display["Expiry"]).dt.strftime("%d %b %y")

    def highlight_db_order_type(val):
        if val in ["Calls Bought", "Puts Sold"]:
            return 'background-color: rgba(113, 210, 138, 0.15); color: #71d28a; font-weight: 600;'
        elif val == "Puts Bought":
            return 'background-color: rgba(242, 156, 160, 0.15); color: #f29ca0; font-weight: 600;'
        return ''

    st.subheader("Non-Expired Trades")
    st.caption("⚠️ User should check OI to confirm trades are still open")
    st.dataframe(
        f_display.style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"})
        .applymap(highlight_db_order_type, subset=["Order Type"]),
        use_container_width=True,
        hide_index=True,
        height=get_table_height(f_display, max_rows=30)
    )

def run_rankings_app(df):
    """Rank symbols based on trade volume and sentiment logic"""
    st.title("🏆 Rankings")

    # Formula is (Calls Bought) + (Puts Sold) - (Puts Bought)
    # Default range: 2 weeks before yesterday to yesterday
    yesterday = date.today() - timedelta(days=1)
    start_default = yesterday - timedelta(days=14)

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        rank_start = st.date_input("Trade Start Date", value=start_default, key="rank_start")
    with c2:
        rank_end = st.date_input("Trade End Date", value=yesterday, key="rank_end")
    st.markdown('</div>', unsafe_allow_html=True)

    f = df.copy()
    if rank_start:
        f = f[f["Trade Date"].dt.date >= rank_start]
    if rank_end:
        f = f[f["Trade Date"].dt.date <= rank_end]

    if f.empty:
        st.warning("No data found matching these dates.")
        return

    # Count occurrences per type per symbol
    # We pivot to get one row per symbol
    counts = f.groupby(["Symbol", "Order Type"]).size().unstack(fill_value=0)
    
    # Ensure all required columns exist
    for col in ["Calls Bought", "Puts Sold", "Puts Bought"]:
        if col not in counts.columns:
            counts[col] = 0

    # Calculate Score
    counts["Score"] = counts["Calls Bought"] + counts["Puts Sold"] - counts["Puts Bought"]
    
    # Sort and display
    res = counts[["Calls Bought", "Puts Sold", "Puts Bought", "Score"]].sort_values(by="Score", ascending=False).reset_index()
    
    st.subheader("Symbol Sentiment Rankings")
    st.caption("Score = (Calls Bought) + (Puts Sold) - (Puts Bought)")
    
    st.dataframe(
        res,
        use_container_width=True,
        hide_index=True,
        height=get_table_height(res, max_rows=30)
    )

def run_strike_zones_app(df):
    """Options Strike Zones with side-by-side charts and interactive inclusion logic"""
    st.title("📊 Options Strike Zones")

    exp_range_default = (date.today() + timedelta(days=365))

    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
    with c2:
        td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
    with c3:
        td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
    with c4:
        exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.sidebar:
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

    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start:
        f = f[f["Trade Date"].dt.date >= td_start]
    if td_end:
        f = f[f["Trade Date"].dt.date <= td_end]
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    
    # Sort pool for data table by most recent first
    edit_pool_raw = f[f["Order Type"].isin(["Calls Bought","Puts Sold","Puts Bought"])].copy()
    if edit_pool_raw.empty:
        st.warning("No trades match current filters.")
        return
        
    edit_pool = edit_pool_raw.sort_values(by="Trade Date", ascending=False).copy()
    edit_pool["Trade Date Display"] = edit_pool["Trade Date"].dt.strftime("%d %b %y")
    edit_pool["Expiry Display"] = edit_pool["Expiry_DT"].dt.strftime("%d %b %y")
    
    state_key = f"sz_include_{ticker}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [True] * len(edit_pool)
    if len(st.session_state[state_key]) != len(edit_pool):
        st.session_state[state_key] = [True] * len(edit_pool)

    edit_pool["Included"] = st.session_state[state_key]
    cols_to_show = ["Trade Date Display", "Order Type", "Symbol", "Strike", "Expiry Display", "Contracts", "Dollars", "Included"]
    
    active_mask = edit_pool["Included"] == True
    used = edit_pool[active_mask].copy()

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

    col_bars, col_chart = st.columns([1.5, 1])

    with col_bars:
        if used.empty:
            st.info("No trades currently included. Check 'Included' in table below.")
        else:
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
                    # Added $ to graphic values
                    st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">${r["Net_Dollars"]:,.0f} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="price-divider"><div class="line"></div><div class="price-badge">SPOT: ${spot:,.2f}</div></div>', unsafe_allow_html=True)
                for _, r in below.iterrows():
                    color = "zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"
                    w = max(6, int((abs(r['Net_Dollars'])/max_abs)*420))
                    st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">${r["Net_Dollars"]:,.0f} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
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
                    st.markdown(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">${r["Net_Dollars"]:,.0f} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)

    with col_chart:
        st.subheader("Price History (60d)")
        if history is not None:
            st.line_chart(history["Close"], use_container_width=True)
        else:
            st.info("No history available for chart.")

    if show_table:
        st.subheader("Data Table")
        st.caption("Tip: Uncheck 'Included' to exclude a trade from calculations and charts above.")
        column_config = {
            "Trade Date Display": st.column_config.TextColumn("Trade Date"),
            "Order Type": st.column_config.TextColumn("Order Type"),
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Strike": st.column_config.TextColumn("Strike"),
            "Expiry Display": st.column_config.TextColumn("Expiry"),
            "Contracts": st.column_config.NumberColumn("Contracts", format="%,d"), 
            "Dollars": st.column_config.NumberColumn("Dollars", format="$%,.0f"),   
            "Included": st.column_config.CheckboxColumn("Included", default=True)
        }
        edited_df = st.data_editor(
            edit_pool[cols_to_show],
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            key="strike_zones_editor"
        )
        if not edited_df.equals(edit_pool[cols_to_show]):
            st.session_state[state_key] = edited_df["Included"].tolist()
            st.rerun()


def run_pivot_tables_app(df):
    """Analyzes exposure using Pivot Tables with robust 1:1 Risk Reversal pairing and split row display"""
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
        choice_keys = list(notional_choices.keys())
        min_notional_label = st.selectbox("Min Dollars", options=choice_keys, index=1, key="pv_notional")
        min_notional = notional_choices[min_notional_label]
    with c5:
        mkt_cap_choices = {"0B": 0, "100B": 100e9, "200B": 200e9, "500B": 500e9, "1T": 1e12}
        min_mkt_cap_label = st.selectbox("Mkt Cap Min", options=list(mkt_cap_choices.keys()), index=1, key="pv_mkt_cap")
        min_mkt_cap = mkt_cap_choices[min_mkt_cap_label]
    with c6:
        ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- RR Pairing Engine ---
    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    d_range['_original_idx'] = d_range.index
    
    cb_pool = d_range[d_range["Order Type"] == "Calls Bought"].copy()
    ps_pool = d_range[d_range["Order Type"] == "Puts Sold"].copy()
    
    match_keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    cb_pool['occ'] = cb_pool.groupby(match_keys).cumcount()
    ps_pool['occ'] = ps_pool.groupby(match_keys).cumcount()
    
    rr_matches = pd.merge(cb_pool, ps_pool, on=match_keys + ['occ'], suffixes=('_c', '_p'))
    used_cb_ids = rr_matches['_original_idx_c'].tolist()
    used_ps_ids = rr_matches['_original_idx_p'].tolist()
    
    df_cb_solo = cb_pool[~cb_pool['_original_idx'].isin(used_cb_ids)].copy()
    df_ps_solo = ps_pool[~ps_pool['_original_idx'].isin(used_ps_ids)].copy()
    
    df_rr_rows = []
    if not rr_matches.empty:
        for idx, row in rr_matches.iterrows():
            df_rr_rows.append({
                'Symbol': row['Symbol'],
                'Trade Date': row['Trade Date'],
                'Expiry_DT': row['Expiry_DT'],
                'Contracts': row['Contracts'],
                'Dollars': row['Dollars_c'],
                'Strike': clean_strike_fmt(row['Strike_c']),
                'Pair_ID': idx,
                'Pair_Side': 0
            })
            df_rr_rows.append({
                'Symbol': row['Symbol'],
                'Trade Date': row['Trade Date'],
                'Expiry_DT': row['Expiry_DT'],
                'Contracts': row['Contracts'],
                'Dollars': row['Dollars_p'],
                'Strike': clean_strike_fmt(row['Strike_p']),
                'Pair_ID': idx,
                'Pair_Side': 1
            })
    df_rr = pd.DataFrame(df_rr_rows)

    def apply_filters(data, exclude_filters=False):
        if data.empty: return data
        f_data = data.copy()
        if ticker_filter: f_data = f_data[f_data["Symbol"].astype(str).str.upper() == ticker_filter]
        
        if not exclude_filters:
            f_data = f_data[f_data["Dollars"] >= min_notional]
            if not f_data.empty and min_mkt_cap > 0:
                unique_syms = f_data["Symbol"].unique()
                f_data = f_data[f_data["Symbol"].isin([s for s in unique_syms if get_market_cap(s) >= min_mkt_cap])]
            if not f_data.empty and ema_filter == "Yes":
                unique_syms = f_data["Symbol"].unique()
                f_data = f_data[f_data["Symbol"].isin([s for s in unique_syms if is_above_ema21(s)])]
        return f_data

    df_cb_f = apply_filters(df_cb_solo, exclude_filters=False)
    df_ps_f = apply_filters(df_ps_solo, exclude_filters=False)
    df_rr_f = apply_filters(df_rr, exclude_filters=True)

    def get_ranked_pivot(data, is_rr=False):
        if data.empty: return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
        sym_rank = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        if is_rr:
            piv = data.merge(sym_rank, on="Symbol")
            piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
            piv = piv.sort_values(by=["Total_Sym_Dollars", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
        else:
            piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index()
            piv = piv.merge(sym_rank, on="Symbol")
            piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
            piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
            
        piv["Symbol_Display"] = piv["Symbol"]
        piv.loc[piv["Symbol"] == piv["Symbol"].shift(1), "Symbol_Display"] = ""
        res = piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})
        return res[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    col1, col2, col3 = st.columns(3)
    fmt = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}

    with col1:
        st.subheader("Calls Bought")
        tbl = get_ranked_pivot(df_cb_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
        else: st.info("None.")
    with col2:
        st.subheader("Puts Sold")
        tbl = get_ranked_pivot(df_ps_f)
        if not tbl.empty: st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
        else: st.info("None.")
    with col3:
        st.subheader("Risk Reversals")
        tbl = get_ranked_pivot(df_rr_f, is_rr=True)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
            st.caption("⚠️ RR Table reflects date range only (ie, ignores all other inputs)")
        else: st.info("None.")

# --- 4. MAIN EXECUTION ---
if st.session_state["authentication_status"]:
    st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")
    st.markdown("""<style>:root{--bg:#1f1f22; --panel:#2a2d31; --panel2:#24272b; --text:#e7e7ea; --green:#71d28a; --red:#f29ca0; --line:#66b7ff; --ema8:#b689ff; --ema21:#ffb86b; --sma200:#ffffff; --price:#bfe7ff;}
    html,body,[class*="css"]{color:var(--text)!important;background-color:var(--bg)!important;}
    .block-container{padding-top:1.2rem;padding-bottom:1rem;}
    .control-box{padding:14px 0; border-radius:10px;}
    .zones-panel{padding:14px 0; border-radius:10px;}
    .zone-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
    .zone-label{width:100px;font-weight:700;color:#fff; text-align: right;}
    .zone-bar{height:22px;border-radius:6px;min-width:6px}
    .zone-bull{background:linear-gradient(90deg,var(--green),#60c57b)}
    .zone-bear{background:linear-gradient(90deg,var(--red),#e4878d)}
    .zone-value{min-width:220px;font-variant-numeric:tabular-nums}
    .price-divider{position:relative;margin:16px 0 12px 0;height:2px;}
    .price-divider .line{height:2px;background:var(--line);opacity:.9;width:652px;margin-left:112px;}
    .price-badge{position:absolute;left:438px;transform:translate(-50%,-50%);top:0;background:#2b3a45;color:#bfe7ff;
      border:1px solid #56b6ff;border-radius:16px;padding:6px 12px;font-weight:800;font-size:12px;letter-spacing:.3px;
      box-shadow:0 2px 8px rgba(0,0,0,.35); white-space: nowrap;}
    .metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
    .badge{background:#2b3a45;border:1px solid #3b5566;color:#cde8ff;border-radius:18px;padding:6px 10px;font-weight:700}
    .price-badge-header{background:#2b3a45;border:1px solid #56b6ff;color:#bfe7ff;border-radius:18px;padding:6px 10px;font-weight:800}
    th,td{border:1px solid #3a3f45;padding:8px} th{background:#343a40;text-align:left}
    .legend-title { font-size: 14px; font-weight: 700; margin-bottom: 12px; margin-top: 25px; color: var(--text); text-transform: uppercase; letter-spacing: 0.8px; opacity: 0.9; }
    .legend-item { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 14px; color: var(--text); }
    .color-dot { width: 14px; height: 14px; border-radius: 3px; }
    </style>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Select Tool")
        # Navigation order: Options Database -> Rankings -> Pivot Tables -> Strike Zones
        app_choice = st.selectbox("Select Tool", ["Options Database", "Rankings", "Pivot Tables", "Strike Zones"], label_visibility="collapsed")
        st.markdown("---")
        
    try:
        sheet_url = st.secrets["GSHEET_URL"]
        df_global = load_and_clean_data(sheet_url)
        
        # Tool logic
        if app_choice == "Options Database": 
            run_options_database_app(df_global)
        elif app_choice == "Rankings":
            run_rankings_app(df_global)
        elif app_choice == "Pivot Tables": 
            run_pivot_tables_app(df_global)
        else: # Strike Zones
            run_strike_zones_app(df_global)
            
        # Add Expiry Legend and Logout at the very end of the sidebar
        with st.sidebar:
            if app_choice == "Pivot Tables":
                st.markdown('<div class="legend-title">Expiry Legend</div>', unsafe_allow_html=True)
                st.markdown('<div class="legend-item"><div class="color-dot" style="background:#2d5a27"></div> This Friday</div>', unsafe_allow_html=True)
                st.markdown('<div class="legend-item"><div class="color-dot" style="background:#8c5e03"></div> Next Friday</div>', unsafe_allow_html=True)
                st.markdown('<div class="legend-item"><div class="color-dot" style="background:#7d3c3c"></div> Two Fridays</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            authenticator.logout('Logout', 'sidebar')
            
    except Exception as e:
        st.error(f"Error: {e}")
elif st.session_state["authentication_status"] is False: st.error('Incorrect password')
elif st.session_state["authentication_status"] is None: st.warning('Please login')
