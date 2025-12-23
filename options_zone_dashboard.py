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

# Column widths for side-by-side tables
COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=65),
    "Strike": st.column_config.TextColumn("Strike", width=95),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=90),
    "Contracts": st.column_config.NumberColumn("Qty", width=60),
    "Dollars": st.column_config.NumberColumn("Dollars", width=110),
}

# --- 3. APP MODULES ---

def run_strike_zones_app(df):
    """Restored full Options Strike Zones logic with interactive inclusion logic"""
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
        
        st.markdown("---")
        if st.button("Reset All Defaults", use_container_width=True):
            keys_to_clear = ["sz_ticker", "sz_start", "sz_end", "sz_exp"]
            inc_key = f"sz_include_{ticker}"
            if inc_key in st.session_state: del st.session_state[inc_key]
            for k in keys_to_clear:
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    # 1. INITIAL FILTERING
    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start:
        f = f[f["Trade Date"].dt.date >= td_start]
    if td_end:
        f = f[f["Trade Date"].dt.date <= td_end]
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    
    # 2. PREPARE EDITABLE TABLE DATA
    edit_pool = f[f["Order Type"].isin(["Calls Bought","Puts Sold","Puts Bought"])].copy()
    if edit_pool.empty:
        st.warning("No trades match current filters.")
        return

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

    # 4. DATA TABLE EDITOR
    if show_table:
        st.subheader("Data Table")
        st.caption("Tip: Uncheck 'Included' to exclude a trade from calculations and charts above.")
        column_config = {
            "Trade Date Display": st.column_config.TextColumn("Trade Date"),
            "Order Type": st.column_config.TextColumn("Order Type"),
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Strike": st.column_config.TextColumn("Strike"),
            "Expiry Display": st.column_config.TextColumn("Expiry"),
            "Contracts": st.column_config.NumberColumn("Contracts", format="%d"),
            "Dollars": st.column_config.NumberColumn("Dollars", format="$%d"),
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
        min_notional = notional_choices[st.selectbox("Min Dollars", options=list(notional_choices.keys()), index=1, key="pv_notional")]
    with c5:
        mkt_cap_choices = {"0B": 0, "100B": 100e9, "200B": 200e9, "500B": 500e9, "1T": 1e12}
        min_mkt_cap = mkt_cap_choices[st.selectbox("Mkt Cap Min", options=list(mkt_cap_choices.keys()), index=1, key="pv_mkt_cap")]
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
    
    # Constructing RR data as split rows
    df_rr_rows = []
    if not rr_matches.empty:
        for idx, row in rr_matches.iterrows():
            # Call side row
            df_rr_rows.append({
                'Symbol': row['Symbol'],
                'Trade Date': row['Trade Date'],
                'Expiry_DT': row['Expiry_DT'],
                'Contracts': row['Contracts'],
                'Dollars': row['Dollars_c'],
                'Strike': f"{clean_strike_fmt(row['Strike_c'])} (C)",
                'Pair_ID': idx,
                'Pair_Side': 0 # Call is first
            })
            # Put side row
            df_rr_rows.append({
                'Symbol': row['Symbol'],
                'Trade Date': row['Trade Date'],
                'Expiry_DT': row['Expiry_DT'],
                'Contracts': row['Contracts'],
                'Dollars': row['Dollars_p'],
                'Strike': f"{clean_strike_fmt(row['Strike_p'])} (P)",
                'Pair_ID': idx,
                'Pair_Side': 1 # Put is second
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
            # For RR, we don't aggregate by Strike because we want to see both rows of the pair.
            # We just merge the rank and sort.
            piv = data.merge(sym_rank, on="Symbol")
            piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
            # Sort by total symbol volume, then individual Pair_ID to keep calls/puts together
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
            st.caption("⚠️ RR Table reflects date range only (ignores Ticker, Min Dollars, Mkt Cap, and EMA filters). Pairs are shown as Call row over Put row.")
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
    .price-badge{position:absolute;left:412px;transform:translate(-50%,-50%);top:0;background:#2b3a45;color:#bfe7ff;
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
        st.header("Navigation")
        app_choice = st.selectbox("Select Tool", ["Strike Zones", "Pivot Tables"])
        st.markdown("---")
        authenticator.logout('Logout', 'sidebar')
        
        if app_choice == "Pivot Tables":
            st.markdown('<div class="legend-title">Expiry Legend</div>', unsafe_allow_html=True)
            st.markdown('<div class="legend-item"><div class="color-dot" style="background:#2d5a27"></div> This Friday</div>', unsafe_allow_html=True)
            st.markdown('<div class="legend-item"><div class="color-dot" style="background:#8c5e03"></div> Next Friday</div>', unsafe_allow_html=True)
            st.markdown('<div class="legend-item"><div class="color-dot" style="background:#7d3c3c"></div> Two Fridays from now</div>', unsafe_allow_html=True)
            st.markdown("---")
            if st.button("Reset All Defaults", use_container_width=True, key="pv_reset"):
                for k in ["pv_start", "pv_end", "pv_ticker", "pv_notional", "pv_mkt_cap", "pv_ema_filter"]:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()
        
    try:
        sheet_url = st.secrets["GSHEET_URL"]
        df_global = load_and_clean_data(sheet_url)
        if app_choice == "Strike Zones": run_strike_zones_app(df_global)
        else: run_pivot_tables_app(df_global)
    except Exception as e:
        st.error(f"Error: {e}")
elif st.session_state["authentication_status"] is False: st.error('Incorrect password')
elif st.session_state["authentication_status"] is None: st.warning('Please login')
