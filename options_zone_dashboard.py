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

# --- 2. GLOBAL DATA LOADING ---
@st.cache_data(show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    want = ["Trade Date","Order Type","Symbol","Strike (Actual)","Strike","Expiry","Contracts","Dollars","Error"]
    keep = [c for c in want if c in df.columns]
    df = df[keep].copy()
    
    if "Dollars" in df.columns:
        df["Dollars"] = (df["Dollars"].astype(str)
                         .str.replace("$", "", regex=False)
                         .str.replace(",", "", regex=False))
        df["Dollars"] = pd.to_numeric(df["Dollars"], errors="coerce").fillna(0.0)
    
    if "Trade Date" in df.columns:
        df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
    
    if "Expiry" in df.columns:
        df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
        
    if "Strike (Actual)" in df.columns:
        df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
        
    if "Error" in df.columns:
        df = df[~df["Error"].astype(str).str.upper().isin(["TRUE","1","YES"])]
        
    return df

# --- 3. APP MODULES ---

def run_strike_zones_app(df):
    """Logic for the original Strike Zones Dashboard"""
    st.title("📊 Options Strike Zones Dashboard")

    yesterday = date.today() - timedelta(days=1)

    # ---------- Controls ----------
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
    with c2:
        td_start = st.date_input("Trade Date (start)", value=yesterday, key="sz_start", format="DD MMM YY")
    with c3:
        td_end = st.date_input("Trade Date (end)", value=yesterday, key="sz_end", format="DD MMM YY")
    with c4:
        exp_range_end = date.today().replace(year=date.today().year+1)
        exp_end = st.date_input("Exp. Range (end)", value=exp_range_end, key="sz_exp", format="DD MMM YY")
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

    # ---------- Filters ----------
    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    f = f[(f["Trade Date"].dt.date >= td_start) & (f["Trade Date"].dt.date <= td_end)]
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

    # ---------- indicators ----------
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

    # ---------- Visualizations ----------
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
        st.markdown("### Data Table")
        display_used = used.copy()
        display_used["Trade Date"] = display_used["Trade Date"].dt.strftime("%d %b %y")
        display_used["Expiry"] = pd.to_datetime(display_used["Expiry"]).dt.strftime("%d %b %y")
        st.dataframe(display_used, use_container_width=True)


def run_pivot_tables_app(df):
    """Analyzes exposure using Pivot Tables across defined order types"""
    st.title("💼 Pivot Tables")
    
    yesterday = date.today() - timedelta(days=1)

    # ---------- Inputs ----------
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        td_start = st.date_input("Trade Start Date", value=yesterday, key="pv_start", format="DD MMM YY")
    with c2:
        td_end = st.date_input("Trade End Date", value=yesterday, key="pv_end", format="DD MMM YY")
    with c3:
        ticker_filter = st.text_input("Ticker (leave blank for all)", value="", key="pv_ticker").strip().upper()
    with c4:
        notional_choices = {"0M": 0, "5M": 5_000_000, "10M": 10_000_000, "50M": 50_000_000, "100M": 100_000_000}
        min_notional_label = st.selectbox("Min Notional", options=list(notional_choices.keys()), index=1, key="pv_notional")
        min_notional = notional_choices[min_notional_label]
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Global Filtering ----------
    f = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    
    if ticker_filter:
        f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
    
    f = f[f["Dollars"] >= min_notional]

    def get_ranked_pivot(data, order_type, columns):
        subset = data[data["Order Type"] == order_type].copy()
        if subset.empty:
            return pd.DataFrame(columns=columns)
        
        # 1. aggregate at Symbol level to find 'Total Rank'
        sym_rank = subset.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        
        # 2. Perform main aggregation
        piv = subset.groupby(["Symbol", "Strike", "Expiry"]).agg({
            "Contracts": "sum",
            "Dollars": "sum"
        }).reset_index()
        
        # 3. Join rank data to main pivot
        piv = piv.merge(sym_rank, on="Symbol")
        
        # 4. Standardize types and format Expiry
        piv["Contracts"] = pd.to_numeric(piv["Contracts"], errors='coerce').fillna(0)
        piv["Dollars"] = pd.to_numeric(piv["Dollars"], errors='coerce').fillna(0.0)
        piv["Expiry"] = pd.to_datetime(piv["Expiry"]).dt.strftime("%d %b %y")
        
        # 5. SORT: By Total Symbol Vol (Desc), then by Strike Dollars (Desc)
        piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        
        # 6. CLEAR REDUNDANT SYMBOLS (Leave rows below the first empty)
        # Use shift() to check if current symbol is same as previous in the sorted order
        piv["Symbol_Display"] = piv["Symbol"]
        piv.loc[piv["Symbol"] == piv["Symbol"].shift(1), "Symbol_Display"] = ""
        
        piv = piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol"})
        return piv[columns]

    std_cols = ["Symbol", "Strike", "Expiry", "Contracts", "Dollars"]

    # 1. Calls Bought Table
    st.subheader("Calls Bought")
    calls_bought = get_ranked_pivot(f, "Calls Bought", std_cols)
    if not calls_bought.empty:
        st.dataframe(calls_bought.style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}), use_container_width=True)
    else:
        st.info("No Calls Bought found matching these filters.")

    # 2. Puts Sold Table
    st.subheader("Puts Sold")
    puts_sold = get_ranked_pivot(f, "Puts Sold", std_cols)
    if not puts_sold.empty:
        st.dataframe(puts_sold.style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}), use_container_width=True)
    else:
        st.info("No Puts Sold found matching these filters.")

    # 3. Risk Reversals Table
    st.subheader("Risk Reversals")
    rr_data = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    rr_data = rr_data[rr_data["Order Type"] == "Risk Reversals"]
    
    if not rr_data.empty:
        sym_rank_rr = rr_data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        rr_pivot = rr_data.groupby(["Symbol", "Order Type", "Strike", "Expiry"]).agg({
            "Contracts": "sum",
            "Dollars": "sum"
        }).reset_index()
        
        rr_pivot = rr_pivot.merge(sym_rank_rr, on="Symbol")
        rr_pivot["Contracts"] = pd.to_numeric(rr_pivot["Contracts"], errors='coerce').fillna(0)
        rr_pivot["Dollars"] = pd.to_numeric(rr_pivot["Dollars"], errors='coerce').fillna(0.0)
        rr_pivot["Expiry"] = pd.to_datetime(rr_pivot["Expiry"]).dt.strftime("%d %b %y")
        
        # Sort
        rr_pivot = rr_pivot.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        
        # Clear redundant symbols for RR table
        rr_pivot["Symbol_Display"] = rr_pivot["Symbol"]
        rr_pivot.loc[rr_pivot["Symbol"] == rr_pivot["Symbol"].shift(1), "Symbol_Display"] = ""
        rr_pivot = rr_pivot.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol"})
        
        rr_cols = ["Symbol", "Order Type", "Strike", "Expiry", "Contracts", "Dollars"]
        st.dataframe(rr_pivot[rr_cols].style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}), use_container_width=True)
    else:
        st.info("No Risk Reversals found in this date range.")


# --- 4. MAIN EXECUTION ---
if st.session_state["authentication_status"]:
    st.set_page_config(page_title="Trading Toolbox", layout="wide")
    
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
