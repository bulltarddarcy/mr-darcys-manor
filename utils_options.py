# utils_options.py
# Contains all constants, data loaders, and helper functions required for the Database, Rankings, Pivot Tables, and Strike Zones apps. 
# Includes necessary technical helpers (like Yahoo fetching) locally to keep this module self-contained.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT SHARED UTILS ---
from utils_shared import get_gdrive_binary_data

# ==========================================
# CONSTANTS
# ==========================================
CACHE_TTL = 600
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# Database
DB_DEFAULT_EXPIRY_OFFSET = 365
DB_TABLE_MAX_ROWS = 30
DB_DATE_FMT = "%d %b %y"
STYLE_BULL_CSS = 'background-color: rgba(113, 210, 138, 0.15); color: #71d28a; font-weight: 600;'
STYLE_BEAR_CSS = 'background-color: rgba(242, 156, 160, 0.15); color: #f29ca0; font-weight: 600;'

# Rankings
RANK_LOOKBACK_DAYS = 14
RANK_LIMIT_DEFAULT = 20
RANK_MC_THRESHOLDS = {"0B": 0, "2B": 2e9, "10B": 1e10, "50B": 5e10, "100B": 1e11}
RANK_SM_WEIGHTS = {'Sentiment': 0.35, 'Impact': 0.30, 'Momentum': 0.35}
RANK_CONVICTION_DIVISOR = 25.0
RANK_TOP_IDEAS_COUNT = 3

# Pivot Tables
PIVOT_NOTIONAL_MAP = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}
PIVOT_MC_MAP = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}
PIVOT_TABLE_FMT = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}
COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=None),
    "Strike": st.column_config.TextColumn("Strike", width=None),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=None),
    "Contracts": st.column_config.NumberColumn("Qty", width=None),
    "Dollars": st.column_config.NumberColumn("Dollars", width=None),
}

# Strike Zones
SZ_DEFAULT_EXP_OFFSET = 365
SZ_DEFAULT_FIXED_SIZE = 10
SZ_AUTO_WIDTH_DENOM = 12.0
SZ_AUTO_STEPS = [1, 2, 5, 10, 25, 50, 100]
SZ_BUCKET_BINS = [0, 7, 30, 60, 90, 120, 180, 365, 10000]
SZ_BUCKET_LABELS = ["0-7d", "8-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-365d", ">365d"]

# ==========================================
# DATA LOADING (GLOBAL)
# ==========================================
@st.cache_data(ttl=CACHE_TTL, show_spinner="Updating Data...")
def load_and_clean_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, engine='c')
        
        want = {"Trade Date", "Order Type", "Symbol", "Strike (Actual)", "Strike", "Expiry", "Contracts", "Dollars", "Error"}
        existing_cols = [c for c in df.columns if c in want]
        df = df[existing_cols]
        
        for c in ["Order Type", "Symbol", "Strike", "Expiry"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        
        if "Dollars" in df.columns and df["Dollars"].dtype == 'object':
             df["Dollars"] = pd.to_numeric(df["Dollars"].str.replace(r'[$,]', '', regex=True), errors="coerce").fillna(0.0)

        if "Contracts" in df.columns and df["Contracts"].dtype == 'object':
             df["Contracts"] = pd.to_numeric(df["Contracts"].str.replace(',', '', regex=False), errors="coerce").fillna(0)
        
        if "Trade Date" in df.columns:
            df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")
        
        if "Expiry" in df.columns:
            df["Expiry_DT"] = pd.to_datetime(df["Expiry"], errors="coerce")
            
        if "Strike (Actual)" in df.columns:
            df["Strike (Actual)"] = pd.to_numeric(df["Strike (Actual)"], errors="coerce").fillna(0.0)
            
        if "Error" in df.columns:
            mask = df["Error"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
            df = df[~mask]
            
        return df
    except Exception as e:
        st.error(f"Error loading global data: {e}")
        return pd.DataFrame()

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid = df["Trade Date"].dropna()
        if not valid.empty: return valid.max().date()
    return date.today() - timedelta(days=1)

# ==========================================
# TECHNICAL HELPERS (LOCAL)
# ==========================================
# Included locally so Options apps can run without 'utils_prices' dependency
def _add_technicals(df):
    if df is None or df.empty: return df
    cols = df.columns
    
    close_col = next((c for c in ['CLOSE', 'Close', 'Price'] if c in cols), None)
    if not close_col: return df
    close_series = df[close_col]

    if 'RSI' not in cols and 'RSI14' not in cols:
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    if 'EMA8' not in cols: df['EMA8'] = close_series.ewm(span=8, adjust=False).mean()
    if 'EMA21' not in cols: df['EMA21'] = close_series.ewm(span=21, adjust=False).mean()
    if 'SMA200' not in cols and len(df) >= 200: df['SMA200'] = close_series.rolling(window=200).mean()
    return df

@st.cache_data(ttl=CACHE_TTL)
def _fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="2y")
        if df.empty: return None
        df = df.reset_index()
        df.columns = [c.upper() for c in df.columns]
        if "DATE" in df.columns and df["DATE"].dt.tz is not None:
            df["DATE"] = df["DATE"].dt.tz_localize(None)
        return _add_technicals(df)
    except: return None

@st.cache_data(ttl=CACHE_TTL)
def _get_stock_indicators(sym: str):
    try:
        df = _fetch_yahoo_data(sym)
        if df is None or df.empty: return None, None, None, None, None
        
        recent = df.iloc[-1]
        spot = recent['CLOSE']
        ema8 = recent.get('EMA8')
        ema21 = recent.get('EMA21')
        sma200 = recent.get('SMA200')
        return spot, ema8, ema21, sma200, df
    except: return None, None, None, None, None

def fetch_technicals_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(_get_stock_indicators, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            try: results[future_to_ticker[future]] = future.result()
            except: results[future_to_ticker[future]] = (None, None, None, None, None)
    return results

@st.cache_data(ttl=43200)
def get_market_cap(symbol: str) -> float:
    try:
        t = yf.Ticker(symbol)
        fi = t.fast_info
        mc = fi.get('marketCap')
        if mc: return float(mc)
    except: pass
    return 0.0

def fetch_market_caps_batch(tickers):
    results = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_ticker = {executor.submit(get_market_cap, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            results[future_to_ticker[future]] = future.result()
    return results

# ==========================================
# DATABASE APP UTILS
# ==========================================
def initialize_database_state(max_date):
    defaults = {
        'saved_db_ticker': "",
        'saved_db_start': max_date,
        'saved_db_end': max_date,
        'saved_db_exp': (date.today() + timedelta(days=DB_DEFAULT_EXPIRY_OFFSET)),
        'saved_db_inc_cb': True, 'saved_db_inc_ps': True, 'saved_db_inc_pb': True
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def _highlight_db_order_type(val):
    if val in ["Calls Bought", "Puts Sold"]: return STYLE_BULL_CSS
    elif val == "Puts Bought": return STYLE_BEAR_CSS
    return ''

def get_database_styled_view(df):
    if df.empty: return df
    ot_col = "Order Type" if "Order Type" in df.columns else "Order type"
    disp = df[["Trade Date", ot_col, "Symbol", "Strike", "Expiry", "Contracts", "Dollars"]].copy()
    
    if not pd.api.types.is_string_dtype(disp["Trade Date"]):
        disp["Trade Date"] = disp["Trade Date"].dt.strftime(DB_DATE_FMT)
    try: disp["Expiry"] = pd.to_datetime(disp["Expiry"]).dt.strftime(DB_DATE_FMT)
    except: pass
        
    return disp.style.format({"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}).map(_highlight_db_order_type, subset=[ot_col])

def filter_database_trades(df, ticker, start_date, end_date, exp_end, inc_cb, inc_ps, inc_pb):
    if df.empty: return pd.DataFrame()
    f = df.copy()
    if ticker: f = f[f["Symbol"].astype(str).str.upper().eq(ticker)]
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    if exp_end: f = f[f["Expiry_DT"].dt.date <= exp_end]
    
    ot_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed = []
    if inc_cb: allowed.append("Calls Bought")
    if inc_ps: allowed.append("Puts Sold")
    if inc_pb: allowed.append("Puts Bought")
    
    return f[f[ot_col].isin(allowed)].sort_values(by=["Trade Date", "Symbol"], ascending=[False, True])

# ==========================================
# RANKINGS APP UTILS
# ==========================================
def initialize_rankings_state(start_default, max_date):
    defaults = {'saved_rank_start': start_default, 'saved_rank_end': max_date, 
                'saved_rank_limit': RANK_LIMIT_DEFAULT, 'saved_rank_mc': "10B", 'saved_rank_ema': False}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def filter_rankings_data(df, start_date, end_date):
    if df.empty: return pd.DataFrame()
    f = df.copy()
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    ot_col = "Order Type" if "Order Type" in f.columns else "Order type"
    return f[f[ot_col].isin(["Calls Bought", "Puts Sold", "Puts Bought"])]

def calculate_smart_money_score(df, start_d, end_d, mc_thresh, filter_ema, limit):
    f = df.copy()
    if start_d: f = f[f["Trade Date"].dt.date >= start_d]
    if end_d: f = f[f["Trade Date"].dt.date <= end_d]
    
    ot_col = "Order Type" if "Order Type" in f.columns else "Order type"
    f = f[f[ot_col].isin(["Calls Bought", "Puts Sold", "Puts Bought"])].copy()
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    f["Signed_Dollars"] = np.where(f[ot_col].isin(["Calls Bought", "Puts Sold"]), f["Dollars"], -f["Dollars"])
    stats = f.groupby("Symbol").agg(Signed_Dollars=("Signed_Dollars", "sum"), Trade_Count=("Symbol", "count"), Last_Trade=("Trade Date", "max")).reset_index()
    
    tickers = stats["Symbol"].unique().tolist()
    stats["Market Cap"] = stats["Symbol"].map(fetch_market_caps_batch(tickers))
    valid = stats[stats["Market Cap"] >= mc_thresh].copy()
    
    # Momentum (Last 3 days of data)
    u_dates = sorted(f["Trade Date"].unique())
    rec_dates = u_dates[-3:] if len(u_dates) >= 3 else u_dates
    mom = f[f["Trade Date"].isin(rec_dates)].groupby("Symbol")["Signed_Dollars"].sum().reset_index().rename(columns={"Signed_Dollars": "Momentum"})
    
    valid = valid.merge(mom, on="Symbol", how="left").fillna(0)
    
    if valid.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def norm(s): return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else 0
    
    valid["Impact"] = valid["Signed_Dollars"] / valid["Market Cap"]
    
    # Scores
    w = RANK_SM_WEIGHTS
    valid["Base_Bull"] = (w['Sentiment'] * norm(valid["Signed_Dollars"].clip(lower=0))) + (w['Impact'] * norm(valid["Impact"].clip(lower=0))) + (w['Momentum'] * norm(valid["Momentum"].clip(lower=0)))
    valid["Base_Bear"] = (w['Sentiment'] * norm(-valid["Signed_Dollars"].clip(upper=0))) + (w['Impact'] * norm(-valid["Impact"].clip(upper=0))) + (w['Momentum'] * norm(-valid["Momentum"].clip(upper=0)))
    
    valid["Last Trade"] = valid["Last_Trade"].dt.strftime("%d %b")
    
    bulls = valid.sort_values("Base_Bull", ascending=False).head(limit*3)
    bears = valid.sort_values("Base_Bear", ascending=False).head(limit*3)
    
    batch_techs = fetch_technicals_batch(set(bulls["Symbol"]).union(set(bears["Symbol"]))) if filter_ema else {}

    def apply_filter(d, mode="Bull"):
        d["Score"] = d[f"Base_{mode}"] * 100
        if not filter_ema: return d.head(limit)
        
        filtered = []
        for _, r in d.iterrows():
            s, e8, _, _, _ = batch_techs.get(r["Symbol"], (None, None, None, None, None))
            if s and e8:
                if (mode == "Bull" and s > e8) or (mode == "Bear" and s < e8): filtered.append(r)
            if len(filtered) >= limit: break
        return pd.DataFrame(filtered)

    return apply_filter(bulls, "Bull"), apply_filter(bears, "Bear"), valid

def find_whale_confluence(ticker, global_df, current_price, order_type_filter=None):
    if global_df.empty: return None
    f = global_df[global_df["Symbol"].astype(str).str.upper() == ticker].copy()
    if f.empty: return None
    f = f[f["Expiry_DT"] > pd.Timestamp.now()]
    if order_type_filter: f = f[f["Order Type"] == order_type_filter]
    else: f = f[f["Order Type"].isin(["Puts Sold", "Calls Bought"])]
    
    if f.empty: return None
    f.sort_values("Dollars", ascending=False, inplace=True)
    whale = f.iloc[0]
    
    # Prefer OTM puts if selling
    if whale["Order Type"] == "Puts Sold" and whale["Strike (Actual)"] > current_price:
        otm = f[(f["Order Type"]=="Puts Sold") & (f["Strike (Actual)"] < current_price)]
        if not otm.empty: whale = otm.iloc[0]
        
    return {"Strike": whale["Strike (Actual)"], "Expiry": whale["Expiry_DT"].strftime("%d %b"), "Dollars": whale["Dollars"], "Type": whale["Order Type"]}

def analyze_trade_setup(ticker, t_df, global_df):
    score = 0; reasons = []; suggs = {'Buy Calls': None, 'Sell Puts': None, 'Buy Commons': None}
    if t_df is None or t_df.empty: return 0, ["No data"], suggs
    
    last = t_df.iloc[-1]
    close, ema8, ema21, sma200 = last['CLOSE'], last.get('EMA8',0), last.get('EMA21',0), last.get('SMA200',0)
    rsi = last.get('RSI', 50)
    
    if close > ema8 and close > ema21: score += 2; reasons.append("Strong Trend (> EMA8/21)")
    elif close > ema21: score += 1; reasons.append("Moderate Trend (> EMA21)")
    if close > sma200: score += 2; reasons.append("Bullish (> SMA200)")
    if 45 < rsi < 65: score += 2; reasons.append(f"Healthy RSI ({rsi:.0f})")
    elif rsi >= 70: score -= 1; reasons.append("Overbought")
    
    opt_days = 30 # Simple default here to avoid importing prices utils
    
    pw = find_whale_confluence(ticker, global_df, close, "Puts Sold")
    cw = find_whale_confluence(ticker, global_df, close, "Calls Bought")
    
    sp_k = pw["Strike"] if pw and pw["Strike"] < close else math.floor(ema21)
    suggs['Sell Puts'] = f"Strike ${sp_k}"
    
    bc_k = cw["Strike"] if cw else math.ceil(close)
    if close > ema8 or cw: suggs['Buy Calls'] = f"Strike ${bc_k}"
    
    suggs['Buy Commons'] = f"Entry: ${close:.2f}"
    if pw: reasons.append(f"Whale Puts @ ${pw['Strike']}")
    if cw: reasons.append(f"Whale Calls @ ${cw['Strike']}")
    
    return score, reasons, suggs

def generate_top_ideas(top_bulls, global_df):
    if top_bulls.empty: return []
    res = []
    batch = fetch_technicals_batch(top_bulls["Symbol"].tolist())
    
    for _, row in top_bulls.iterrows():
        t = row["Symbol"]
        t_data = batch.get(t)
        t_df = t_data[4] if t_data else None
        
        tech_s, reasons, suggs = analyze_trade_setup(t, t_df, global_df)
        final_s = (row["Score"] / RANK_CONVICTION_DIVISOR) + tech_s
        price = t_df.iloc[-1]['CLOSE'] if t_df is not None else 0
        
        res.append({"Ticker": t, "Score": final_s, "Price": price, "Reasons": reasons, "Suggestions": suggs})
        
    return sorted(res, key=lambda x: x['Score'], reverse=True)[:RANK_TOP_IDEAS_COUNT]

def calculate_volume_rankings(f_filtered, mc_thresh, filter_ema, limit):
    if f_filtered.empty: return pd.DataFrame(), pd.DataFrame()
    ot_col = "Order Type" if "Order Type" in f_filtered.columns else "Order type"
    
    cnt = f_filtered.groupby(["Symbol", ot_col]).size().unstack(fill_value=0)
    for c in ["Calls Bought", "Puts Sold", "Puts Bought"]: 
        if c not in cnt.columns: cnt[c] = 0
        
    scores = pd.DataFrame(index=cnt.index)
    scores["Score"] = cnt["Calls Bought"] + cnt["Puts Sold"] - cnt["Puts Bought"]
    scores["Trade Count"] = cnt.sum(axis=1)
    scores["Last Trade"] = f_filtered.groupby("Symbol")["Trade Date"].max().dt.strftime(DB_DATE_FMT)
    
    res = scores.reset_index()
    res["Market Cap"] = res["Symbol"].map(fetch_market_caps_batch(res["Symbol"].tolist()))
    res = res[res["Market Cap"] >= mc_thresh]
    
    def _ema_filt(df, mode):
        if not filter_ema: return df.head(limit)
        out = []
        batch = fetch_technicals_batch(df.head(limit*3)["Symbol"].tolist())
        for _, r in df.iterrows():
            s, e8, _, _, _ = batch.get(r["Symbol"], (None, None, None, None, None))
            if s and e8:
                if (mode=="Bull" and s > e8) or (mode=="Bear" and s < e8): out.append(r)
            if len(out) >= limit: break
        return pd.DataFrame(out)

    bull = _ema_filt(res.sort_values(["Score", "Trade Count"], ascending=[False, False]), "Bull")
    bear = _ema_filt(res.sort_values(["Score", "Trade Count"], ascending=[True, False]), "Bear")
    return bull, bear

# ==========================================
# PIVOT APP UTILS
# ==========================================
def initialize_pivot_state(start_default, max_date):
    defaults = {'saved_pv_start': max_date, 'saved_pv_end': max_date, 'saved_pv_ticker': "", 
                'saved_pv_notional': "0M", 'saved_pv_mkt_cap': "0B", 'saved_pv_ema': "All",
                'saved_calc_strike': 100.0, 'saved_calc_premium': 2.50, 'saved_calc_expiry': date.today()+timedelta(days=30)}
    for k, v in defaults.items(): 
        if k not in st.session_state: st.session_state[k] = v

def generate_pivot_pools(d_range):
    ot = "Order Type" if "Order Type" in d_range.columns else "Order type"
    cb = d_range[d_range[ot] == "Calls Bought"].copy()
    ps = d_range[d_range[ot] == "Puts Sold"].copy()
    pb = d_range[d_range[ot] == "Puts Bought"].copy()
    
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    cb['occ'] = cb.groupby(keys).cumcount(); ps['occ'] = ps.groupby(keys).cumcount()
    matches = pd.merge(cb, ps, on=keys+['occ'], suffixes=('_c', '_p'))
    
    if not matches.empty:
        rr_c = matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_c', 'Strike_c']].rename(columns={'Dollars_c':'Dollars','Strike_c':'Strike'})
        rr_c['Pair_ID'] = matches.index; rr_c['Pair_Side'] = 0
        rr_p = matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_p', 'Strike_p']].rename(columns={'Dollars_p':'Dollars','Strike_p':'Strike'})
        rr_p['Pair_ID'] = matches.index; rr_p['Pair_Side'] = 1
        rr = pd.concat([rr_c, rr_p])
        
        def filter_out(pool, m):
            m = m[keys+['occ']].copy(); m['_rm'] = True
            merged = pool.merge(m, on=keys+['occ'], how='left')
            return merged[merged['_rm'].isna()].drop(columns=['_rm'])
        
        cb = filter_out(cb, matches)
        ps = filter_out(ps, matches)
    else:
        rr = pd.DataFrame(columns=['Symbol','Dollars','Strike','Pair_ID','Pair_Side'])
        
    return cb, ps, pb, rr

def filter_pivot_dataframe(data, ticker, min_not, min_mc, ema):
    if data.empty: return data
    f = data[data["Dollars"] >= min_not].copy()
    if ticker: f = f[f["Symbol"].astype(str).str.upper() == ticker]
    
    if not f.empty and (min_mc > 0 or ema == "Yes"):
        syms = set(f["Symbol"].unique())
        if min_mc > 0: syms = {s for s in syms if get_market_cap(s) >= float(min_mc)}
        if ema == "Yes":
            batch = fetch_technicals_batch(list(syms))
            syms = {s for s in syms if batch.get(s,(None,None,None))[2] is None or (batch[s][0] and batch[s][2] and batch[s][0] > batch[s][2])}
        f = f[f["Symbol"].isin(syms)]
    return f

def get_expiry_color_map():
    try:
        t = date.today()
        fri = t + timedelta(days=(4 - t.weekday()) % 7)
        return {fri.strftime(DB_DATE_FMT): "background-color: #b7e1cd; color: black;",
                (fri + timedelta(7)).strftime(DB_DATE_FMT): "background-color: #fce8b2; color: black;",
                (fri + timedelta(14)).strftime(DB_DATE_FMT): "background-color: #f4c7c3; color: black;"}
    except: return {}

def highlight_expiry(val): return get_expiry_color_map().get(val, "")

def get_pivot_styled_view(data, is_rr=False):
    if data.empty: return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
    sr = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym")
    
    if is_rr: piv = data.merge(sr, on="Symbol").sort_values(["Total_Sym", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
    else:
        piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts":"sum","Dollars":"sum"}).reset_index().merge(sr, on="Symbol")
        piv = piv.sort_values(["Total_Sym", "Dollars"], ascending=[False, False])
        
    piv["Expiry_Table"] = piv["Expiry_DT"].dt.strftime(DB_DATE_FMT)
    piv["Symbol"] = np.where(piv["Symbol"] == piv["Symbol"].shift(1), "", piv["Symbol"])
    return piv[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

# ==========================================
# STRIKE ZONES APP UTILS
# ==========================================
def initialize_strike_zone_state(exp_default):
    defaults = {'saved_sz_ticker': "AMZN", 'saved_sz_start': None, 'saved_sz_end': None, 'saved_sz_exp': exp_default,
                'saved_sz_view': "Price Zones", 'saved_sz_width_mode': "Auto", 'saved_sz_fixed': SZ_DEFAULT_FIXED_SIZE,
                'saved_sz_inc_cb': True, 'saved_sz_inc_ps': True, 'saved_sz_inc_pb': True}
    for k, v in defaults.items(): 
        if k not in st.session_state: st.session_state[k] = v

def filter_strike_zone_data(df, ticker, start, end, exp_end, cb, ps, pb):
    f = df[df["Symbol"].astype(str).str.upper() == ticker].copy()
    if f.empty: return f
    if start: f = f[f["Trade Date"].dt.date >= start]
    if end: f = f[f["Trade Date"].dt.date <= end]
    f = f[(f["Expiry_DT"].dt.date >= date.today()) & (f["Expiry_DT"].dt.date <= exp_end)]
    
    ot = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed = []
    if cb: allowed.append("Calls Bought")
    if ps: allowed.append("Puts Sold")
    if pb: allowed.append("Puts Bought")
    
    f = f[f[ot].isin(allowed)].copy()
    if not f.empty: f.insert(0, "Include", True)
    return f

def get_strike_zone_technicals(ticker):
    # Returns (Spot, Ema8, Ema21, Sma200)
    res = _get_stock_indicators(ticker)
    return (res[0] or 100.0, res[1], res[2], res[3])

def generate_price_zones_html(df, spot, width_mode, fixed_size):
    f = df.copy()
    ot = "Order Type" if "Order Type" in f.columns else "Order type"
    f["Signed"] = np.where(f[ot].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0)
    
    vals = f["Strike (Actual)"].values
    mn, mx = float(np.nanmin(vals)), float(np.nanmax(vals))
    
    if width_mode == "Auto":
        raw = max(1e-9, mx - mn) / SZ_AUTO_WIDTH_DENOM
        w = float(next((s for s in SZ_AUTO_STEPS if s >= raw), 100))
    else: w = float(fixed_size)
    
    n_dn = int(math.ceil(max(0, spot - mn) / w))
    n_up = int(math.ceil(max(0, mx - spot) / w))
    low_edge = spot - n_dn * w
    total = max(1, n_dn + n_up)
    
    f["Zone"] = np.clip(np.floor((f["Strike (Actual)"] - low_edge)/w).astype(int), 0, total-1)
    agg = f.groupby("Zone").agg(Net=("Signed","sum"), Count=("Signed","count")).reset_index()
    
    html = ['<div class="zones-panel">']
    max_v = max(1.0, agg["Net"].abs().max())
    
    # Helper to generate rows
    def gen_rows(subset):
        for _, r in subset.iterrows():
            z_low, z_high = low_edge + r.Zone*w, low_edge + (r.Zone+1)*w
            c = "zone-bull" if r.Net >= 0 else "zone-bear"
            pct = (abs(r.Net)/max_v)*100
            txt = f"(${abs(r.Net):,.0f})" if r.Net < 0 else f"${r.Net:,.0f}"
            html.append(f'<div class="zone-row"><div class="zone-label">${z_low:.0f}-${z_high:.0f}</div><div class="zone-wrapper"><div class="zone-bar {c}" style="width:{pct:.1f}%"></div><div class="zone-value">{txt} | n={int(r.Count)}</div></div></div>')
            
    zs = agg.merge(pd.DataFrame({"Zone":range(total)}), on="Zone", how="right").fillna(0).sort_values("Zone", ascending=False)
    gen_rows(zs[(low_edge + zs.Zone*w + w/2) > spot])
    html.append(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>')
    gen_rows(zs[(low_edge + zs.Zone*w + w/2) <= spot])
    html.append('</div>')
    
    return "".join(html)

def generate_expiry_buckets_html(df):
    f = df.copy()
    ot = "Order Type" if "Order Type" in f.columns else "Order type"
    f["Signed"] = np.where(f[ot].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0)
    
    days = (pd.to_datetime(f["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days)
    f["B"] = pd.cut(days, bins=SZ_BUCKET_BINS, labels=SZ_BUCKET_LABELS)
    agg = f.groupby("B").agg(Net=("Signed","sum"), Count=("Signed","count")).reset_index()
    
    html = []
    max_v = max(1.0, agg["Net"].abs().max())
    for _, r in agg.iterrows():
        c = "zone-bull" if r.Net >= 0 else "zone-bear"
        pct = (abs(r.Net)/max_v)*100
        txt = f"(${abs(r.Net):,.0f})" if r.Net < 0 else f"${r.Net:,.0f}"
        html.append(f'<div class="zone-row"><div class="zone-label">{r.B}</div><div class="zone-wrapper"><div class="zone-bar {c}" style="width:{pct:.1f}%"></div><div class="zone-value">{txt} | n={int(r.Count)}</div></div></div>')
        
    return "".join(html)
