# utils_prices.py
# This file contains constants, technical analysis functions, and scanners for the Price Divergences, RSI Scanner, Seasonality, and EMA Distance apps.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math
from datetime import date, timedelta
from io import BytesIO

# --- IMPORT SHARED UTILS ---
from utils_shared import get_gdrive_binary_data

# ==========================================
# CONSTANTS
# ==========================================
CACHE_TTL = 600
VOL_SMA_PERIOD = 30
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# ==========================================
# DATA LOADING (PARQUET)
# ==========================================
@st.cache_data(ttl=CACHE_TTL)
def get_parquet_config():
    config = {}
    try:
        raw = st.secrets.get("PARQUET_CONFIG", "")
        if raw:
            for line in raw.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2 and parts[1] in st.secrets: config[parts[0]] = parts[1]
    except: pass
    return config

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading Dataset...")
def load_parquet_and_clean(key):
    if key not in st.secrets: return None
    try:
        buf = get_gdrive_binary_data(st.secrets[key])
        if not buf: return None
        try: df = pd.read_parquet(buf)
        except: df = pd.read_csv(buf, engine='c')
        
        df.columns = [str(c).strip().upper() for c in df.columns]
        
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df.rename(columns={'DATE': 'ChartDate'}, inplace=True)
            df.sort_values('ChartDate', inplace=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: 'ChartDate'}, inplace=True)
            df.sort_values('ChartDate', inplace=True)
        else: return None

        if 'CLOSE' in df.columns: df['Price'] = df['CLOSE']
        return df
    except: return None

@st.cache_data(ttl=CACHE_TTL)
def load_ticker_map():
    try:
        url = st.secrets.get("URL_TICKER_MAP")
        if not url: return {}
        buf = get_gdrive_binary_data(url)
        if buf:
            df = pd.read_csv(buf, engine='c')
            if len(df.columns) >= 2:
                return dict(zip(df.iloc[:, 0].astype(str).str.strip().str.upper(), df.iloc[:, 1].astype(str).str.strip()))
    except: pass
    return {}

# ==========================================
# TECHNICAL ANALYSIS
# ==========================================
def add_technicals(df):
    if df is None or df.empty: return df
    cols = df.columns
    
    # Aliasing
    if 'RSI14' in cols and 'RSI' not in cols: df['RSI'] = df['RSI14']
    
    c_col = next((c for c in ['CLOSE', 'Close', 'Price'] if c in cols), None)
    if not c_col: return df
    prices = df[c_col]
    
    if 'RSI' not in df.columns:
        d = prices.diff()
        g = (d.where(d>0,0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        l = (-d.where(d<0,0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        df['RSI'] = 100 - (100 / (1 + g/l))
        
    if 'EMA8' not in cols and 'EMA_8' not in cols: df['EMA8'] = prices.ewm(span=8, adjust=False).mean()
    if 'EMA21' not in cols and 'EMA_21' not in cols: df['EMA21'] = prices.ewm(span=21, adjust=False).mean()
    if 'SMA200' not in cols and 'SMA_200' not in cols and len(df)>=200: df['SMA200'] = prices.rolling(200).mean()
    
    return df

@st.cache_data(ttl=CACHE_TTL)
def fetch_yahoo_data(ticker):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10y")
        if df.empty: return None
        df = df.reset_index()
        df.columns = [c.upper() for c in df.columns]
        if "DATE" in df.columns and df["DATE"].dt.tz is not None: df["DATE"] = df["DATE"].dt.tz_localize(None)
        return add_technicals(df)
    except: return None

@st.cache_data(ttl=CACHE_TTL)
def get_stock_indicators(sym):
    try:
        df = fetch_yahoo_data(sym)
        if df is None: return None, None, None, None, None
        
        last = df.iloc[-1]
        spot = last['CLOSE']
        e8 = last.get('EMA8', last.get('EMA_8'))
        e21 = last.get('EMA21', last.get('EMA_21'))
        s200 = last.get('SMA200', last.get('SMA_200'))
        return spot, e8, e21, s200, df
    except: return None, None, None, None, None

@st.cache_data(ttl=CACHE_TTL)
def get_ticker_technicals(ticker, mapping):
    if not mapping or ticker not in mapping: return None
    try:
        url = f"https://drive.google.com/uc?export=download&id={mapping[ticker]}"
        buf = get_gdrive_binary_data(url)
        if buf:
            df = pd.read_csv(buf, engine='c')
            df.columns = [c.strip().upper() for c in df.columns]
            if "DATE" not in df.columns: df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
            return add_technicals(df)
    except: pass
    return None

def parse_periods(s):
    try: return sorted(list(set([int(x.strip()) for x in s.split(',') if x.strip().isdigit()])))
    except: return [5, 21, 63, 126]

# ==========================================
# DIVERGENCE & RSI LOGIC
# ==========================================
def prepare_data(df):
    df.columns = [c.strip().upper() for c in df.columns]
    if 'DATE' in df.columns: 
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE').sort_index()
    elif 'CHARTDATE' in df.columns:
        df['CHARTDATE'] = pd.to_datetime(df['CHARTDATE'])
        df = df.set_index('CHARTDATE').sort_index()

    # Daily
    d_map = {'CLOSE':'Price', 'VOLUME':'Volume', 'HIGH':'High', 'LOW':'Low', 'OPEN':'Open',
             'RSI':'RSI', 'RSI14':'RSI', 'EMA8':'EMA8', 'EMA21':'EMA21'}
    d_cols = [c for c in d_map.keys() if c in df.columns]
    df_d = df[d_cols].rename(columns=d_map).copy()
    if 'Volume' in df_d.columns: df_d['VolSMA'] = df_d['Volume'].rolling(VOL_SMA_PERIOD).mean()
    df_d = add_technicals(df_d)
    
    if 'Price' not in df_d.columns or 'RSI' not in df_d.columns: return None, None
    df_d.dropna(subset=['Price','RSI'], inplace=True)

    # Weekly
    w_map = {'W_CLOSE':'Price', 'W_HIGH':'High', 'W_LOW':'Low', 'W_VOLUME':'Volume', 'W_RSI14':'RSI', 'W_EMA8':'EMA8'}
    w_cols = [c for c in w_map.keys() if c in df.columns]
    if not w_cols: return df_d, None
    
    df_w = df[w_cols].rename(columns=w_map).copy()
    df_w['ChartDate'] = df_w.index - pd.to_timedelta(df_w.index.dayofweek, unit='D')
    df_w = df_w.groupby('ChartDate').last().sort_index()
    df_w['ChartDate'] = df_w.index
    
    if 'Volume' in df_w.columns: df_w['VolSMA'] = df_w['Volume'].rolling(VOL_SMA_PERIOD).mean()
    df_w = add_technicals(df_w)
    
    if 'Price' in df_w.columns and 'RSI' in df_w.columns: df_w.dropna(subset=['Price','RSI'], inplace=True)
    else: return df_d, None
    
    return df_d, df_w

def calculate_optimal_signal_stats(hist_indices, prices, curr_idx, signal_type='Bullish', timeframe='Daily', periods_input=None, optimize_for='PF'):
    hist = np.array(hist_indices); valid = hist[hist < curr_idx]
    if len(valid) == 0: return None
    
    pers = np.array(periods_input or [5, 21, 63, 126])
    n = len(prices)
    exits = valid[:, None] + pers[None, :]
    mask = exits < n
    safe_exits = np.clip(exits, 0, n-1)
    
    entries = prices[valid]
    exit_ps = prices[safe_exits]
    
    rets = (exit_ps - entries[:, None]) / entries[:, None]
    if signal_type == 'Bearish': rets = -rets
    
    best_s = -999.0; best = None
    unit = 'w' if timeframe.lower() == 'weekly' else 'd'
    
    for i, p in enumerate(pers):
        p_rets = rets[mask[:, i], i]
        if len(p_rets) == 0: continue
        
        wins = p_rets[p_rets > 0]
        gl = np.abs(np.sum(p_rets[p_rets < 0]))
        pf = (np.sum(wins)/gl if gl > 0 else 999.0) if np.sum(wins) > 0 else 0.0
        
        wr = (len(wins)/len(p_rets))*100
        ev = np.mean(p_rets)*100
        std = np.std(p_rets)
        sqn = (np.mean(p_rets)/std)*np.sqrt(len(p_rets)) if std > 0 else 0.0
        
        score = pf if optimize_for == 'PF' else sqn
        if score > best_s:
            best_s = score
            best = {"Best Period": f"{p}{unit}", "Profit Factor": pf, "Win Rate": wr, "EV": ev, "N": len(p_rets), "SQN": sqn}
            
    return best

def find_divergences(df_tf, ticker, timeframe, min_n=0, periods_input=None, optimize_for='PF', lookback_period=90, price_source='High/Low', strict_validation=True, recent_days_filter=25, rsi_diff_threshold=2.0):
    divs = []
    if len(df_tf) < lookback_period+1: return divs
    
    rsi = df_tf['RSI'].values; close = df_tf['Price'].values
    vol = df_tf['Volume'].values; vol_sma = df_tf['VolSMA'].values
    
    if price_source == 'Close': low = high = close
    else: low = df_tf['Low'].values; high = df_tf['High'].values
    
    roll_low = pd.Series(low).shift(1).rolling(lookback_period).min().values
    roll_high = pd.Series(high).shift(1).rolling(lookback_period).max().values
    
    new_low = low < roll_low; new_high = high > roll_high
    
    for i in range(lookback_period, len(df_tf)):
        if not (new_low[i] or new_high[i]): continue
        
        lb_start = i - lookback_period
        lb_rsi = rsi[lb_start:i]
        p2_rsi = rsi[i]
        
        vol_hi = int(vol[i] > (vol_sma[i]*1.5)) if not np.isnan(vol_sma[i]) else 0
        
        if new_low[i]: # Bullish
            p1_rel = np.argmin(lb_rsi)
            if p2_rsi > (lb_rsi[p1_rel] + rsi_diff_threshold):
                p1_abs = lb_start + p1_rel
                if strict_validation and np.any(rsi[p1_abs:i+1] > 50): continue
                divs.append({"index": i, "type": "Bullish", "p1_idx": p1_abs, "vol_high": vol_hi})
                
        elif new_high[i]: # Bearish
            p1_rel = np.argmax(lb_rsi)
            if p2_rsi < (lb_rsi[p1_rel] - rsi_diff_threshold):
                p1_abs = lb_start + p1_rel
                if strict_validation and np.any(rsi[p1_abs:i+1] < 50): continue
                divs.append({"index": i, "type": "Bearish", "p1_idx": p1_abs, "vol_high": vol_hi})
                
    results = []
    disp_thresh = len(df_tf) - recent_days_filter
    
    bull_idxs = [d['index'] for d in divs if d['type'] == 'Bullish']
    bear_idxs = [d['index'] for d in divs if d['type'] == 'Bearish']
    
    for d in divs:
        i, t, p1 = d['index'], d['type'], d['p1_idx']
        
        hist = bull_idxs if t == 'Bullish' else bear_idxs
        best = calculate_optimal_signal_stats(hist, close, i, signal_type=t, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if not best or best['N'] < min_n: continue
        
        def get_dt(ix):
            if timeframe == 'Weekly': return df_tf.iloc[ix]['ChartDate'].strftime('%Y-%m-%d')
            return df_tf.index[ix].strftime('%Y-%m-%d')
            
        row = {
            'Ticker': ticker, 'Type': t, 'Timeframe': timeframe,
            'Signal_Date_ISO': get_dt(i), 'P1_Date_ISO': get_dt(p1),
            'RSI1': rsi[p1], 'RSI2': rsi[i], 
            'Price1': low[p1] if t=='Bullish' else high[p1],
            'Price2': low[i] if t=='Bullish' else high[i],
            'Is_Recent': i >= disp_thresh
        }
        
        # Tags
        tags = []
        if d['vol_high']: tags.append("V_HI")
        last = df_tf.iloc[-1]
        e8 = last.get('EMA8'); e21 = last.get('EMA21'); lp = last['Price']
        if e8 and ((t=='Bullish' and lp > e8) or (t=='Bearish' and lp < e8)): tags.append(f"EMA{EMA8_PERIOD}")
        if e21 and ((t=='Bullish' and lp > e21) or (t=='Bearish' and lp < e21)): tags.append(f"EMA{EMA21_PERIOD}")
        
        row['Tags'] = tags
        
        # Display logic
        p1_d_str = df_tf.index[p1].strftime('%b %d') if timeframe!='Weekly' else df_tf.iloc[p1]['ChartDate'].strftime('%b %d')
        p2_d_str = df_tf.index[i].strftime('%b %d') if timeframe!='Weekly' else df_tf.iloc[i]['ChartDate'].strftime('%b %d')
        
        row['Date_Display'] = f"{p1_d_str} → {p2_d_str}"
        row['RSI_Display'] = f"{rsi[p1]:.0f} {'↗' if rsi[i]>rsi[p1] else '↘'} {rsi[i]:.0f}"
        row['Price_Display'] = f"${row['Price1']:,.2f} {'↘' if row['Price2']<row['Price1'] else '↗'} ${row['Price2']:,.2f}"
        row['Last_Close'] = f"${lp:,.2f}"
        
        # Calculate Forward Returns if needed for export
        if periods_input:
            for p in periods_input:
                if i+p < len(close):
                    ret = (close[i+p] - close[i])/close[i]
                    if t == 'Bearish': ret = -ret
                    row[f'Ret_{p}'] = ret * 100
        
        results.append(row)
        
    return results

def find_rsi_percentile_signals(df, ticker, pct_low=0.1, pct_high=0.9, min_n=1, timeframe='Daily', periods_input=None, optimize_for='SQN'):
    if len(df) < 200: return []
    
    rsi = df['RSI'].values; price = df['Price'].values
    p10 = np.quantile(rsi, pct_low); p90 = np.quantile(rsi, pct_high)
    prev = np.roll(rsi, 1); prev[0] = rsi[0]
    
    bull = (prev < p10) & (rsi >= (p10 + 1.0))
    bear = (prev > p90) & (rsi <= (p90 - 1.0))
    
    b_idxs = np.where(bull)[0]; s_idxs = np.where(bear)[0]
    all_idxs = np.sort(np.concatenate([b_idxs, s_idxs]))
    
    sigs = []
    latest = df['Price'].iloc[-1]
    
    # Analyze recent history (last 10 years only for signal generation to save time, but use full hist for stats)
    # Actually, iterate all for simplicity, but optimized
    
    for i in all_idxs:
        is_bull = i in b_idxs
        t = 'Bullish' if is_bull else 'Bearish'
        hist = b_idxs if is_bull else s_idxs
        
        best = calculate_optimal_signal_stats(hist, price, i, signal_type=t, timeframe=timeframe, periods_input=periods_input, optimize_for=optimize_for)
        
        if best and best['N'] >= min_n:
            ev_p = price[i] * (1 + best['EV']/100) if is_bull else price[i] * (1 - best['EV']/100)
            
            sigs.append({
                'Ticker': ticker, 'Date': df.index[i].strftime('%b %d'), 'Date_Obj': df.index[i].date(),
                'Action': "Leaving Low" if is_bull else "Leaving High",
                'RSI_Display': f"{p10 if is_bull else p90:.0f} {'↗' if is_bull else '↘'} {rsi[i]:.0f}",
                'Signal_Price': f"${price[i]:,.2f}", 'Last_Close': f"${latest:,.2f}", 'Signal': t,
                'Profit Factor': best['Profit Factor'], 'Win Rate': best['Win Rate'], 'EV': best['EV'],
                'EV Target': ev_p, 'SQN': best['SQN']
            })
            
    return sigs

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_history_optimized(sym, mapping):
    return get_ticker_technicals(sym, mapping) or fetch_yahoo_data(sym)

@st.cache_data(ttl=CACHE_TTL)
def is_above_ema21(sym):
    try:
        t = yf.Ticker(sym); h = t.history(period="60d")
        if len(h)<21: return True
        return h["Close"].iloc[-1] > h["Close"].ewm(span=21).mean().iloc[-1]
    except: return True
