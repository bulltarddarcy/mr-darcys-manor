import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from datetime import datetime

# --- Secrets & Path Configuration ---
def get_confirmed_gdrive_data(url):
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
    try:
        if "URL_CONFIG" not in st.secrets:
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

# --- Logic Constants ---
VOL_SMA_PERIOD = 30
DIVERGENCE_LOOKBACK = 90
SIGNAL_LOOKBACK_PERIOD = 25
RSI_DIFF_THRESHOLD = 2
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Analysis Pro", layout="wide")

st.markdown("""
    <style>
    table { width: 100%; border-collapse: collapse; margin-bottom: 2rem; font-family: monospace; }
    thead tr th { background-color: #f0f2f6 !important; color: #31333f !important; padding: 10px !important; border-bottom: 2px solid #dee2e6; text-align: left; }
    tbody tr td { padding: 8px !important; border-bottom: 1px solid #eee; }
    .analysis-card { background-color: #ffffff; border: 1px solid #e1e4e8; border-radius: 10px; padding: 25px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .pos-val { color: #27ae60; font-weight: bold; }
    .neg-val { color: #eb4d4b; font-weight: bold; }
    .tag-bubble { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 14px; font-weight: 600; margin: 2px 4px; color: white; white-space: nowrap; }
    .grey-note { color: #888888; font-size: 16px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- Shared Logic ---
def style_tags(tag_str):
    if not tag_str: return ''
    tags = tag_str.split(", ")
    html_str = ''
    colors = {f"EMA{EMA8_PERIOD}": "#4a90e2", f"EMA{EMA21_PERIOD}": "#9b59b6", "VOL_HIGH": "#e67e22", "V_GROW": "#27ae60"}
    for t in tags:
        color = colors.get(t, "#7f8c8d")
        html_str += f'<span class="tag-bubble" style="background-color: {color};">{t}</span>'
    return html_str

def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    w_close_col, w_vol_col, w_rsi_col = 'W_CLOSE', 'W_VOLUME', 'W_RSI_14'

    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low', d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'}, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])

    df_w = None
    if all(c in df.columns for c in [w_close_col, w_vol_col, w_rsi_col]):
        df_w = df[[w_close_col, w_vol_col, w_rsi_col]].copy()
        df_w.rename(columns={w_close_col: 'Price', w_vol_col: 'Volume', w_rsi_col: 'RSI'}, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    
    return df_d, df_w

def calculate_win_rates(df, current_rsi, tol=2):
    lower, upper = current_rsi - tol, current_rsi + tol
    matches = df[(df['RSI'] >= lower) & (df['RSI'] <= upper)].index
    windows = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
    results = []
    for w in windows:
        rets = []
        for idx in matches:
            pos = df.index.get_loc(idx)
            if pos + w < len(df):
                rets.append((df.iloc[pos + w]['Price'] - df.iloc[pos]['Price']) / df.iloc[pos]['Price'])
        if rets:
            results.append({"Days": w, "Win Rate": f"{(sum(1 for r in rets if r > 0) / len(rets)) * 100:.1f}%", "Avg": np.mean(rets) * 100, "Med": np.median(rets) * 100})
    return results, len(matches)

def find_divergences(df_tf, ticker, timeframe):
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                divergences.append({'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe, 'P1 Date': p1.name.strftime('%Y-%m-%d'), 'Signal Date': p2.name.strftime('%Y-%m-%d'), 'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}", 'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"})
    return divergences

# --- App Structure ---
st.title("📈 RSI Multi-Analysis Dashboard")
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

# Navigation via Tabs (Reliable visibility)
tab_div, tab_win = st.tabs(["📊 Divergence Scanner", "🎯 Forward Win Rates"])

if data_option:
    secret_key = dataset_map[data_option]
    target_url = st.secrets[secret_key]
    csv_buffer = get_confirmed_gdrive_data(target_url)

    if csv_buffer and csv_buffer != "HTML_ERROR":
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), 'TICKER')

        with tab_div:
            st.markdown('<div class="grey-note">ℹ️ Scans for Bullish RSI Divergences within the last 25 periods.</div>', unsafe_allow_html=True)
            all_divs = []
            for ticker, group in master.groupby(t_col):
                d_d, _ = prepare_data(group)
                if d_d is not None:
                    all_divs.extend(find_divergences(d_d, ticker, 'Daily'))
            if all_divs:
                st.table(pd.DataFrame(all_divs))
            else:
                st.write("No signals found.")

        with tab_win:
            st.header("Forward Returns Analysis (NFLX Test)")
            # TESTING LOCK: NFLX ONLY
            target_tickers = ["NFLX"]
            for ticker in target_tickers:
                t_df = master[master[t_col] == ticker].copy()
                df_d, _ = prepare_data(t_df)
                if df_d is not None:
                    curr_rsi = df_d['RSI'].iloc[-1]
                    stats, sample_size = calculate_win_rates(df_d, curr_rsi)
                    
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h2 style='margin-top:0;'>RSI Analysis: {ticker}</h2>
                        <p><b>Current RSI: {curr_rsi:.2f}</b> (as of {df_d.index[-1].strftime('%Y-%m-%d')})</p>
                        <p style="color: #666;">RSI Range: [{curr_rsi-2:.2f}, {curr_rsi+2:.2f}] | Samples: {sample_size}</p>
                    """, unsafe_allow_html=True)
                    
                    for title, data_slice in [("Short-Term Forward Returns", stats[:6]), ("Long-Term Forward Returns", stats[6:])]:
                        st.write(f"**{title}**")
                        tbl = "<table><thead><tr><th>Days</th><th>Win Rate</th><th>Avg Ret</th><th>Med Ret</th></tr></thead><tbody>"
                        for r in data_slice:
                            tbl += f"<tr><td>{r['Days']}</td><td>{r['Win Rate']}</td><td class='{'pos-val' if r['Avg']>0 else 'neg-val'}'>{r['Avg']:+.2f}%</td><td class='{'pos-val' if r['Med']>0 else 'neg-val'}'>{r['Med']:+.2f}%</td></tr>"
                        st.markdown(tbl + "</tbody></table>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
