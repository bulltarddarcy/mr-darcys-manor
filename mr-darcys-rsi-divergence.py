import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from datetime import datetime, timedelta

# --- Secrets & Path Configuration ---
def get_confirmed_gdrive_data(url):
    """Bypasses the 'File too large to scan' warning page using a token handshake."""
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
    """Reads the TXT file from Drive and returns a dictionary {Name: SecretKey}"""
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
EV_LOOKBACK_YEARS = 10
MIN_N_THRESHOLD = 5

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Divergence Scanner", layout="wide")

st.markdown("""
    <style>
    .top-note {
        color: #888888;
        font-size: 14px;
        margin-bottom: 2px;
        font-family: inherit;
    }
    
    table { width: 100%; border-collapse: collapse; table-layout: fixed; margin-bottom: 2rem; }
    thead tr th { background-color: #f0f2f6 !important; color: #31333f !important; padding: 12px !important; border-bottom: 2px solid #dee2e6; }
    
    th:nth-child(1) { width: 8%; }  
    th:nth-child(2) { width: 22%; } 
    th:nth-child(3) { width: 10%; }  
    th:nth-child(4) { width: 10%; }  
    th:nth-child(5) { width: 8%; }  
    th:nth-child(6) { width: 10%; }  
    th:nth-child(7) { width: 10%; }  
    th:nth-child(8) { width: 11%; } 
    th:nth-child(9) { width: 11%; } 
    
    tbody tr td { padding: 10px !important; border-bottom: 1px solid #eee; word-wrap: break-word; font-size: 14px; }
    .align-left { text-align: left !important; }
    .align-center { text-align: center !important; }
    
    .ev-positive { background-color: #e6f4ea !important; color: #1e7e34; font-weight: 500; }
    .ev-negative { background-color: #fce8e6 !important; color: #c5221f; font-weight: 500; }
    .ev-neutral { color: #5f6368; }

    .tag-bubble { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin: 2px 4px 2px 0; color: white; white-space: nowrap; }
    
    .footer-header { color: #31333f; margin-top: 1.5rem; border-bottom: 1px solid #ddd; padding-bottom: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 RSI Divergence Scanner")

# --- Helpers ---
def style_tags(tag_str):
    if not tag_str: return ''
    tags = tag_str.split(", ")
    colors = {f"EMA{EMA8_PERIOD}": "#4a90e2", f"EMA{EMA21_PERIOD}": "#9b59b6", "VOL_HIGH": "#e67e22", "V_GROW": "#27ae60"}
    html_str = ''
    for t in tags:
        color = colors.get(t, "#7f8c8d")
        html_str += f'<span class="tag-bubble" style="background-color: {color};">{t}</span>'
    return html_str

def calculate_ev_data(df, target_rsi, periods, current_price):
    if df.empty or pd.isna(target_rsi): return None
    cutoff_date = df.index.max() - timedelta(days=365 * EV_LOOKBACK_YEARS)
    hist_df = df[df.index >= cutoff_date].copy()
    
    # Matching Logic: Within +/- 2 of the target RSI
    mask = (hist_df['RSI'] >= target_rsi - 2) & (hist_df['RSI'] <= target_rsi + 2)
    indices = np.where(mask)[0]
    
    returns = []
    for idx in indices:
        # Check forward window availability
        if idx + periods < len(hist_df):
            entry_p = hist_df.iloc[idx]['Price']
            exit_p = hist_df.iloc[idx + periods]['Price']
            if entry_p > 0: 
                returns.append((exit_p - entry_p) / entry_p)
                
    if not returns or len(returns) < MIN_N_THRESHOLD: return None
    
    avg_ret = np.mean(returns)
    ev_price = current_price * (1 + avg_ret)
    return {"price": ev_price, "n": len(returns), "return": avg_ret}

def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    d_rsi_col, d_ema8_col, d_ema21_col = 'RSI_14', 'EMA_8', 'EMA_21'
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low', d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'}, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])
    
    # Process weekly data if available in columns starting with 'W_'
    w_close_col = 'W_CLOSE'
    if w_close_col in df.columns:
        df_w = df[[w_close_col, 'W_VOLUME', 'W_HIGH', 'W_LOW', 'W_RSI_14', 'W_EMA_8', 'W_EMA_21']].copy()
        df_w.rename(columns={w_close_col: 'Price', 'W_VOLUME': 'Volume', 'W_HIGH': 'High', 'W_LOW': 'Low', 'W_RSI_14': 'RSI', 'W_EMA_8': 'EMA8', 'W_EMA_21': 'EMA21'}, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else: df_w = None
    
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences
    latest_p = df_tf.iloc[-1]
    
    # Calculate EV based on the most recent RSI (e.g., 12/26/2025)
    ev30 = calculate_ev_data(df_tf, latest_p['RSI'], 30, latest_p['Price'])
    ev90 = calculate_ev_data(df_tf, latest_p['RSI'], 90, latest_p['Price'])

    def get_date_str(p): return df_tf.loc[p.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else p.name.strftime('%Y-%m-%d')
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        for s_type in ['Bullish', 'Bearish']:
            trigger = False
            if s_type == 'Bullish' and p2['Low'] < lookback['Low'].min():
                p1 = lookback.loc[lookback['RSI'].idxmin()]
                if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD) and not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any(): trigger = True
            elif s_type == 'Bearish' and p2['High'] > lookback['High'].max():
                p1 = lookback.loc[lookback['RSI'].idxmax()]
                if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD) and not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any(): trigger = True
            
            if trigger:
                post_df = df_tf.iloc[i + 1 :]
                valid = True
                if s_type == 'Bullish' and not post_df.empty and (post_df['RSI'] <= p1['RSI']).any(): valid = False
                if s_type == 'Bearish' and not post_df.empty and (post_df['RSI'] >= p1['RSI']).any(): valid = False
                
                if valid:
                    tags = []
                    # Filter for tags based on price relationship to EMAs
                    if s_type == 'Bullish':
                        if latest_p['Price'] >= latest_p.get('EMA8', 0): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] >= latest_p.get('EMA21', 0): tags.append(f"EMA{EMA21_PERIOD}")
                    else:
                        if latest_p['Price'] <= latest_p.get('EMA8', 999999): tags.append(f"EMA{EMA8_PERIOD}")
                        if latest_p['Price'] <= latest_p.get('EMA21', 999999): tags.append(f"EMA{EMA21_PERIOD}")
                    if is_vol_high: tags.append("VOL_HIGH")
                    if p2['Volume'] > p1['Volume']: tags.append("V_GROW")
                    
                    divergences.append({
                        'Ticker': ticker, 'Type': s_type, 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                        'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                        'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                        'P1 Price': f"${p1['Low' if s_type=='Bullish' else 'High']:,.2f}", 
                        'P2 Price': f"${p2['Low' if s_type=='Bullish' else 'High']:,.2f}",
                        'ev30_raw': ev30, 'ev90_raw': ev90
                    })
    return divergences

# --- App Logic ---
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

if data_option:
    try:
        target_url = st.secrets[dataset_map[data_option]]
        csv_buffer = get_confirmed_gdrive_data(target_url)
        if csv_buffer and csv_buffer != "HTML_ERROR":
            master = pd.read_csv(csv_buffer)
            date_col = next((col for col in master.columns if 'DATE' in col.upper()), None)
            last_updated_str = pd.to_datetime(master[date_col]).max().strftime('%Y-%m-%d') if date_col else "Unknown"
            
            st.markdown('<div class="top-note">ℹ️ See bottom of page for strategy logic and tag explanations.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="top-note">📅 Last Updated: {last_updated_str}</div>', unsafe_allow_html=True)
            
            t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
            all_tickers = sorted(master[t_col].unique())
            with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                sq = st.text_input("Filter...").upper()
                ft = [t for t in all_tickers if sq in t]
                cols = st.columns(6)
                for i, ticker in enumerate(ft): cols[i % 6].write(ticker)
                
            raw_results = []
            progress_bar = st.progress(0, text="Scanning...")
            grouped = master.groupby(t_col)
            for i, (ticker, group) in enumerate(grouped):
                d_d, d_w = prepare_data(group.copy())
                if d_d is not None: raw_results.extend(find_divergences(d_d, ticker, 'Daily'))
                if d_w is not None: raw_results.extend(find_divergences(d_w, ticker, 'Weekly'))
                progress_bar.progress((i + 1) / len(grouped))
                
            if raw_results:
                res_df = pd.DataFrame(raw_results).sort_values(by='Signal Date', ascending=False)
                consolidated = res_df.groupby(['Ticker', 'Type', 'Timeframe']).head(1)
                for tf in ['Daily', 'Weekly']:
                    st.divider()
                    st.header(f"📅 {tf} Divergence Analysis")
                    for s_type, emoji in [('Bullish', '🟢'), ('Bearish', '🔴')]:
                        st.subheader(f"{emoji} {s_type} Signals")
                        tbl_df = consolidated[(consolidated['Type']==s_type) & (consolidated['Timeframe']==tf)].copy()
                        if not tbl_df.empty:
                            html = '<table><thead><tr><th>Ticker</th><th>Tags</th><th>P1 Date</th><th>Signal Date</th><th>RSI</th><th>P1 Price</th><th>P2 Price</th><th>EV 30p</th><th>EV 90p</th></tr></thead><tbody>'
                            for _, row in tbl_df.iterrows():
                                html += '<tr>'
                                html += f'<td class="align-left"><b>{row["Ticker"]}</b></td>'
                                html += f'<td class="align-left">{style_tags(row["Tags"])}</td>'
                                html += f'<td class="align-center">{row["P1 Date"]}</td>'
                                html += f'<td class="align-center">{row["Signal Date"]}</td>'
                                html += f'<td class="align-center">{row["RSI"]}</td>'
                                html += f'<td class="align-left">{row["P1 Price"]}</td>'
                                html += f'<td class="align-left">{row["P2 Price"]}</td>'
                                for ev_key in ['ev30_raw', 'ev90_raw']:
                                    data = row[ev_key]
                                    if data:
                                        is_pos = data['return'] > 0
                                        cls = ("ev-positive" if is_pos else "ev-negative") if s_type == 'Bullish' else ("ev-positive" if not is_pos else "ev-negative")
                                        html += f'<td class="{cls}">{data["return"]*100:+.1f}% <br><small>(${data["price"]:,.2f}, N={data["n"]})</small></td>'
                                    else: html += '<td class="ev-neutral">N/A</td>'
                                html += '</tr>'
                            html += '</tbody></table>'
                            st.markdown(html, unsafe_allow_html=True)
                        else: st.write("No signals.")
            else: st.warning("No signals.")
            
            # --- Robust Footer ---
            st.divider()
            f_col1, f_col2, f_col3 = st.columns(3)
            
            with f_col1:
                st.markdown('<div class="footer-header">📉 SIGNAL LOGIC</div>', unsafe_allow_html=True)
                st.markdown(f"""
                * **Signal Identification**: Scans for price extremes within a **{SIGNAL_LOOKBACK_PERIOD}-period window**.
                * **Divergence Mechanism**: Compares the current RSI at a new extreme to a previous extreme within the **{DIVERGENCE_LOOKBACK}-period lookback**.
                * **Standards**: RSI must not cross the 50 midline between divergence points.
                """)

            with f_col2:
                st.markdown('<div class="footer-header">🔮 EXPECTED VALUE (EV) ANALYSIS</div>', unsafe_allow_html=True)
                st.markdown(f"""
                * **Data Pool**: Analyzes the last **{EV_LOOKBACK_YEARS} years** of historical data.
                * **RSI Matching**: Identifies every day where the RSI was within **±2 points** of the most recent RSI (e.g., 12/26/2025).
                * **Forward Projection**: Calculates the **Average (Mean)** return for those instances exactly 30 and 90 trading days later.
                * **Statistical Filter**: EV displays only if at least **{MIN_N_THRESHOLD} occurrences (N)** are found.
                """)

            with f_col3:
                st.markdown('<div class="footer-header">🏷️ TECHNICAL TAGS</div>', unsafe_allow_html=True)
                st.markdown(f"""
                * **EMA{EMA8_PERIOD} / EMA{EMA21_PERIOD}**: Added if the latest price is trading above (Bullish) or below (Bearish) these EMAs.
                * **VOL_HIGH**: Triggered if the signal candle volume is > 150% of the **{VOL_SMA_PERIOD}-period average**.
                * **V_GROW**: Triggered if volume at P2 is higher than volume at P1.
                """)
    except Exception as e: st.error(f"Error: {e}")
