import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from datetime import datetime

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

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Analysis Pro", layout="wide")

# Restored Original CSS
st.markdown("""
    <style>
    table { 
        width: 100%; 
        border-collapse: collapse; 
        table-layout: fixed; 
        margin-bottom: 2rem;
    }
    thead tr th {
        background-color: #f0f2f6 !important;
        color: #31333f !important;
        padding: 12px !important;
        border-bottom: 2px solid #dee2e6;
    }
    th:nth-child(1) { width: 10%; } /* Ticker */
    th:nth-child(2) { width: 32%; } /* Tags */
    th:nth-child(3) { width: 12%; } /* P1 Date */
    th:nth-child(4) { width: 12%; } /* Signal Date */
    th:nth-child(5) { width: 10%; } /* RSI */
    th:nth-child(6) { width: 12%; } /* P1 Price */
    th:nth-child(7) { width: 12%; } /* P2 Price */
    
    tbody tr td { 
        padding: 10px !important; 
        border-bottom: 1px solid #eee; 
        word-wrap: break-word;
    }
    .align-left { text-align: left !important; }
    .align-center { text-align: center !important; }
    .tag-bubble {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        color: white;
        white-space: nowrap;
    }
    .grey-note {
        color: #888888;
        font-size: 16px;
        margin-bottom: 20px;
    }
    /* Win Rate Card Styling */
    .analysis-card {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .pos-val { color: #27ae60; font-weight: bold; }
    .neg-val { color: #eb4d4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Original Helper Functions ---
def style_tags(tag_str):
    if not tag_str: return ''
    tags = tag_str.split(", ")
    html_str = ''
    colors = {
        f"EMA{EMA8_PERIOD}": "#4a90e2", 
        f"EMA{EMA21_PERIOD}": "#9b59b6", 
        "VOL_HIGH": "#e67e22",        
        "V_GROW": "#27ae60"           
    }
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
    w_ema8_col, w_ema21_col = 'W_EMA_8', 'W_EMA_21'
    w_high_col, w_low_col = 'W_HIGH', 'W_LOW'

    if not all([date_col, close_col, vol_col, high_col, low_col]): return None, None

    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col, d_ema8_col, d_ema21_col]].copy()
    df_d.rename(columns={
        close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low',
        d_rsi_col: 'RSI', d_ema8_col: 'EMA8', d_ema21_col: 'EMA21'
    }, inplace=True)
    df_d['VolSMA'] = df_d['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
    df_d = df_d.dropna(subset=['Price', 'RSI'])

    if all(c in df.columns for c in [w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col]):
        df_w = df[[w_close_col, w_vol_col, w_high_col, w_low_col, w_rsi_col, w_ema8_col, w_ema21_col]].copy()
        df_w.rename(columns={
            w_close_col: 'Price', w_vol_col: 'Volume', w_high_col: 'High', w_low_col: 'Low',
            w_rsi_col: 'RSI', w_ema8_col: 'EMA8', w_ema21_col: 'EMA21'
        }, inplace=True)
        df_w['VolSMA'] = df_w['Volume'].rolling(window=VOL_SMA_PERIOD).mean()
        df_w['ChartDate'] = df_w.index - pd.Timedelta(days=4)
        df_w = df_w.dropna(subset=['Price', 'RSI'])
    else:
        df_w = None
    
    return df_d, df_w

def find_divergences(df_tf, ticker, timeframe):
    divergences = []
    if len(df_tf) < DIVERGENCE_LOOKBACK + 1: return divergences
    latest_p = df_tf.iloc[-1]

    def get_date_str(p):
        return df_tf.loc[p.name, 'ChartDate'].strftime('%Y-%m-%d') if timeframe.lower() == 'weekly' else p.name.strftime('%Y-%m-%d')
            
    start_idx = max(DIVERGENCE_LOOKBACK, len(df_tf) - SIGNAL_LOOKBACK_PERIOD)
    
    for i in range(start_idx, len(df_tf)):
        p2 = df_tf.iloc[i]
        lookback = df_tf.iloc[i - DIVERGENCE_LOOKBACK : i]
        is_vol_high = int(p2['Volume'] > (p2['VolSMA'] * 1.5)) if not pd.isna(p2['VolSMA']) else 0
        
        # Bullish Divergence
        if p2['Low'] < lookback['Low'].min():
            p1 = lookback.loc[lookback['RSI'].idxmin()]
            if p2['RSI'] > (p1['RSI'] + RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] > 50).any():
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] <= p1['RSI']).any()):
                        tags = []
                        if 'EMA8' in latest_p and latest_p['Price'] >= latest_p['EMA8']: tags.append(f"EMA{EMA8_PERIOD}")
                        if 'EMA21' in latest_p and latest_p['Price'] >= latest_p['EMA21']: tags.append(f"EMA{EMA21_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROW")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bullish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['Low']:,.2f}", 'P2 Price': f"${p2['Low']:,.2f}"
                        })

        # Bearish Divergence
        if p2['High'] > lookback['High'].max():
            p1 = lookback.loc[lookback['RSI'].idxmax()]
            if p2['RSI'] < (p1['RSI'] - RSI_DIFF_THRESHOLD):
                if not (df_tf.loc[p1.name : p2.name, 'RSI'] < 50).any():
                    post_df = df_tf.iloc[i + 1 :]
                    if not (not post_df.empty and (post_df['RSI'] >= p1['RSI']).any()):
                        tags = []
                        if 'EMA8' in latest_p and latest_p['Price'] <= latest_p['EMA8']: tags.append(f"EMA{EMA8_PERIOD}")
                        if 'EMA21' in latest_p and latest_p['Price'] <= latest_p['EMA21']: tags.append(f"EMA{EMA21_PERIOD}")
                        if is_vol_high: tags.append("VOL_HIGH")
                        if p2['Volume'] > p1['Volume']: tags.append("V_GROW")
                        divergences.append({
                            'Ticker': ticker, 'Type': 'Bearish', 'Timeframe': timeframe, 'Tags': ", ".join(tags),
                            'P1 Date': get_date_str(p1), 'Signal Date': get_date_str(p2),
                            'RSI': f"{int(round(p1['RSI']))} → {int(round(p2['RSI']))}",
                            'P1 Price': f"${p1['High']:,.2f}", 'P2 Price': f"${p2['High']:,.2f}"
                        })
    return divergences

# --- Win Rate Helper ---
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

# --- App Execution ---
st.title("📈 RSI Multi-Analysis Dashboard")
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

# Top Tabs for navigation
tab_div, tab_win = st.tabs(["🔍 Divergence Scanner", "📊 Forward Win Rates"])

if data_option:
    secret_key = dataset_map[data_option]
    target_url = st.secrets[secret_key]
    csv_buffer = get_confirmed_gdrive_data(target_url)

    if csv_buffer and csv_buffer != "HTML_ERROR":
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), 'TICKER')
        all_tickers = sorted(master[t_col].unique())

        with tab_div:
            st.markdown('<div class="grey-note">ℹ️ See bottom of page for strategy logic and tag explanations.</div>', unsafe_allow_html=True)
            
            with st.expander(f"🔍 View Scanned Tickers ({len(all_tickers)} symbols)"):
                sq = st.text_input("Filter...").upper()
                ft = [t for t in all_tickers if sq in t]
                cols = st.columns(6)
                for i, ticker in enumerate(ft): cols[i % 6].write(ticker)

            raw_results = []
            progress_bar = st.progress(0, text="Scanning for Divergences...")
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
                            display_df = tbl_df.drop(columns=['Type', 'Timeframe'])
                            html = '<table><thead><tr>'
                            for col in display_df.columns:
                                cls = 'align-left' if col in ['Ticker', 'Tags', 'P1 Price', 'P2 Price'] else 'align-center'
                                html += f'<th class="{cls}">{col}</th>'
                            html += '</tr></thead><tbody>'
                            for _, row in display_df.iterrows():
                                html += f'<tr><td class="align-left"><b>{row["Ticker"]}</b></td>'
                                html += f'<td class="align-left">{style_tags(row["Tags"])}</td>'
                                html += f'<td class="align-center">{row["P1 Date"]}</td>'
                                html += f'<td class="align-center">{row["Signal Date"]}</td>'
                                html += f'<td class="align-center">{row["RSI"]}</td>'
                                html += f'<td class="align-left">{row["P1 Price"]}</td>'
                                html += f'<td class="align-left">{row["P2 Price"]}</td></tr>'
                            html += '</tbody></table>'
                            st.markdown(html, unsafe_allow_html=True)
                        else: st.write("No signals.")

            # Restored Original Footer
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📝 Strategy Logic")
                st.markdown(f"* **Signal Window**: Last **{SIGNAL_LOOKBACK_PERIOD} periods**.\n* **Lookback Window**: Preceding **{DIVERGENCE_LOOKBACK} periods**.")
            with col2:
                st.subheader("🏷️ Tags Explained")
                st.markdown(f"* **EMA8 / EMA21**: Holding above/below levels.\n* **VOL_HIGH**: Vol > 150% avg.\n* **V_GROW**: Vol > P1 Vol.")

        with tab_win:
            st.header("🎯 Forward Returns Analysis (NFLX Test)")
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
