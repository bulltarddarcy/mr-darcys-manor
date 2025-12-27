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

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RSI Analysis Tool", layout="wide")

# --- Custom CSS for the Win Rate Boxes ---
st.markdown("""
<style>
    .win-rate-card {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stats-table {
        width: 100%;
        border-collapse: collapse;
        font-family: monospace;
        font-size: 14px;
    }
    .stats-table th {
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #eee;
        color: #666;
    }
    .stats-table td {
        padding: 8px;
        border-bottom: 1px solid #f9f9f9;
    }
    .positive { color: green; }
    .negative { color: red; }
</style>
""", unsafe_allow_html=True)

# --- Navigation ---
page = st.sidebar.selectbox("Select Page", ["RSI Divergences", "RSI Forward Win Rates"])

# --- Data Preparation Helpers ---
def prepare_data(df):
    df.columns = [col.strip().replace(' ', '').replace('-', '').upper() for col in df.columns]
    date_col = next((col for col in df.columns if 'DATE' in col), None)
    close_col = next((col for col in df.columns if 'CLOSE' in col and 'W_' not in col), None)
    vol_col = next((col for col in df.columns if ('VOL' in col or 'VOLUME' in col) and 'W_' not in col), None)
    high_col = next((col for col in df.columns if 'HIGH' in col and 'W_' not in col), None)
    low_col = next((col for col in df.columns if 'LOW' in col and 'W_' not in col), None)
    
    d_rsi_col = 'RSI_14'
    if not all([date_col, close_col, vol_col, high_col, low_col]): return None
    
    df.index = pd.to_datetime(df[date_col])
    df = df.sort_index()
    
    df_d = df[[close_col, vol_col, high_col, low_col, d_rsi_col]].copy()
    df_d.rename(columns={close_col: 'Price', vol_col: 'Volume', high_col: 'High', low_col: 'Low', d_rsi_col: 'RSI'}, inplace=True)
    return df_d

# --- RSI Forward Win Rate Logic ---
def calculate_forward_stats(df, current_rsi, tolerance=2):
    # RSI Range for matching
    lower_rsi = current_rsi - tolerance
    upper_rsi = current_rsi + tolerance
    
    # Find historical periods where RSI was in this range
    matching_indices = df[(df['RSI'] >= lower_rsi) & (df['RSI'] <= upper_rsi)].index
    sample_count = len(matching_indices)
    
    windows = [1, 3, 5, 7, 10, 14, 30, 60, 90, 180]
    results = []
    
    for w in windows:
        returns = []
        for idx in matching_indices:
            try:
                # Get the integer location to find forward price
                pos = df.index.get_loc(idx)
                if pos + w < len(df):
                    future_price = df.iloc[pos + w]['Price']
                    entry_price = df.iloc[pos]['Price']
                    ret = (future_price - entry_price) / entry_price
                    returns.append(ret)
            except: continue
        
        if returns:
            win_rate = (sum(1 for r in returns if r > 0) / len(returns)) * 100
            avg_ret = np.mean(returns) * 100
            med_ret = np.median(returns) * 100
            results.append({
                "Days": w,
                "Win Rate": f"{win_rate:.1f}%",
                "Avg Ret": f"{'+' if avg_ret > 0 else ''}{avg_ret:.2f}%",
                "Med Ret": f"{'+' if med_ret > 0 else ''}{med_ret:.2f}%",
                "raw_avg": avg_ret,
                "raw_med": med_ret
            })
    
    return results, sample_count

# --- App Execution ---
dataset_map = load_dataset_config()
data_option = st.pills("Select Dataset", options=list(dataset_map.keys()), selection_mode="single", default=list(dataset_map.keys())[0])

if data_option:
    target_url = st.secrets[dataset_map[data_option]]
    csv_buffer = get_confirmed_gdrive_data(target_url)

    if csv_buffer and csv_buffer != "HTML_ERROR":
        master = pd.read_csv(csv_buffer)
        t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
        
        if page == "RSI Forward Win Rates":
            st.title("📊 RSI Forward Win Rates")
            
            # --- TESTING LOCK: NFLX ONLY ---
            target_tickers = ["NFLX"] 
            st.info("Testing Mode: Displaying NFLX only.")
            
            for ticker in target_tickers:
                ticker_df = master[master[t_col] == ticker].copy()
                df_clean = prepare_data(ticker_df)
                
                if df_clean is not None:
                    current_rsi = df_clean['RSI'].iloc[-1]
                    stats, samples = calculate_forward_stats(df_clean, current_rsi)
                    
                    # Box Header
                    st.markdown(f"""
                    <div class="win-rate-card">
                        <h3>RSI Analysis: {ticker}</h3>
                        <p><b>Current RSI: {current_rsi:.2f}</b> (live as of {df_clean.index[-1].strftime('%Y-%m-%d')})</p>
                        <p style="color: #666;">RSI Range: [{current_rsi-2:.2f}, {current_rsi+2:.2f}]<br>Matching Periods: {samples}</p>
                    """, unsafe_allow_html=True)
                    
                    # Split into Short and Long Term Tables
                    for label, slice_range in [("Short-Term Forward Returns", stats[:6]), ("Long-Term Forward Returns", stats[6:])]:
                        st.write(f"**{label}**")
                        html = '<table class="stats-table"><tr><th>Days</th><th>Win Rate</th><th>Avg Ret</th><th>Med Ret</th></tr>'
                        for row in slice_range:
                            avg_cls = "positive" if row['raw_avg'] > 0 else "negative"
                            med_cls = "positive" if row['raw_med'] > 0 else "negative"
                            html += f"""
                            <tr>
                                <td>{row['Days']}</td>
                                <td>{row['Win Rate']}</td>
                                <td class="{avg_cls}">{row['Avg Ret']}</td>
                                <td class="{med_cls}">{row['Med Ret']}</td>
                            </tr>
                            """
                        html += "</table><br>"
                        st.markdown(html, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

        elif page == "RSI Divergences":
            # (Keep your existing divergence logic here)
            st.write("Divergence Page - (Add previous logic here)")
