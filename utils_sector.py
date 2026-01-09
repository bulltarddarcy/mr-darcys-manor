import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import StringIO

# Try to import the GDrive loader from the main utils
try:
    from utils import get_gdrive_binary_data
except ImportError:
    # Fallback if utils.py isn't found or structure differs
    import requests
    import re
    def get_gdrive_binary_data(url):
        match = re.search(r'/d/([a-zA-Z0-9_-]{25,})', url)
        if not match: match = re.search(r'id=([a-zA-Z0-9_-]{25,})', url)
        if not match: return None
        file_id = match.group(1)
        download_url = "https://drive.google.com/uc?export=download"
        response = requests.get(download_url, params={'id': file_id})
        return response.content if response.status_code == 200 else None

# --- CONFIGURATION (From Market Rotation) ---
HISTORY_YEARS = 1
BENCHMARK_TICKER = "SPY"
MA_FAST = 8         # EMA
MA_MEDIUM = 21      # EMA
MA_SLOW = 50        # SMA
MA_BASE = 200       # SMA
ATR_WINDOW = 20
AVG_VOLUME_WINDOW = 20

TIMEFRAMES = {
    'Short': 5,    # 5 Trading Days
    'Med':   10,   # 10 Trading Days
    'Long':  20    # 20 Trading Days
}

# --- DATA LOADING ---

@st.cache_data(ttl=3600)
def load_sector_universe():
    """
    Loads the universe from the SECTOR_UNIVERSE secret.
    Handles both a direct CSV string or a Google Drive URL/ID.
    """
    secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
    
    if not secret_val:
        st.error("Missing Secret: SECTOR_UNIVERSE")
        return pd.DataFrame(), [], {}

    df = None
    
    # Case A: Secret is a URL (GDrive)
    if "http" in secret_val or "drive.google.com" in secret_val:
        content = get_gdrive_binary_data(secret_val)
        if content:
            df = pd.read_csv(pd.io.common.BytesIO(content))
    # Case B: Secret is the raw CSV string
    else:
        try:
            df = pd.read_csv(StringIO(secret_val))
        except:
            pass
            
    if df is None or df.empty:
        st.error("Failed to load Sector Universe data.")
        return pd.DataFrame(), [], {}

    # Clean Data
    df.columns = [c.strip() for c in df.columns]
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].str.strip().str.upper()
    if 'Theme' in df.columns:
        df['Theme'] = df['Theme'].str.strip()
    
    # Default Role if missing
    if 'Role' not in df.columns:
        df['Role'] = 'Stock'
        
    # Map Themes to ETFs
    # Assuming 'Role' == 'ETF' defines the theme's benchmark
    theme_map = {}
    etf_rows = df[df['Role'].isin(['ETF', 'Benchmark'])]
    for _, row in etf_rows.iterrows():
        theme_map[row['Theme']] = row['Ticker']
        
    tickers = df['Ticker'].unique().tolist()
    
    # Ensure Benchmark is in the list
    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)
        
    return df, tickers, theme_map

@st.cache_data(ttl=3600)
def fetch_sector_data(tickers):
    """
    Downloads historical data for all tickers.
    Uses st.cache_data to prevent re-downloading on every rerun.
    """
    if not tickers:
        return {}
        
    end_date = datetime.today()
    start_date = end_date - timedelta(days=HISTORY_YEARS * 365 + 60) # Extra buffer for MAs
    
    # Download in bulk
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False,
        threads=True,
        progress=False
    )
    
    processed_data = {}
    
    # Handle single ticker case vs multi-ticker
    if len(tickers) == 1:
        processed_data[tickers[0]] = data
    else:
        for t in tickers:
            try:
                df = data[t].copy()
                if not df.empty:
                    # Drop multi-index level if exists
                    processed_data[t] = df
            except KeyError:
                pass
                
    return processed_data

# --- CALCULATIONS ---

def calculate_technical_indicators(df):
    if df.empty: return df
    
    # Moving Averages
    df['EMA_8'] = df['Close'].ewm(span=MA_FAST, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=MA_MEDIUM, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=MA_SLOW).mean()
    df['SMA_200'] = df['Close'].rolling(window=MA_BASE).mean()
    
    # Volatility (ADR)
    daily_range_pct = ((df['High'] - df['Low']) / df['Low']) * 100
    df['ADR_Pct'] = daily_range_pct.rolling(window=ATR_WINDOW).mean()
    
    # Relative Volume
    avg_vol = df["Volume"].rolling(window=AVG_VOLUME_WINDOW).mean()
    df["RVOL"] = df["Volume"] / avg_vol
    
    # RVOL over timeframes
    for label, days in TIMEFRAMES.items():
        df[f'RVOL_{label}'] = df['RVOL'].rolling(window=days).mean()
        
    return df

def calculate_rrg_metrics(df, benchmark_df):
    """
    Calculates Relative Rotation Graph (RRG) coordinates.
    JdK RS-Ratio (Trend) and JdK RS-Momentum (Momentum of Trend).
    """
    if df.empty or benchmark_df.empty: return df
    
    # 1. Relative Strength (RS) = Price / Benchmark
    # Align indices
    common_idx = df.index.intersection(benchmark_df.index)
    price = df.loc[common_idx, 'Close']
    bench = benchmark_df.loc[common_idx, 'Close']
    
    rs = (price / bench) * 100
    
    # 2. Process for each timeframe (Short, Med, Long)
    # RRG logic simplified approximation:
    # Ratio = RS / MovingAverage(RS)
    # Momentum = Ratio / MovingAverage(Ratio)
    
    for label, window in TIMEFRAMES.items():
        # Adjust windows based on timeframe "speed"
        # Standard RRG often uses roughly 10-week window. We adapt to our days.
        # Short: fast rotation. Med: standard. Long: slow.
        
        trend_window = window * 4  # e.g. 20 days for Short
        mom_window = window        # e.g. 5 days for Short
        
        # RS-Ratio (Trend)
        # 100 + ((RS - MA(RS)) / MA(RS)) * 100 approx, or just RS / MA(RS) * 100
        rs_ma = rs.rolling(window=trend_window).mean()
        rs_ratio = 100 + ((rs - rs_ma) / rs_ma) * 100
        
        # RS-Momentum (Rate of Change of Ratio)
        ratio_ma = rs_ratio.rolling(window=mom_window).mean()
        rs_mom = 100 + ((rs_ratio - ratio_ma) / ratio_ma) * 100
        
        # Reindex back to original df
        df.loc[common_idx, f'RRG_Ratio_{label}'] = rs_ratio
        df.loc[common_idx, f'RRG_Mom_{label}'] = rs_mom
        
        # Pure Alpha (Performance vs Spy)
        # Returns over N days - SPY returns over N days
        pct_change = df['Close'].pct_change(periods=window)
        spy_change = benchmark_df['Close'].pct_change(periods=window)
        
        df[f'Alpha_{label}'] = (pct_change - spy_change) * 100
        
    return df

def process_market_data(processed_data, theme_map, universe_df):
    """
    Master processing function.
    Returns:
    1. full_data: Dict of processed dataframes
    2. summary_df: Snapshot for tables
    """
    if BENCHMARK_TICKER not in processed_data:
        return {}, pd.DataFrame()
        
    spy_df = processed_data[BENCHMARK_TICKER]
    
    summary_rows = []
    
    for ticker, df in processed_data.items():
        if df.empty: continue
        
        # 1. Technicals
        df = calculate_technical_indicators(df)
        
        # 2. RRG & Alpha
        df = calculate_rrg_metrics(df, spy_df)
        
        # 3. Last Row Summary
        last = df.iloc[-1]
        
        # Find Theme info
        theme_row = universe_df[universe_df['Ticker'] == ticker]
        theme = theme_row['Theme'].values[0] if not theme_row.empty else "Unknown"
        role = theme_row['Role'].values[0] if not theme_row.empty else "Stock"
        
        row = {
            'Ticker': ticker,
            'Theme': theme,
            'Role': role,
            'Price': last['Close'],
            'EMA8_Above_21': last['EMA_8'] > last['EMA_21'],
            'Above_SMA200': last['Close'] > last['SMA_200'],
            'RVOL': last['RVOL']
        }
        
        # Add Timeframe specific stats
        for label in TIMEFRAMES.keys():
            row[f'Alpha_{label}'] = last.get(f'Alpha_{label}', 0)
            row[f'RRG_Ratio_{label}'] = last.get(f'RRG_Ratio_{label}', 100)
            row[f'RRG_Mom_{label}'] = last.get(f'RRG_Mom_{label}', 100)
            row[f'RVOL_{label}'] = last.get(f'RVOL_{label}', 1)
            
        summary_rows.append(row)
        
    return pd.DataFrame(summary_rows)

# --- PLOTTING ---

def get_quadrant_color(x, y):
    if x > 100 and y > 100: return "rgb(0, 255, 0)"      # Leading (Green)
    if x < 100 and y > 100: return "rgb(0, 100, 255)"    # Improving (Blue)
    if x < 100 and y < 100: return "rgb(255, 0, 0)"      # Lagging (Red)
    if x > 100 and y < 100: return "rgb(255, 165, 0)"    # Weakening (Yellow)
    return "grey"

def plot_rrg_chart(df, view='Med'):
    """
    Plots the Interactive RRG Chart using Plotly
    """
    if df.empty:
        return go.Figure()
        
    ratio_col = f'RRG_Ratio_{view}'
    mom_col = f'RRG_Mom_{view}'
    
    fig = go.Figure()
    
    # Draw Quadrant Lines
    fig.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", width=1, dash="dash"))
    
    # Background Labels
    fig.add_annotation(x=102, y=102, text="LEADING", showarrow=False, font=dict(color="rgba(0,255,0,0.5)", size=20), xanchor="left", yanchor="bottom")
    fig.add_annotation(x=98, y=102, text="IMPROVING", showarrow=False, font=dict(color="rgba(0,100,255,0.5)", size=20), xanchor="right", yanchor="bottom")
    fig.add_annotation(x=98, y=98, text="LAGGING", showarrow=False, font=dict(color="rgba(255,0,0,0.5)", size=20), xanchor="right", yanchor="top")
    fig.add_annotation(x=102, y=98, text="WEAKENING", showarrow=False, font=dict(color="rgba(255,165,0,0.5)", size=20), xanchor="left", yanchor="top")

    # Plot Points
    for i, row in df.iterrows():
        color = get_quadrant_color(row[ratio_col], row[mom_col])
        
        # Size based on Role (ETFs bigger)
        size = 12 if row['Role'] == 'ETF' else 8
        symbol = 'diamond' if row['Role'] == 'ETF' else 'circle'
        opacity = 0.9 if row['Role'] == 'ETF' else 0.6
        
        fig.add_trace(go.Scatter(
            x=[row[ratio_col]], 
            y=[row[mom_col]],
            mode='markers+text',
            text=[row['Ticker']],
            textposition="top center",
            name=row['Ticker'],
            marker=dict(color=color, size=size, symbol=symbol, opacity=opacity),
            hovertemplate=f"<b>{row['Ticker']}</b> ({row['Theme']})<br>Ratio: %{{x:.1f}}<br>Mom: %{{y:.1f}}<extra></extra>"
        ))
        
        # Add simple tail (last 5% movement trace could be added here if full history was passed, 
        # but for simplicity we are plotting the snapshot)

    # Dynamic Range
    x_max = max(abs(df[ratio_col].max() - 100), abs(df[ratio_col].min() - 100)) + 2
    y_max = max(abs(df[mom_col].max() - 100), abs(df[mom_col].min() - 100)) + 2
    limit = max(x_max, y_max)
    
    fig.update_layout(
        xaxis=dict(title="Relative Strength (Trend)", range=[100-limit, 100+limit], showgrid=False),
        yaxis=dict(title="Relative Momentum (Velocity)", range=[100-limit, 100+limit], showgrid=False),
        width=900, height=700,
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# --- STYLING UTILS ---

def highlight_alpha(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}'