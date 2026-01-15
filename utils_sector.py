"""
Sector rotation utilities - AUTO-SAVE VERSION
Automatically loads optimized settings from a local JSON file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
import json
import os
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)

# --- IMPORT SHARED UTILS ---
import utils_darcy as ud

# ==========================================
# 0. SMART CONFIGURATION (AUTO-LOADER)
# ==========================================
CONFIG_FILENAME = "sector_optimized_config.json"

def load_dynamic_config() -> Dict:
    """Loads optimized settings from local JSON file. Returns empty dict if file missing."""
    if os.path.exists(CONFIG_FILENAME):
        try:
            with open(CONFIG_FILENAME, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    return {}

def save_dynamic_config(config_dict: Dict):
    """Saves optimized settings to local JSON file."""
    try:
        with open(CONFIG_FILENAME, 'w') as f:
            json.dump(config_dict, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving config: {e}")

# Default Fallback
DEFAULT_WINDOW = 'Med'   # 10 Days
DEFAULT_SMOOTH = 3       # 3 Days

# ==========================================
# 1. CONSTANTS
# ==========================================
HISTORY_YEARS = 1

# Volatility & Normalization
ATR_WINDOW = 20
AVG_VOLUME_WINDOW = 20
BETA_WINDOW = 60

# Timeframes (Trading Days)
TIMEFRAMES = {
    'Short': 5,
    'Med': 10,
    'Long': 20
}

# Filters
MIN_DOLLAR_VOLUME = 10_000_000

# Regression Weights
WEIGHT_MIN = 1.0
WEIGHT_MAX = 3.0

# Visualization
MARKER_SIZE_TRAIL = 8
MARKER_SIZE_CURRENT = 15
TRAIL_OPACITY = 0.4
CURRENT_OPACITY = 1.0

# Pattern Detection Thresholds
JHOOK_MIN_SHIFT = 2.0
ALPHA_DIP_BUY_THRESHOLD = 2.0
ALPHA_NEUTRAL_RANGE = 0.5
ALPHA_BREAKOUT_THRESHOLD = 1.0
ALPHA_FADING_THRESHOLD = 3.0
RVOL_HIGH_THRESHOLD = 1.3
RVOL_BREAKOUT_THRESHOLD = 1.3

# ==========================================
# 2. DATA MANAGER
# ==========================================
class SectorDataManager:
    """Manages sector universe configuration and ticker mapping."""
    
    def __init__(self):
        self.universe = pd.DataFrame()
        self.ticker_map = {}

    def load_universe(self, benchmark_ticker: str = "SPY") -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
        """Load sector universe from secrets."""
        secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
        if not secret_val:
            st.error("❌ Secret 'SECTOR_UNIVERSE' is missing or empty.")
            return pd.DataFrame(), [], {}
            
        try:
            if secret_val.strip().startswith("http"):
                if "docs.google.com/spreadsheets" in secret_val:
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
                elif "drive.google.com" in secret_val and "/d/" in secret_val:
                    file_id = secret_val.split("/d/")[1].split("/")[0]
                    csv_source = f"https://drive.google.com/uc?id={file_id}&export=download"
                else:
                    csv_source = secret_val
                df = pd.read_csv(csv_source)
            else:
                df = pd.read_csv(StringIO(secret_val))
            
            required_cols = ['Ticker', 'Theme']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Universe CSV missing required columns: {required_cols}")
                return pd.DataFrame(), [], {}

            df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
            df['Theme'] = df['Theme'].astype(str).str.strip()
            df['Role'] = df['Role'].astype(str).str.strip().str.title() if 'Role' in df.columns else 'Stock'

            tickers = df['Ticker'].unique().tolist()
            if benchmark_ticker not in tickers:
                tickers.append(benchmark_ticker)
                
            etf_rows = df[df['Role'] == 'Etf']
            theme_map = dict(zip(etf_rows['Theme'], etf_rows['Ticker'])) if not etf_rows.empty else {}
            
            self.universe = df
            logger.info(f"Loaded universe with {len(tickers)} tickers, {len(theme_map)} themes")
            return df, tickers, theme_map
            
        except Exception as e:
            logger.exception(f"Error loading SECTOR_UNIVERSE: {e}")
            st.error(f"Error loading SECTOR_UNIVERSE: {e}")
            return pd.DataFrame(), [], {}

# ==========================================
# 3. HELPERS
# ==========================================
def get_ma_signal(price: float, ma_val: float) -> str:
    if pd.isna(ma_val) or ma_val == 0:
        return "⚠️"
    return "✅" if price > ma_val else "❌"

def _shorten_category_name(name: str) -> str:
    return name.replace("Gaining Momentum", "Gain Mom") \
               .replace("Losing Momentum", "Lose Mom") \
               .replace("Outperforming", "Outperf") \
               .replace("Underperforming", "Underperf")

# ==========================================
# 4. CALCULATOR & PIPELINE
# ==========================================
class SectorAlphaCalculator:
    def process_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty: return None

        if 'Close' not in df.columns and 'CLOSE' in df.columns: df['Close'] = df['CLOSE']
        if 'Close' not in df.columns: return None
        
        if 'Volume' in df.columns:
            avg_vol = df["Volume"].rolling(window=AVG_VOLUME_WINDOW).mean()
            df["RVOL"] = df["Volume"] / avg_vol
            for label, time_window in TIMEFRAMES.items():
                df[f"RVOL_{label}"] = df["RVOL"].rolling(window=time_window).mean()
        
        if 'High' in df.columns and 'Low' in df.columns:
            daily_range_pct = ((df['High'] - df['Low']) / df['Low']) * 100
            df['ADR_Pct'] = daily_range_pct.rolling(window=ATR_WINDOW).mean()
        
        return df

    def _calc_slope(self, series: pd.Series, window: int) -> pd.Series:
        x = np.arange(window)
        weights = np.ones(window) if window < 10 else np.linspace(WEIGHT_MIN, WEIGHT_MAX, window)
        def slope_func(y):
            try: return np.polyfit(x, y, 1, w=weights)[0]
            except: return 0.0
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(self, df: pd.DataFrame, bench_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty or bench_df is None or bench_df.empty: return None
        common_index = df.index.intersection(bench_df.index)
        if common_index.empty: return None
        
        df_aligned = df.loc[common_index].copy()
        bench_aligned = bench_df.loc[common_index].copy()
        
        asset_col = 'Adj Close' if 'Adj Close' in df_aligned.columns else 'Close'
        bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else 'Close'
        
        raw_ratio = df_aligned[asset_col] / bench_aligned[bench_col]
        
        for label, time_window in TIMEFRAMES.items():
            ratio_mean = raw_ratio.rolling(window=time_window).mean()
            col_ratio = f"RRG_Ratio_{label}"
            df_aligned[col_ratio] = ((raw_ratio - ratio_mean) / ratio_mean) * 100 + 100
            
            raw_slope = self._calc_slope(raw_ratio, time_window)
            velocity = (raw_slope / ratio_mean) * time_window * 100
            
            col_mom = f"RRG_Mom_{label}"
            df_aligned[col_mom] = 100 + velocity
        
        for col in df_aligned.columns:
            if col.startswith('RRG_'): df[col] = df_aligned[col]
        return df

    def calculate_stock_alpha_multi_theme(self, df: pd.DataFrame, parent_df: pd.DataFrame, theme_suffix: str) -> pd.DataFrame:
        if df is None or df.empty or parent_df is None or parent_df.empty: return df
        common_index = df.index.intersection(parent_df.index)
        if common_index.empty: return df
        
        df_aligned = df.loc[common_index].copy()
        parent_aligned = parent_df.loc[common_index].copy()
        
        if 'Pct_Change' not in df_aligned.columns: df_aligned['Pct_Change'] = df_aligned['Close'].pct_change()
        if 'Pct_Change' not in parent_aligned.columns: parent_aligned['Pct_Change'] = parent_aligned['Close'].pct_change()
        
        rolling_cov = df_aligned['Pct_Change'].rolling(window=BETA_WINDOW).cov(parent_aligned['Pct_Change'])
        rolling_var = parent_aligned['Pct_Change'].rolling(window=BETA_WINDOW).var()
        beta_col = f"Beta_{theme_suffix}"
        df_aligned[beta_col] = np.where(rolling_var > 1e-8, rolling_cov / rolling_var, 1.0)
        
        expected_return_col = f"Expected_Return_{theme_suffix}"
        df_aligned[expected_return_col] = parent_aligned['Pct_Change'] * df_aligned[beta_col]
        
        alpha_1d_col = f"Alpha_1D_{theme_suffix}"
        df_aligned[alpha_1d_col] = df_aligned['Pct_Change'] - df_aligned[expected_return_col]
        
        for label, time_window in TIMEFRAMES.items():
            alpha_col = f"Alpha_{label}_{theme_suffix}"
            df_aligned[alpha_col] = df_aligned[alpha_1d_col].fillna(0).rolling(window=time_window).sum() * 100
        
        for col in df_aligned.columns:
            if col.endswith(f"_{theme_suffix}") or col == beta_col: df[col] = df_aligned[col]
        return df

# ==========================================
# 5. ANALYSIS FUNCTIONS
# ==========================================

def analyze_stocks_batch(etf_data_cache, stock_theme_pairs, show_biotech, theme_category_map):
    if not stock_theme_pairs: return pd.DataFrame()
    def process_single_stock(stock, stock_theme):
        if stock_theme == "Biotech" and not show_biotech: return None
        sdf = etf_data_cache.get(stock)
        if sdf is None or sdf.empty or len(sdf) < 20: return None
        try:
            if (sdf['Volume'].values[-20:].mean() * sdf['Close'].values[-20:].mean()) < MIN_DOLLAR_VOLUME: return None
        except: return None
        try:
            last = sdf.iloc[-1]
            return {
                "Ticker": stock, "Theme": stock_theme, "Theme Category": theme_category_map.get(stock_theme, "Unknown"),
                "Price": last['Close'], "Alpha 5d": last.get(f"Alpha_Short_{stock_theme}", 0),
                "Alpha 10d": last.get(f"Alpha_Med_{stock_theme}", 0), "Alpha 20d": last.get(f"Alpha_Long_{stock_theme}", 0),
                "RVOL 5d": last.get('RVOL_Short', 0), "RVOL 10d": last.get('RVOL_Med', 0), "RVOL 20d": last.get('RVOL_Long', 0),
                "Market Cap (B)": 0.0, "Div": "-", "Beta": last.get(f"Beta_{stock_theme}", 1.0),
                "8 EMA": get_ma_signal(last['Close'], last.get('Ema8', 0)), "21 EMA": get_ma_signal(last['Close'], last.get('Ema21', 0)),
                "50 MA": get_ma_signal(last['Close'], last.get('Sma50', 0)), "200 MA": get_ma_signal(last['Close'], last.get('Sma200', 0)),
            }
        except Exception: return None

    stock_data = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_stock = {executor.submit(process_single_stock, s, t): s for s, t in stock_theme_pairs}
        for future in as_completed(future_to_stock):
            result = future.result()
            if result: stock_data.append(result)
    return pd.DataFrame(stock_data)

def enrich_stock_data(df, etf_data_cache, show_mkt_caps, show_divergences):
    if df.empty or (not show_mkt_caps and not show_divergences): return df
    unique_tickers = df['Ticker'].unique().tolist()
    enriched_df = df.copy()
    
    if show_mkt_caps:
        with st.spinner("Fetching Market Caps..."):
            mc_map = ud.fetch_market_caps_batch(unique_tickers)
            enriched_df['Market Cap (B)'] = enriched_df['Ticker'].map(mc_map).fillna(0) / 1e9

    if show_divergences:
        div_map = {}
        with st.spinner("Scanning Divergences..."):
            def process_div_single(stock):
                sdf = etf_data_cache.get(stock)
                if sdf is None: return stock, "—"
                try:
                    d_d, _ = ud.prepare_data(sdf.copy())
                    if d_d is not None and not d_d.empty:
                        divs = ud.find_divergences(d_d, stock, 'Daily', min_n=0,
                            periods_input=ud.DIV_CSV_PERIODS_DAYS, optimize_for='PF',
                            lookback_period=ud.DIV_LOOKBACK_DEFAULT, price_source=ud.DIV_SOURCE_DEFAULT,
                            strict_validation=(ud.DIV_STRICT_DEFAULT == "Yes"),
                            recent_days_filter=ud.DIV_DAYS_SINCE_DEFAULT, rsi_diff_threshold=ud.DIV_RSI_DIFF_DEFAULT
                        )
                        active_divs = [d for d in divs if d.get('Is_Recent', False)]
                        if active_divs:
                            d_type = active_divs[-1]['Type']
                            return stock, (f"🟢 {d_type}" if d_type == 'Bullish' else f"🔴 {d_type}")
                except: pass
                return stock, "—"

            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_div = {executor.submit(process_div_single, t): t for t in unique_tickers}
                for future in as_completed(future_to_div):
                    t, d_str = future.result()
                    div_map[t] = d_str
            enriched_df['Div'] = enriched_df['Ticker'].map(div_map).fillna("-")
    return enriched_df

def apply_stock_filters(df_stocks: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
    if df_stocks.empty or not filters: return df_stocks
    df_filtered = df_stocks.copy()
    numeric_filters = [f for f in filters if f['value_type'] in ['Number', 'Column']]
    categorical_filters = [f for f in filters if f['value_type'] == 'Categorical']
    
    numeric_conditions = []
    for f in numeric_filters:
        col, op = f['column'], f['operator']
        if col not in df_filtered.columns: continue
        if f['value_type'] == 'Number':
            cond = (df_filtered[col] >= f['value']) if op == '>=' else (df_filtered[col] <= f['value'])
        else:
            if f['value_column'] not in df_filtered.columns: continue
            cond = (df_filtered[col] >= df_filtered[f['value_column']]) if op == '>=' else (df_filtered[col] <= df_filtered[f['value_column']])
        numeric_conditions.append(cond)
        
    categorical_conditions = []
    for i, f in enumerate(categorical_filters):
        col, val = f['column'], f['value_categorical']
        if col not in df_filtered.columns: continue
        cond = (df_filtered[col] == val)
        logic = f.get('logic', 'AND') if i > 0 else None
        categorical_conditions.append((cond, logic))
        
    final_condition = None
    if numeric_conditions:
        final_condition = numeric_conditions[0]
        for c in numeric_conditions[1:]: final_condition = final_condition & c
        
    if categorical_conditions:
        cat_combined = categorical_conditions[0][0]
        for i in range(1, len(categorical_conditions)):
            cond, logic = categorical_conditions[i]
            if logic == 'OR': cat_combined = cat_combined | cond
            else: cat_combined = cat_combined & cond
        final_condition = (final_condition & cat_combined) if final_condition is not None else cat_combined
        
    if final_condition is not None: df_filtered = df_filtered[final_condition]
    return df_filtered

# ==========================================
# 6. DYNAMIC CORE LOGIC (CATEGORIES & DAYS)
# ==========================================

def calculate_days_in_category_dynamic(df: pd.DataFrame, window_key: str, smooth: int) -> Dict[str, int]:
    if df is None or df.empty or len(df) < (smooth + 1): return {'days': 0, 'category': 'Unknown'}
    try:
        col_r = f'RRG_Ratio_{window_key}'
        col_m = f'RRG_Mom_{window_key}'
        recent_days = df.tail(30).copy()
        recent_days['Mean_R'] = recent_days[col_r].shift(1).rolling(window=smooth).mean()
        recent_days['Mean_M'] = recent_days[col_m].shift(1).rolling(window=smooth).mean()
        is_out = recent_days[col_r] > recent_days['Mean_R']
        is_gain = recent_days[col_m] > recent_days['Mean_M']
        
        cats = []
        for i in range(len(recent_days)):
            if pd.isna(recent_days['Mean_R'].iloc[i]): cats.append("Unknown")
            else:
                p = "outperforming" if is_out.iloc[i] else "underperforming"
                m = "gaining" if is_gain.iloc[i] else "losing"
                cats.append(f"{m}_{p}")
                
        if not cats: return {'days': 0, 'category': 'Unknown'}
        current_cat = cats[-1]
        days = 0
        for i in range(len(cats) - 1, -1, -1):
            if cats[i] == current_cat: days += 1
            else: break
        return {'days': days, 'category': current_cat}
    except Exception as e:
        logger.error(f"Error calculating dynamic days: {e}")
        return {'days': 0, 'category': 'Unknown'}

def get_momentum_performance_categories(
    etf_data_cache: Dict, 
    theme_map: Dict, 
    force_timeframe: str = None
) -> Dict[str, List[Dict]]:
    """
    Categorize themes using either DYNAMIC SETTINGS from file 
    OR a forced timeframe (if provided).
    """
    
    # 1. LOAD CONFIG FROM FILE (If needed)
    dynamic_config = load_dynamic_config() if not force_timeframe else {}
    
    categories = {
        'gaining_mom_outperforming': [],
        'gaining_mom_underperforming': [],
        'losing_mom_outperforming': [],
        'losing_mom_underperforming': []
    }
    
    for theme, ticker in theme_map.items():
        df = etf_data_cache.get(ticker)
        if df is None or df.empty or len(df) < 20: continue
        
        try:
            # 2. DECIDE SETTINGS
            if force_timeframe:
                # Manual Override (e.g. User unchecked "Smart Optimization")
                # Use standard smoothing (3) with the forced window
                win_key = force_timeframe
                smooth_win = 3 
            else:
                # Optimized Logic (User checked "Smart Optimization")
                settings = dynamic_config.get(ticker, [DEFAULT_WINDOW, DEFAULT_SMOOTH])
                win_key, smooth_win = settings[0], int(settings[1])
            
            # 3. Calculation
            col_r, col_m = f"RRG_Ratio_{win_key}", f"RRG_Mom_{win_key}"
            
            recent = df.tail(smooth_win + 2).copy()
            today_r, today_m = recent[col_r].iloc[-1], recent[col_m].iloc[-1]
            prev_r_avg = recent[col_r].iloc[-(smooth_win+1):-1].mean()
            prev_m_avg = recent[col_m].iloc[-(smooth_win+1):-1].mean()
            
            perf_dir = "Outperforming" if today_r > prev_r_avg else "Underperforming"
            mom_dir = "Gaining Momentum" if today_m > prev_m_avg else "Losing Momentum"
            
            bucket_key = f"{'gaining_mom' if today_m > prev_m_avg else 'losing_mom'}_{'outperforming' if today_r > prev_r_avg else 'underperforming'}"
            arrow = "⬈" if bucket_key == 'gaining_mom_outperforming' else "⬉" if bucket_key == 'gaining_mom_underperforming' else "⬊" if bucket_key == 'losing_mom_outperforming' else "⬋"
            
            full_cat_name = f"{mom_dir} & {perf_dir}"
            short_cat_name = _shorten_category_name(full_cat_name)
            streak_info = calculate_days_in_category_dynamic(df, win_key, smooth_win)
            
            # Standard metrics for reference
            r5, m5 = df['RRG_Ratio_Short'].iloc[-1], df['RRG_Mom_Short'].iloc[-1]
            r10, m10 = df['RRG_Ratio_Med'].iloc[-1], df['RRG_Mom_Med'].iloc[-1]
            r20, m20 = df['RRG_Ratio_Long'].iloc[-1], df['RRG_Mom_Long'].iloc[-1]

            theme_info = {
                'theme': theme, 
                'category': full_cat_name, 
                'display_category': arrow + " " + short_cat_name,
                'arrow': arrow, 
                'quadrant_5d': get_quadrant_name(r5, m5),
                'quadrant_10d': get_quadrant_name(r10, m10), 
                'quadrant_20d': get_quadrant_name(r20, m20),
                'reason': f"Logic: {win_key}/{smooth_win}d", 
                'days_in_category': streak_info['days']
            }
            categories[bucket_key].append(theme_info)
        except: continue
            
    for k in categories: categories[k].sort(key=lambda x: x['theme'])
    return categories

def get_quadrant_name(ratio: float, momentum: float) -> str:
    if ratio >= 100 and momentum >= 100: return "🟢 Leading"
    elif ratio < 100 and momentum >= 100: return "🔵 Improving"
    elif ratio >= 100 and momentum < 100: return "🟡 Weakening"
    else: return "🔴 Lagging"

# ==========================================
# 7. ORCHESTRATOR (The Missing Function Restored!)
# ==========================================
@st.cache_data(ttl=ud.CACHE_TTL, show_spinner=False)
def fetch_and_process_universe(benchmark_ticker: str = "SPY"):
    dm = SectorDataManager()
    uni_df, tickers, theme_map = dm.load_universe(benchmark_ticker)
    if uni_df.empty: return {}, ["SECTOR_UNIVERSE is empty"], theme_map, uni_df, {}

    db_url = st.secrets.get("PARQUET_SECTOR_ROTATION")
    if not db_url: return {}, ["PARQUET secret missing"], theme_map, uni_df, {}

    try:
        buffer = ud.get_gdrive_binary_data(db_url)
        master_df = pd.read_parquet(buffer)
    except Exception as e: return {}, [f"Error reading Parquet: {e}"], theme_map, uni_df, {}

    # Standardize & Filter
    master_df.columns = [c.strip().title() for c in master_df.columns]
    if 'Symbol' in master_df.columns: master_df.rename(columns={'Symbol': 'Ticker'}, inplace=True)
    master_df['Ticker'] = master_df['Ticker'].str.upper().str.strip()
    if 'Date' in master_df.columns:
        master_df['Date'] = pd.to_datetime(master_df['Date'])
        master_df = master_df.set_index('Date').sort_index()

    needed = set(tickers) | set(theme_map.values()) | {benchmark_ticker, "SPY", "QQQ"} 
    master_df = master_df[master_df['Ticker'].isin(needed)].copy()
    if master_df.empty: return {}, ["No tickers found"], theme_map, uni_df, {}

    # Vectorized Calcs
    master_df.sort_values(['Ticker', 'Date'], inplace=True)
    master_df['Pct_Change'] = master_df.groupby('Ticker')['Close'].pct_change()
    
    if 'Volume' in master_df.columns:
        avg_vol = master_df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=AVG_VOLUME_WINDOW).mean())
        master_df['RVOL'] = master_df['Volume'] / avg_vol
        for label, time_window in TIMEFRAMES.items():
            master_df[f"RVOL_{label}"] = master_df.groupby('Ticker')['RVOL'].transform(lambda x: x.rolling(window=time_window).mean())
            
    if 'Close' not in master_df.columns and 'Adj Close' in master_df.columns: master_df['Close'] = master_df['Adj Close']

    # Cache Building
    calc = SectorAlphaCalculator()
    data_cache = {}
    missing_tickers = []
    full_data_map = {t: df for t, df in master_df.groupby('Ticker')}

    if benchmark_ticker not in full_data_map: return {}, [f"Benchmark {benchmark_ticker} missing"], theme_map, uni_df, {}
    bench_df = full_data_map[benchmark_ticker].copy()
    data_cache[benchmark_ticker] = bench_df
    
    for b in ["SPY", "QQQ"]:
        if b in full_data_map: data_cache[b] = full_data_map[b]

    # Process ETFs
    for etf in theme_map.values():
        if etf in full_data_map:
            data_cache[etf] = calc.calculate_rrg_metrics(full_data_map[etf].copy(), bench_df)
        else: missing_tickers.append(etf)

    # Process Stocks (Parallel)
    stocks = uni_df[uni_df['Role'] == 'Stock']
    stock_themes_map = stocks.groupby('Ticker')['Theme'].apply(list).to_dict()
    
    def process_stock_worker(ticker, themes_list):
        if ticker not in full_data_map: return ticker, None, True
        try:
            df = full_data_map[ticker].copy()
            for theme in themes_list:
                parent_df = data_cache.get(theme_map.get(theme), bench_df)
                if parent_df is not None: df = calc.calculate_stock_alpha_multi_theme(df, parent_df, theme)
            return ticker, df, False
        except: return ticker, None, False

    if stock_themes_map:
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_stock = {executor.submit(process_stock_worker, t, stock_themes_map[t]): t for t in stock_themes_map.keys()}
            for future in as_completed(future_to_stock):
                t, res, is_miss = future.result()
                if is_miss: missing_tickers.append(t)
                elif res is not None: data_cache[t] = res

    logger.info(f"Sync complete. {len(data_cache)} tickers.")
    return data_cache, missing_tickers, theme_map, uni_df, stock_themes_map

# ==========================================
# 8. EXPORT & GENERATORS (PHASE 1)
# ==========================================
class UniverseGenerator:
    def fetch_etf_holdings(self, etf_ticker: str) -> List[Dict]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(etf_ticker)
            try:
                holdings = ticker.funds_data.top_holdings.reset_index()
                holdings.rename(columns={'Symbol': 'ticker', 'Holding Percent': 'weight'}, inplace=True)
                return holdings.to_dict('records')
            except: pass
            return []
        except Exception as e:
            logger.error(f"Error fetching holdings for {etf_ticker}: {e}")
            return []

    def generate_universe(self, theme_map, target_cumulative_weight=0.60, max_tickers_per_sector=None, min_dollar_volume=10_000_000):
        summary_stats = []
        raw_candidates = {} 
        progress_bar = st.progress(0); status_text = st.empty()
        items = list(theme_map.items()); total_steps = len(items) + 1
        
        for i, (theme, etf) in enumerate(items):
            status_text.text(f"Scanning {theme} ({etf})...")
            progress_bar.progress(i / total_steps)
            holdings = self.fetch_etf_holdings(etf)
            if not holdings:
                summary_stats.append({"Sector": theme, "ETF": etf, "Tickers Selected": 0, "Weight Pulled": 0.0, "Status": "❌ No Data"})
                continue
            holdings.sort(key=lambda x: x['weight'], reverse=True)
            selected = []; current_weight = 0.0
            for h in holdings:
                if current_weight >= target_cumulative_weight: break
                if max_tickers_per_sector and len(selected) >= max_tickers_per_sector: break
                selected.append(h)
                current_weight += h['weight']
            for s in selected:
                t = s['ticker']
                if t not in raw_candidates: raw_candidates[t] = []
                raw_candidates[t].append(theme)
            is_limited = (len(holdings) <= 10) and (current_weight < target_cumulative_weight - 0.01)
            summary_stats.append({"Sector": theme, "ETF": etf, "Tickers Selected": len(selected), "Weight Pulled": round(current_weight * 100, 1), "Status": "⚠️ Top 10 Limit" if is_limited else "✅ OK"})

        status_text.text("Validating Liquidity...")
        progress_bar.progress(0.9)
        unique_tickers = list(raw_candidates.keys())
        if not unique_tickers: return "", pd.DataFrame(summary_stats)
        
        try:
            import yfinance as yf
            valid_tickers = set()
            chunk_size = 100
            for i in range(0, len(unique_tickers), chunk_size):
                chunk = unique_tickers[i:i+chunk_size]
                df_vol = yf.download(chunk, period="5d", progress=False)
                if 'Close' in df_vol and 'Volume' in df_vol:
                     for t in chunk:
                        try:
                            if len(chunk) > 1: c, v = df_vol['Close'][t], df_vol['Volume'][t]
                            else: c, v = df_vol['Close'], df_vol['Volume']
                            if (c * v).mean() >= min_dollar_volume: valid_tickers.add(t)
                        except: continue     
        except Exception: valid_tickers = set(unique_tickers)

        final_rows = []
        for ticker, themes in raw_candidates.items():
            if ticker in valid_tickers:
                for theme in themes: final_rows.append({"Ticker": ticker, "Theme": theme, "Role": "Stock"})
        for theme, etf in theme_map.items(): final_rows.append({"Ticker": etf, "Theme": theme, "Role": "Etf"})
            
        df_final = pd.DataFrame(final_rows).sort_values(['Theme', 'Role', 'Ticker'])
        return df_final.to_csv(index=False), pd.DataFrame(summary_stats)

# ==========================================
# 9. COMPASS GENERATOR (PHASE 2 & 3)
# ==========================================
def generate_compass_data(etf_data_cache, theme_map):
    results = []
    etf_to_theme = {v: k for k, v in theme_map.items()}
    unique_etfs = list(set(theme_map.values()))
    for ticker in unique_etfs:
        if ticker not in etf_data_cache: continue
        df = etf_data_cache[ticker]
        if df is None or df.empty: continue
        d = df.copy().sort_index()
        variations = {
            'Logic_A_10d_3s': ('RRG_Ratio_Med',   'RRG_Mom_Med',   3),
            'Logic_B_10d_5s': ('RRG_Ratio_Med',   'RRG_Mom_Med',   5),
            'Logic_C_20d_3s': ('RRG_Ratio_Long',  'RRG_Mom_Long',  3),
            'Logic_D_20d_5s': ('RRG_Ratio_Long',  'RRG_Mom_Long',  5),
            'Logic_E_5d_2s':  ('RRG_Ratio_Short', 'RRG_Mom_Short', 2),
            'Logic_F_5d_3s':  ('RRG_Ratio_Short', 'RRG_Mom_Short', 3),
        }
        for label, (col_r, col_m, smooth) in variations.items():
            if col_r not in d.columns or col_m not in d.columns:
                d[f'{label}_Signal'] = 0; d[f'{label}_Streak'] = 0; continue
            prev_r_avg = d[col_r].shift(1).rolling(window=smooth).mean()
            prev_m_avg = d[col_m].shift(1).rolling(window=smooth).mean()
            is_bullish = (d[col_m] > prev_m_avg) & (d[col_r] > prev_r_avg)
            d[f'{label}_Signal'] = is_bullish.astype(int)
            group_id = (d[f'{label}_Signal'] != d[f'{label}_Signal'].shift()).cumsum()
            d[f'{label}_Streak'] = d.groupby(group_id).cumcount() + 1
            d.loc[d[f'{label}_Signal'] == 0, f'{label}_Streak'] = 0
        d['Target_1d'] = d['Close'].shift(-1) / d['Close'] - 1
        d['Target_3d'] = d['Close'].shift(-3) / d['Close'] - 1
        d['Target_5d'] = d['Close'].shift(-5) / d['Close'] - 1
        d['Target_10d'] = d['Close'].shift(-10) / d['Close'] - 1
        d['Target_20d'] = d['Close'].shift(-20) / d['Close'] - 1
        if 'Sma50' in d.columns: d['Context_Above50'] = (d['Close'] > d['Sma50']).astype(int)
        else: d['Context_Above50'] = 0
        cols_logic = [c for c in d.columns if 'Logic_' in c]
        cols_target = [c for c in d.columns if 'Target_' in c]
        export_chunk = d[['Close', 'Context_Above50'] + cols_target + cols_logic].copy()
        export_chunk['Ticker'] = ticker
        export_chunk['Theme'] = etf_to_theme.get(ticker, ticker)
        export_chunk.dropna(subset=['Target_1d'], inplace=True)
        results.append(export_chunk)
    if not results: return pd.DataFrame()
    return pd.concat(results)

def optimize_compass_settings(compass_df: pd.DataFrame) -> Tuple[Dict, str]:
    """
    Analyzes Compass Data to find the Logic with the highest 5-day Win Rate per Ticker.
    Returns Dictionary and saves it to JSON via save_dynamic_config().
    """
    if compass_df is None or compass_df.empty: return {}, "No data"
    results = []
    logics = [c.replace('_Signal', '') for c in compass_df.columns if '_Signal' in c]
    
    for ticker, group in compass_df.groupby('Ticker'):
        best_score = -999; best_logic = None
        for logic in logics:
            signal_col = f"{logic}_Signal"
            hits = group[group[signal_col] == 1]
            if len(hits) < 10: continue
            score = hits['Target_5d'].mean() # Metric: Average 5d Return
            if score > best_score: best_score = score; best_logic = logic
        if best_logic:
            win = 'Short' if '5d' in best_logic else 'Med' if '10d' in best_logic else 'Long'
            smooth = int(best_logic.split('_')[-1].replace('s', ''))
            results.append({'Ticker': ticker, 'Best_Logic': best_logic, 'Avg_Return_5d': best_score, 'Window': win, 'Smooth': smooth})
            
    if not results: return {}, "No valid results found."
    
    config_dict = {}
    for _, row in pd.DataFrame(results).iterrows():
        config_dict[row['Ticker']] = (row['Window'], row['Smooth'])
        
    # AUTO-SAVE TO JSON
    save_dynamic_config(config_dict)
    
    return config_dict, "Optimization Complete & Saved to 'sector_optimized_config.json'"

def plot_simple_rrg(data_cache, target_map, view_key, show_trails):
    if not target_map: return go.Figure()
    fig = go.Figure()
    all_x, all_y = [], []
    for theme, ticker in target_map.items():
        df = data_cache.get(ticker)
        if df is None or df.empty: continue
        col_x, col_y = f"RRG_Ratio_{view_key}", f"RRG_Mom_{view_key}"
        data_slice = df.tail(3) if show_trails else df.tail(1)
        if data_slice.empty: continue
        x_vals, y_vals = data_slice[col_x].tolist(), data_slice[col_y].tolist()
        all_x.extend(x_vals); all_y.extend(y_vals)
        last_x, last_y = x_vals[-1], y_vals[-1]
        color = '#00CC96' if last_x > 100 and last_y > 100 else '#636EFA' if last_x < 100 and last_y > 100 else '#FFA15A' if last_x > 100 else '#EF553B'
        n = len(x_vals)
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers+text', name=theme, text=[""]*(n-1)+[theme], customdata=[theme]*n,
            textposition="top center", marker=dict(size=[MARKER_SIZE_TRAIL]*(n-1)+[MARKER_SIZE_CURRENT], color=color, opacity=[TRAIL_OPACITY]*(n-1)+[CURRENT_OPACITY], line=dict(width=1, color='white')),
            line=dict(color=color, width=1 if show_trails else 0, shape='spline', smoothing=1.3), hoverinfo='text+name'))
    limit_x = max(max([abs(x - 100) for x in all_x]) * 1.1, 2.0) if all_x else 2.0
    limit_y = max(max([abs(y - 100) for y in all_y]) * 1.1, 2.0) if all_y else 2.0
    fig.add_hline(y=100, line_width=1, line_color="gray", line_dash="dash")
    fig.add_vline(x=100, line_width=1, line_color="gray", line_dash="dash")
    fig.update_layout(xaxis=dict(title="Relative Trend", showgrid=False, range=[100-limit_x, 100+limit_x]), 
                      yaxis=dict(title="Relative Momentum", showgrid=False, range=[100-limit_y, 100+limit_y]),
                      height=750, showlegend=False, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    return fig

# ==========================================
# 10. AI TRAINING DATA & EXPORTS (PHASE 4)
# ==========================================
def generate_ai_training_data(etf_ticker, etf_data_cache, stock_tickers, theme_name, benchmark_ticker):
    etf_df = etf_data_cache.get(etf_ticker)
    if etf_df is None or etf_df.empty: return pd.DataFrame()
    
    # We use basic cat calculation for training data export
    r_med = etf_df['RRG_Ratio_Med']
    m_med = etf_df['RRG_Mom_Med']
    avg_r_prev3 = r_med.shift(1).rolling(window=3).mean()
    avg_m_prev3 = m_med.shift(1).rolling(window=3).mean()
    is_outperf = r_med > avg_r_prev3
    is_gaining = m_med > avg_m_prev3
    cats = []
    for i in range(len(etf_df)):
        if pd.isna(avg_r_prev3.iloc[i]): cats.append("Unknown")
        else:
            p = "Outperforming" if is_outperf.iloc[i] else "Underperforming"
            m = "Gaining Momentum" if is_gaining.iloc[i] else "Losing Momentum"
            cats.append(f"{m} & {p}")
    etf_context = pd.DataFrame({'Theme_Category': cats}, index=etf_df.index)
    group_id = (etf_context['Theme_Category'] != etf_context['Theme_Category'].shift()).cumsum()
    etf_context['Days_In_Category'] = etf_context.groupby(group_id).cumcount() + 1
    etf_context.loc[etf_context['Theme_Category']=="Unknown", 'Days_In_Category'] = 0

    parent_df = etf_df
    windows = [5, 10, 15, 20, 30, 50]
    
    def process_stock_full_history(ticker):
        df = etf_data_cache.get(ticker)
        if df is None or df.empty: return None
        common_idx = df.index.intersection(etf_context.index)
        if common_idx.empty: return None
        df = df.loc[common_idx].copy()
        p_df = parent_df.loc[common_idx].copy()
        
        res = pd.DataFrame(index=df.index)
        res['Ticker'] = ticker; res['Theme'] = theme_name; res['Role'] = 'Stock'; res['Close'] = df['Close']
        res = res.join(etf_context)
        res['Target_FwdRet_1d'] = df['Close'].shift(-1) / df['Close'] - 1
        res['Target_FwdRet_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        res['Target_FwdRet_10d'] = df['Close'].shift(-10) / df['Close'] - 1
        
        if 'Volume' in df.columns:
            rvol_base = df['Volume'] / df['Volume'].rolling(window=20).mean()
            for w in windows: res[f'Metric_RVOL_{w}d'] = rvol_base.rolling(window=w).mean()
        
        stock_pct = df['Close'].pct_change()
        etf_pct = p_df['Close'].pct_change()
        rolling_cov = stock_pct.rolling(window=60).cov(etf_pct)
        rolling_var = etf_pct.rolling(window=60).var()
        beta = np.where(rolling_var > 1e-8, rolling_cov / rolling_var, 1.0)
        alpha_daily = stock_pct - (etf_pct * beta)
        for w in windows: res[f'Metric_Alpha_{w}d'] = alpha_daily.rolling(window=w).sum() * 100
        return res

    results = []
    for ticker in stock_tickers:
        res = process_stock_full_history(ticker)
        if res is not None: results.append(res)
    if not results: return pd.DataFrame()
    return pd.concat(results).fillna(0)

def generate_benchmark_export(ticker, etf_data_cache):
    df = etf_data_cache.get(ticker)
    if df is None or df.empty: return pd.DataFrame()
    export_df = df.copy()
    cols_to_drop = [c for c in export_df.columns if c.startswith('W_')]
    export_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return export_df
