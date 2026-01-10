"""
Sector rotation utilities - REFACTORED VERSION
Performance optimizations and multi-theme support added.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from io import StringIO
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

# Configure logging
logger = logging.getLogger(__name__)

# --- IMPORT SHARED UTILS ---
from utils_shared import get_gdrive_binary_data

# ==========================================
# 1. CONFIGURATION & CONSTANTS
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
MIN_DOLLAR_VOLUME = 2_000_000

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
        """
        Load sector universe from secrets.
        
        Args:
            benchmark_ticker: Benchmark to include (SPY or QQQ)
            
        Returns:
            Tuple of (universe_df, all_tickers, theme_to_etf_map)
        """
        secret_val = st.secrets.get("SECTOR_UNIVERSE", "")
        if not secret_val:
            st.error("❌ Secret 'SECTOR_UNIVERSE' is missing or empty.")
            return pd.DataFrame(), [], {}
            
        try:
            # Handle Google Sheet Links or Raw CSV
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
# 3. CALCULATOR & PIPELINE
# ==========================================
class SectorAlphaCalculator:
    """Calculates relative performance metrics for sectors and stocks."""
    
    def process_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare raw dataframe with technical indicators.
        
        Args:
            df: Raw price dataframe
            
        Returns:
            Processed dataframe with technicals, or None on error
        """
        if df is None or df.empty:
            return None

        # Ensure 'Close' exists
        if 'Close' not in df.columns and 'CLOSE' in df.columns:
            df['Close'] = df['CLOSE']
        
        if 'Close' not in df.columns:
            logger.error("No 'Close' column found in dataframe")
            return None
        
        # Calculate RVOL
        if 'Volume' in df.columns:
            avg_vol = df["Volume"].rolling(window=AVG_VOLUME_WINDOW).mean()
            df["RVOL"] = df["Volume"] / avg_vol

            for label, time_window in TIMEFRAMES.items():
                df[f"RVOL_{label}"] = df["RVOL"].rolling(window=time_window).mean()
        
        # Daily Range % (ADR)
        if 'High' in df.columns and 'Low' in df.columns:
            daily_range_pct = ((df['High'] - df['Low']) / df['Low']) * 100
            df['ADR_Pct'] = daily_range_pct.rolling(window=ATR_WINDOW).mean()
        
        return df

    def _calc_slope(self, series: pd.Series, window: int) -> pd.Series:
        """
        Calculate weighted linear regression slope.
        
        Args:
            series: Time series data
            window: Rolling window size
            
        Returns:
            Series of slopes
        """
        x = np.arange(window)
        weights = np.ones(window) if window < 10 else np.linspace(WEIGHT_MIN, WEIGHT_MAX, window)
        
        def slope_func(y):
            try:
                return np.polyfit(x, y, 1, w=weights)[0]
            except:
                return 0.0
                
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(
        self, 
        df: pd.DataFrame, 
        bench_df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Calculate Relative Rotation Graph metrics vs benchmark.
        
        Args:
            df: Asset dataframe with DatetimeIndex and Close column
            bench_df: Benchmark dataframe with same structure
            
        Returns:
            DataFrame with RRG columns added, or None on error
        """
        if df is None or df.empty or bench_df is None or bench_df.empty:
            logger.warning("Empty dataframe passed to calculate_rrg_metrics")
            return None
        
        # Align dates
        common_index = df.index.intersection(bench_df.index)
        
        if common_index.empty:
            logger.error(
                f"No date overlap. Asset: {df.index.min()} to {df.index.max()}, "
                f"Bench: {bench_df.index.min()} to {bench_df.index.max()}"
            )
            return None
        
        if len(common_index) < 50:
            logger.warning(f"Only {len(common_index)} overlapping dates - results may be unreliable")
        
        df_aligned = df.loc[common_index].copy()
        bench_aligned = bench_df.loc[common_index].copy()
        
        # Determine Close columns
        asset_col = 'Adj Close' if 'Adj Close' in df_aligned.columns else 'Close'
        bench_col = 'Adj Close' if 'Adj Close' in bench_df.columns else 'Close'
        
        # Relative Ratio
        raw_ratio = df_aligned[asset_col] / bench_aligned[bench_col]
        
        # RRG Logic for each timeframe
        for label, time_window in TIMEFRAMES.items():
            ratio_mean = raw_ratio.rolling(window=time_window).mean()
            
            # Trend (Ratio) - Normalized around 100
            col_ratio = f"RRG_Ratio_{label}"
            df_aligned[col_ratio] = ((raw_ratio - ratio_mean) / ratio_mean) * 100 + 100
            
            # Momentum (Velocity of the Ratio)
            raw_slope = self._calc_slope(raw_ratio, time_window)
            velocity = (raw_slope / ratio_mean) * time_window * 100
            
            col_mom = f"RRG_Mom_{label}"
            df_aligned[col_mom] = 100 + velocity
        
        # Merge back to original df
        for col in df_aligned.columns:
            if col.startswith('RRG_'):
                df[col] = df_aligned[col]
                
        return df

    def calculate_stock_alpha_multi_theme(
        self, 
        df: pd.DataFrame, 
        parent_df: pd.DataFrame, 
        theme_suffix: str
    ) -> pd.DataFrame:
        """
        Calculate beta and alpha with theme-specific column names.
        
        This allows a stock to have multiple betas (one per theme).
        
        Args:
            df: Stock dataframe
            parent_df: Theme/sector ETF dataframe
            theme_suffix: Theme name for column naming (e.g., "Technology")
            
        Returns:
            DataFrame with added columns:
            - Beta_{theme}
            - Alpha_Short_{theme}
            - Alpha_Med_{theme}
            - Alpha_Long_{theme}
        """
        if df is None or df.empty or parent_df is None or parent_df.empty:
            logger.warning(f"Empty dataframe in alpha calculation for theme {theme_suffix}")
            return df
        
        common_index = df.index.intersection(parent_df.index)
        
        if common_index.empty:
            logger.error(f"No date overlap for theme {theme_suffix}")
            return df
        
        if len(common_index) < BETA_WINDOW:
            logger.warning(
                f"Only {len(common_index)} overlapping dates for {theme_suffix} "
                f"(need {BETA_WINDOW} for beta calculation)"
            )
        
        df_aligned = df.loc[common_index].copy()
        parent_aligned = parent_df.loc[common_index].copy()
        
        # Calculate returns if not already present
        if 'Pct_Change' not in df_aligned.columns:
            df_aligned['Pct_Change'] = df_aligned['Close'].pct_change()
        if 'Pct_Change' not in parent_aligned.columns:
            parent_aligned['Pct_Change'] = parent_aligned['Close'].pct_change()
        
        # Beta calculation with theme-specific name
        rolling_cov = df_aligned['Pct_Change'].rolling(window=BETA_WINDOW).cov(
            parent_aligned['Pct_Change']
        )
        rolling_var = parent_aligned['Pct_Change'].rolling(window=BETA_WINDOW).var()
        
        beta_col = f"Beta_{theme_suffix}"
        df_aligned[beta_col] = np.where(
            rolling_var > 1e-8,  # Avoid division by zero
            rolling_cov / rolling_var,
            1.0  # Default to market beta
        )
        
        # Expected return (CAPM)
        expected_return_col = f"Expected_Return_{theme_suffix}"
        df_aligned[expected_return_col] = parent_aligned['Pct_Change'] * df_aligned[beta_col]
        
        # Alpha (actual - expected)
        alpha_1d_col = f"Alpha_1D_{theme_suffix}"
        df_aligned[alpha_1d_col] = df_aligned['Pct_Change'] - df_aligned[expected_return_col]
        
        # Multi-timeframe cumulative alpha
        for label, time_window in TIMEFRAMES.items():
            alpha_col = f"Alpha_{label}_{theme_suffix}"
            df_aligned[alpha_col] = (
                df_aligned[alpha_1d_col]
                .fillna(0)
                .rolling(window=time_window)
                .sum() * 100
            )
        
        # Merge back to original df (preserving all dates)
        for col in df_aligned.columns:
            if col.endswith(f"_{theme_suffix}") or col == beta_col:
                df[col] = df_aligned[col]
        
        return df

# ==========================================
# 4. PATTERN DETECTION
# ==========================================

def detect_dip_buy_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    """
    Detect stocks that were outperforming but pulled back to average.
    
    Criteria:
    - 20-day alpha was significantly positive
    - 5-day alpha is near zero
    - Stock above 21-EMA (trend intact)
    
    Args:
        df: Stock dataframe with alpha columns
        theme_suffix: Theme name for column lookup
        
    Returns:
        True if stock matches dip buy pattern
    """
    if df is None or df.empty:
        return False
    
    try:
        last = df.iloc[-1]
        
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        
        # Historical outperformance
        was_strong = alpha_20d > ALPHA_DIP_BUY_THRESHOLD
        
        # Recent pullback to average
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_5d <= ALPHA_NEUTRAL_RANGE
        
        # Trend still intact
        price = last.get('Close', 0)
        ema21 = last.get('Ema21', 0)
        trend_intact = price > ema21 if ema21 > 0 else False
        
        return was_strong and now_neutral and trend_intact
    except Exception as e:
        logger.error(f"Error in dip_buy detection: {e}")
        return False

def detect_breakout_candidates(
    df: pd.DataFrame, 
    theme_suffix: str
) -> Optional[Dict]:
    """
    Detect stocks transitioning from underperformance to outperformance.
    
    Criteria:
    - 20-day alpha was negative
    - 10-day alpha neutral
    - 5-day alpha positive
    - Volume increasing
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        
    Returns:
        Dict with pattern details or None
    """
    if df is None or df.empty:
        return None
    
    try:
        last = df.iloc[-1]
        
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        rvol = last.get('RVOL_Short', 0)
        
        # Classic acceleration pattern
        was_lagging = alpha_20d < -ALPHA_BREAKOUT_THRESHOLD
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_10d <= ALPHA_NEUTRAL_RANGE
        now_leading = alpha_5d > ALPHA_BREAKOUT_THRESHOLD
        volume_confirms = rvol > RVOL_BREAKOUT_THRESHOLD
        
        if was_lagging and now_neutral and now_leading and volume_confirms:
            alpha_delta = alpha_5d - alpha_20d
            strength = min(100, (alpha_delta / 5.0) * 100)
            
            return {
                'pattern': 'breakout',
                'strength': strength,
                'alpha_change': alpha_delta
            }
    except Exception as e:
        logger.error(f"Error in breakout detection: {e}")
        
    return None

def detect_fading_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    """
    Detect stocks losing their alpha advantage.
    
    Criteria:
    - 20-day alpha very positive
    - 5-day alpha declining but still positive
    - Alpha slope negative
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        
    Returns:
        True if stock is fading
    """
    if df is None or df.empty or len(df) < 10:
        return False
    
    try:
        last = df.iloc[-1]
        prev_5 = df.iloc[-6]
        
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d_now = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_5d_prev = prev_5.get(f"Alpha_Short_{theme_suffix}", 0)
        
        price = last.get('Close', 0)
        sma50 = last.get('Sma50', 0)
        
        # Was very strong
        was_very_strong = alpha_20d > ALPHA_FADING_THRESHOLD
        
        # Still positive but declining
        still_positive = 0 < alpha_5d_now < ALPHA_DIP_BUY_THRESHOLD
        declining = alpha_5d_now < alpha_5d_prev
        
        # Hasn't broken down yet
        not_broken = price > sma50 if sma50 > 0 else True
        
        return was_very_strong and still_positive and declining and not_broken
    except Exception as e:
        logger.error(f"Error in fader detection: {e}")
        return False

def detect_relative_strength_divergence(
    df: pd.DataFrame, 
    theme_suffix: str, 
    lookback: int = 20
) -> Optional[str]:
    """
    Detect divergence between price and alpha.
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        lookback: Days to look back
        
    Returns:
        'bullish_divergence', 'bearish_divergence', or None
    """
    if df is None or df.empty or len(df) < lookback:
        return None
    
    try:
        recent = df.tail(lookback)
        alpha_col = f"Alpha_Short_{theme_suffix}"
        
        if alpha_col not in recent.columns:
            return None
        
        # Find extremes
        price_high = recent['Close'].max()
        price_low = recent['Close'].min()
        alpha_high = recent[alpha_col].max()
        alpha_low = recent[alpha_col].min()
        
        # Current values
        current_price = recent['Close'].iloc[-1]
        current_alpha = recent[alpha_col].iloc[-1]
        
        # Bearish Divergence: Price high, alpha declining
        if current_price >= price_high * 0.98:  # Near high
            if current_alpha < alpha_high * 0.7:  # Alpha well off highs
                return 'bearish_divergence'
        
        # Bullish Divergence: Price low, alpha improving
        if current_price <= price_low * 1.05:  # Near lows
            if current_alpha > alpha_low * 1.3:  # Alpha recovering
                return 'bullish_divergence'
    except Exception as e:
        logger.error(f"Error in divergence detection: {e}")
    
    return None

def calculate_comprehensive_stock_score(
    df: pd.DataFrame, 
    theme_suffix: str, 
    theme_quadrant: str
) -> Optional[Dict]:
    """
    Multi-factor scoring system for stock selection.
    
    Factors:
    1. Alpha trajectory (40%)
    2. Volume confirmation (20%)
    3. Technical position (20%)
    4. Theme alignment (20%)
    
    Args:
        df: Stock dataframe
        theme_suffix: Theme name
        theme_quadrant: Current quadrant of theme (e.g., "🟢 Leading")
        
    Returns:
        Dict with score and breakdown
    """
    if df is None or df.empty:
        return None
    
    try:
        last = df.iloc[-1]
        score_breakdown = {}
        
        # --- FACTOR 1: Alpha Trajectory (40 points) ---
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        
        # Consistency (15 points)
        if alpha_5d > 0 and alpha_10d > 0 and alpha_20d > 0:
            score_breakdown['alpha_consistency'] = 15
        elif alpha_5d > 0 and alpha_10d > 0:
            score_breakdown['alpha_consistency'] = 10
        elif alpha_5d > 0:
            score_breakdown['alpha_consistency'] = 5
        else:
            score_breakdown['alpha_consistency'] = 0
        
        # Acceleration (15 points)
        if alpha_5d > alpha_10d > alpha_20d:
            score_breakdown['alpha_acceleration'] = 15
        elif alpha_5d > alpha_10d:
            score_breakdown['alpha_acceleration'] = 10
        elif alpha_5d > alpha_20d:
            score_breakdown['alpha_acceleration'] = 5
        else:
            score_breakdown['alpha_acceleration'] = 0
        
        # Magnitude (10 points)
        max_alpha = max(alpha_5d, alpha_10d, alpha_20d)
        if max_alpha > 5.0:
            score_breakdown['alpha_magnitude'] = 10
        elif max_alpha > 2.0:
            score_breakdown['alpha_magnitude'] = 7
        elif max_alpha > 1.0:
            score_breakdown['alpha_magnitude'] = 4
        else:
            score_breakdown['alpha_magnitude'] = 0
        
        # --- FACTOR 2: Volume Confirmation (20 points) ---
        rvol_5d = last.get('RVOL_Short', 0)
        rvol_10d = last.get('RVOL_Med', 0)
        
        if rvol_5d > 1.5:
            score_breakdown['volume'] = 20
        elif rvol_5d > 1.2:
            score_breakdown['volume'] = 15
        elif rvol_10d > 1.2:
            score_breakdown['volume'] = 10
        else:
            score_breakdown['volume'] = 0
        
        # --- FACTOR 3: Technical Position (20 points) ---
        price = last.get('Close', 0)
        ema8 = last.get('Ema8', 0)
        ema21 = last.get('Ema21', 0)
        sma50 = last.get('Sma50', 0)
        sma200 = last.get('Sma200', 0)
        
        ma_score = 0
        if price > ema8 and ema8 > 0: ma_score += 5
        if price > ema21 and ema21 > 0: ma_score += 5
        if price > sma50 and sma50 > 0: ma_score += 5
        if price > sma200 and sma200 > 0: ma_score += 5
        
        score_breakdown['technical'] = ma_score
        
        # --- FACTOR 4: Theme Alignment (20 points) ---
        if "Leading" in theme_quadrant:
            score_breakdown['theme_sync'] = 20 if alpha_5d > 1.0 else (10 if alpha_5d > 0 else 0)
        elif "Improving" in theme_quadrant:
            score_breakdown['theme_sync'] = 20 if alpha_5d > alpha_20d else 5
        elif "Weakening" in theme_quadrant:
            score_breakdown['theme_sync'] = 0 if alpha_5d < 0 else 10
        else:  # Lagging
            score_breakdown['theme_sync'] = 0
        
        # --- CALCULATE TOTAL ---
        total_score = sum(score_breakdown.values())
        
        # Pattern bonuses
        if detect_breakout_candidates(df, theme_suffix):
            score_breakdown['pattern_bonus'] = 10
            total_score += 10
        
        if detect_dip_buy_candidates(df, theme_suffix):
            score_breakdown['dip_buy_bonus'] = 5
            total_score += 5
        
        divergence = detect_relative_strength_divergence(df, theme_suffix)
        if divergence == 'bullish_divergence':
            score_breakdown['divergence_bonus'] = 5
            total_score += 5
        elif divergence == 'bearish_divergence':
            score_breakdown['divergence_penalty'] = -10
            total_score -= 10
        
        # Cap score at 0-100
        total_score = min(100, max(0, total_score))
        
        return {
            'total_score': total_score,
            'breakdown': score_breakdown,
            'grade': _score_to_grade(total_score)
        }
    except Exception as e:
        logger.error(f"Error calculating score: {e}")
        return None

def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 80: return 'A'
    if score >= 70: return 'B'
    if score >= 60: return 'C'
    if score >= 50: return 'D'
    return 'F'

# ==========================================
# 5. ORCHESTRATOR (OPTIMIZED)
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_process_universe(
    benchmark_ticker: str = "SPY"
) -> Tuple[Dict, List[str], Dict[str, str], pd.DataFrame, Dict[str, List[str]]]:
    """
    OPTIMIZED data pipeline with multi-theme support.
    
    Key optimizations:
    1. Single parquet download
    2. Vectorized filtering with groupby
    3. Multi-theme beta calculations
    4. Efficient caching
    
    Args:
        benchmark_ticker: SPY or QQQ
        
    Returns:
        Tuple of (data_cache, missing_tickers, theme_map, universe_df, stock_themes_map)
    """
    dm = SectorDataManager()
    uni_df, tickers, theme_map = dm.load_universe(benchmark_ticker)
    
    if uni_df.empty:
        return {}, ["SECTOR_UNIVERSE is empty"], theme_map, uni_df, {}

    # --- 1. DOWNLOAD MASTER DB ---
    db_url = st.secrets.get("PARQUET_SECTOR_ROTATION")
    if not db_url:
        st.error("⛔ Secret 'PARQUET_SECTOR_ROTATION' is missing.")
        return {}, ["PARQUET_SECTOR_ROTATION secret missing"], theme_map, uni_df, {}

    try:
        buffer = get_gdrive_binary_data(db_url)
        if not buffer:
            logger.error("Failed to download sector rotation parquet")
            return {}, ["Failed to download Master DB"], theme_map, uni_df, {}
        
        master_df = pd.read_parquet(buffer)
        logger.info(f"Loaded master parquet with {len(master_df)} rows")
        
    except Exception as e:
        logger.exception(f"Error reading parquet: {e}")
        return {}, [f"Error reading Parquet file: {e}"], theme_map, uni_df, {}

    # --- 2. STANDARDIZE MASTER DB ---
    master_df.columns = [c.strip().title() for c in master_df.columns]
    
    if 'Symbol' in master_df.columns and 'Ticker' not in master_df.columns:
        master_df.rename(columns={'Symbol': 'Ticker'}, inplace=True)
        
    if 'Date' in master_df.columns:
        master_df['Date'] = pd.to_datetime(master_df['Date'])
        master_df = master_df.set_index('Date').sort_index()
    elif isinstance(master_df.index, pd.DatetimeIndex):
        master_df = master_df.sort_index()
    
    if 'Ticker' in master_df.columns:
        master_df['Ticker'] = master_df['Ticker'].astype(str).str.upper().str.strip()
    else:
        return {}, ["Critical: 'Ticker' column missing in Master DB"], theme_map, uni_df, {}

    calc = SectorAlphaCalculator()
    data_cache = {}
    missing_tickers = []

    # --- 3. PROCESS BENCHMARK ---
    bench_df = master_df[master_df['Ticker'] == benchmark_ticker].copy()
    
    if bench_df.empty:
        logger.error(f"Benchmark {benchmark_ticker} not found in parquet")
        return {}, [f"Benchmark '{benchmark_ticker}' not found"], theme_map, uni_df, {}

    bench_df = calc.process_dataframe(bench_df)
    data_cache[benchmark_ticker] = bench_df
    logger.info(f"Processed benchmark {benchmark_ticker}")

    # --- 4. PROCESS ETFS (OPTIMIZED WITH GROUPBY) ---
    etf_tickers = list(theme_map.values())
    
    if etf_tickers:
        etf_mask = master_df['Ticker'].isin(etf_tickers)
        etf_groups = master_df[etf_mask].groupby('Ticker')
        
        for etf in etf_tickers:
            try:
                df = etf_groups.get_group(etf).copy()
            except KeyError:
                missing_tickers.append(etf)
                logger.warning(f"ETF {etf} not found in parquet")
                continue
                
            df = calc.process_dataframe(df)
            df = calc.calculate_rrg_metrics(df, bench_df)
            data_cache[etf] = df
        
        logger.info(f"Processed {len(etf_tickers)} ETFs")

    # --- 5. PROCESS STOCKS (MULTI-THEME AWARE) ---
    stocks = uni_df[uni_df['Role'] == 'Stock']
    
    # Build stock-to-themes mapping
    stock_themes_map = stocks.groupby('Ticker')['Theme'].apply(list).to_dict()
    logger.info(f"Found {len(stock_themes_map)} unique stocks across themes")
    
    # Get all unique stock tickers
    stock_tickers = list(stock_themes_map.keys())
    
    if stock_tickers:
        # OPTIMIZATION: Filter and group once
        stock_mask = master_df['Ticker'].isin(stock_tickers)
        stock_groups = master_df[stock_mask].groupby('Ticker')
        
        for ticker, themes_list in stock_themes_map.items():
            try:
                df = stock_groups.get_group(ticker).copy()
            except KeyError:
                missing_tickers.append(ticker)
                logger.warning(f"Stock {ticker} not found in parquet")
                continue
            
            # Process base dataframe once
            df = calc.process_dataframe(df)
            
            # Calculate beta/alpha vs EACH theme
            for theme in themes_list:
                parent_etf = theme_map.get(theme)
                parent_df = data_cache.get(parent_etf, bench_df)
                
                if parent_df is not None:
                    df = calc.calculate_stock_alpha_multi_theme(df, parent_df, theme)
            
            data_cache[ticker] = df
        
        logger.info(f"Processed {len(stock_tickers)} stocks with multi-theme support")

    return data_cache, missing_tickers, theme_map, uni_df, stock_themes_map


# ==========================================
# 6. VISUALIZATION & CLASSIFICATION
# ==========================================

def classify_setup(df: pd.DataFrame) -> Optional[str]:
    """
    Classify sector/theme setup pattern.
    
    Patterns:
    - J-Hook: Turnaround from weak to strong
    - Bull Flag: Strong trend with breakout
    - Rocket: Parabolic acceleration
    - Breakdown: Accelerating downward
    - Dead Cat: Weak bounce in downtrend
    - Power Trend: Sustained strength
    - Deteriorating: Losing momentum
    
    Args:
        df: Theme dataframe with RRG metrics
        
    Returns:
        Pattern string or None
    """
    if df is None or df.empty:
        return None
        
    try:
        last = df.iloc[-1]
        
        if "RRG_Mom_Short" not in last or "RRG_Mom_Long" not in last:
            return None

        m5 = last["RRG_Mom_Short"]
        m10 = last.get("RRG_Mom_Med", 0)
        m20 = last["RRG_Mom_Long"]
        ratio_20 = last.get("RRG_Ratio_Long", 100)

        # J-Hook: Turnaround play
        if m20 < 100 and m5 > 100 and m5 > (m20 + JHOOK_MIN_SHIFT):
            return "🪝 J-Hook"
        
        # Bull Flag: Continuation
        if ratio_20 > 100 and m5 > 100 and m5 > m10:
            return "🚩 Bull Flag"
        
        # Rocket: Parabolic
        if m5 > m10 > m20 and m20 > 100:
            return "🚀 Rocket"
        
        # Breakdown: Accelerating down
        if m5 < m10 < m20 and m20 < 100:
            return "💥 Breakdown"
        
        # Dead Cat Bounce: Weak bounce
        if m20 < 100 and m5 > 100 and m5 < (m20 + JHOOK_MIN_SHIFT):
            return "🐱 Dead Cat"
        
        # Power Trend: Sustained strength
        if ratio_20 > 100 and m20 > 100 and m10 > 100 and m5 > 100:
            return "⚡ Power"
        
        # Deteriorating: Losing steam
        if ratio_20 > 100 and m5 < 100 and m5 < m10:
            return "📉 Fading"
            
    except Exception as e:
        logger.error(f"Error in classify_setup: {e}")
    
    return None

def get_quadrant_status(df: pd.DataFrame, timeframe_key: str) -> str:
    """
    Get quadrant status for a theme.
    
    Args:
        df: Theme dataframe with RRG metrics
        timeframe_key: 'Short', 'Med', or 'Long'
        
    Returns:
        Quadrant status string
    """
    if df is None or df.empty:
        return "N/A"
    
    try:
        col_r = f"RRG_Ratio_{timeframe_key}"
        col_m = f"RRG_Mom_{timeframe_key}"
        
        if col_r not in df.columns or col_m not in df.columns:
            return "N/A"
        
        r = df[col_r].iloc[-1]
        m = df[col_m].iloc[-1]
        
        if r >= 100 and m >= 100:
            return "🟢 Leading"
        elif r < 100 and m >= 100:
            return "🔵 Improving"
        elif r < 100 and m < 100:
            return "🔴 Lagging"
        else:
            return "🟡 Weakening"
    except Exception as e:
        logger.error(f"Error getting quadrant status: {e}")
        return "N/A"

def plot_simple_rrg(
    data_cache: Dict, 
    target_map: Dict, 
    view_key: str, 
    show_trails: bool
) -> go.Figure:
    """
    Create Relative Rotation Graph scatter plot.
    
    Args:
        data_cache: Dict of {ticker: DataFrame}
        target_map: Dict of {theme: ticker}
        view_key: 'Short', 'Med', or 'Long'
        show_trails: Whether to show 3-day trails
        
    Returns:
        Plotly Figure object
    """
    # Validate view_key
    valid_keys = ['Short', 'Med', 'Long']
    if view_key not in valid_keys:
        logger.error(f"Invalid view_key: {view_key}")
        raise ValueError(f"view_key must be one of {valid_keys}, got: {view_key}")
    
    if not target_map:
        logger.warning("Empty target_map in plot_simple_rrg")
        fig = go.Figure()
        fig.add_annotation(
            text="No sectors selected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = go.Figure()
    all_x, all_y = [], []
    
    for theme, ticker in target_map.items():
        df = data_cache.get(ticker)
        if df is None or df.empty:
            continue
        
        col_x = f"RRG_Ratio_{view_key}"
        col_y = f"RRG_Mom_{view_key}"
        
        if col_x not in df.columns or col_y not in df.columns:
            continue
        
        data_slice = df.tail(3) if show_trails else df.tail(1)
        if data_slice.empty:
            continue

        x_vals = data_slice[col_x].tolist()
        y_vals = data_slice[col_y].tolist()
        all_x.extend(x_vals)
        all_y.extend(y_vals)
        
        # Determine color based on quadrant
        last_x, last_y = x_vals[-1], y_vals[-1]
        if last_x > 100 and last_y > 100:
            color = '#00CC96'  # Green
        elif last_x < 100 and last_y > 100:
            color = '#636EFA'  # Blue
        elif last_x > 100 and last_y < 100:
            color = '#FFA15A'  # Orange
        else:
            color = '#EF553B'  # Red
        
        # Marker sizes and opacities
        n = len(x_vals)
        sizes = [MARKER_SIZE_TRAIL] * (n - 1) + [MARKER_SIZE_CURRENT]
        opacities = [TRAIL_OPACITY] * (n - 1) + [CURRENT_OPACITY]
        texts = [""] * (n - 1) + [theme]
        custom_data = [theme] * n

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers+text',
            name=theme,
            text=texts,
            customdata=custom_data,
            textposition="top center",
            marker=dict(
                size=sizes,
                color=color,
                opacity=opacities,
                line=dict(width=1, color='white')
            ),
            line=dict(
                color=color,
                width=1 if show_trails else 0,
                shape='spline',
                smoothing=1.3
            ),
            hoverinfo='text+name',
            hovertext=[
                f"{theme}<br>Trend: {x:.1f}<br>Mom: {y:.1f}"
                for x, y in zip(x_vals, y_vals)
            ]
        ))

    # Dynamic scaling
    if all_x and all_y:
        limit_x = max(max([abs(x - 100) for x in all_x]) * 1.1, 2.0)
        limit_y = max(max([abs(y - 100) for y in all_y]) * 1.1, 2.0)
        x_range = [100 - limit_x, 100 + limit_x]
        y_range = [100 - limit_y, 100 + limit_y]
    else:
        x_range, y_range = [98, 102], [98, 102]
        limit_x, limit_y = 2, 2

    # Add crosshairs
    fig.add_hline(y=100, line_width=1, line_color="gray", line_dash="dash")
    fig.add_vline(x=100, line_width=1, line_color="gray", line_dash="dash")
    
    # Add quadrant labels
    lbl_x, lbl_y = limit_x * 0.5, limit_y * 0.5
    
    fig.add_annotation(
        x=100+lbl_x, y=100+lbl_y,
        text="<b>LEADING</b>",
        showarrow=False,
        font=dict(color="rgba(0, 255, 0, 0.7)", size=20)
    )
    fig.add_annotation(
        x=100-lbl_x, y=100+lbl_y,
        text="<b>IMPROVING</b>",
        showarrow=False,
        font=dict(color="rgba(0, 100, 255, 0.7)", size=20)
    )
    fig.add_annotation(
        x=100+lbl_x, y=100-lbl_y,
        text="<b>WEAKENING</b>",
        showarrow=False,
        font=dict(color="rgba(255, 165, 0, 0.7)", size=20)
    )
    fig.add_annotation(
        x=100-lbl_x, y=100-lbl_y,
        text="<b>LAGGING</b>",
        showarrow=False,
        font=dict(color="rgba(255, 0, 0, 0.7)", size=20)
    )

    fig.update_layout(
        xaxis=dict(
            title="Relative Trend",
            showgrid=False,
            range=x_range,
            constrain='domain'
        ),
        yaxis=dict(
            title="Relative Momentum",
            showgrid=False,
            range=y_range
        ),
        height=750,
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig
