"""
Sector rotation utilities - REFACTORED VERSION
Performance optimizations, multi-theme support, and centralized logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from io import StringIO
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logger = logging.getLogger(__name__)

# --- IMPORT SHARED UTILS ---
import utils_darcy as ud

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
# 3. HELPERS (NEWLY ADDED)
# ==========================================
def get_ma_signal(price: float, ma_val: float) -> str:
    """Return emoji based on price vs moving average."""
    if pd.isna(ma_val) or ma_val == 0:
        return "⚠️"
    return "✅" if price > ma_val else "❌"

def _shorten_category_name(name: str) -> str:
    """Helper to shorten category names for UI display."""
    return name.replace("Gaining Momentum", "Gain Mom") \
               .replace("Losing Momentum", "Lose Mom") \
               .replace("Outperforming", "Outperf") \
               .replace("Underperforming", "Underperf")

# ==========================================
# 4. CALCULATOR & PIPELINE
# ==========================================
class SectorAlphaCalculator:
    """Calculates relative performance metrics for sectors and stocks."""
    
    def process_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty: return None

        if 'Close' not in df.columns and 'CLOSE' in df.columns:
            df['Close'] = df['CLOSE']
        
        if 'Close' not in df.columns:
            logger.error("No 'Close' column found in dataframe")
            return None
        
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
            try:
                return np.polyfit(x, y, 1, w=weights)[0]
            except:
                return 0.0
                
        return series.rolling(window=window).apply(slope_func, raw=True)

    def calculate_rrg_metrics(self, df: pd.DataFrame, bench_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty or bench_df is None or bench_df.empty:
            logger.warning("Empty dataframe passed to calculate_rrg_metrics")
            return None
        
        common_index = df.index.intersection(bench_df.index)
        
        if common_index.empty:
            return None
        
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
            if col.startswith('RRG_'):
                df[col] = df_aligned[col]
                
        return df

    def calculate_stock_alpha_multi_theme(self, df: pd.DataFrame, parent_df: pd.DataFrame, theme_suffix: str) -> pd.DataFrame:
        if df is None or df.empty or parent_df is None or parent_df.empty:
            return df
        
        common_index = df.index.intersection(parent_df.index)
        if common_index.empty:
            return df
        
        df_aligned = df.loc[common_index].copy()
        parent_aligned = parent_df.loc[common_index].copy()
        
        if 'Pct_Change' not in df_aligned.columns:
            df_aligned['Pct_Change'] = df_aligned['Close'].pct_change()
        if 'Pct_Change' not in parent_aligned.columns:
            parent_aligned['Pct_Change'] = parent_aligned['Close'].pct_change()
        
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
            if col.endswith(f"_{theme_suffix}") or col == beta_col:
                df[col] = df_aligned[col]
        
        return df

# ==========================================
# 5. PATTERN DETECTION (PRESERVED)
# ==========================================

def detect_dip_buy_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    if df is None or df.empty: return False
    try:
        last = df.iloc[-1]
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        was_strong = alpha_20d > ALPHA_DIP_BUY_THRESHOLD
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_5d <= ALPHA_NEUTRAL_RANGE
        price = last.get('Close', 0)
        ema21 = last.get('Ema21', 0)
        trend_intact = price > ema21 if ema21 > 0 else False
        return was_strong and now_neutral and trend_intact
    except Exception as e:
        logger.error(f"Error in dip_buy detection: {e}")
        return False

def detect_breakout_candidates(df: pd.DataFrame, theme_suffix: str) -> Optional[Dict]:
    if df is None or df.empty: return None
    try:
        last = df.iloc[-1]
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        rvol = last.get('RVOL_Short', 0)
        
        was_lagging = alpha_20d < -ALPHA_BREAKOUT_THRESHOLD
        now_neutral = -ALPHA_NEUTRAL_RANGE <= alpha_10d <= ALPHA_NEUTRAL_RANGE
        now_leading = alpha_5d > ALPHA_BREAKOUT_THRESHOLD
        volume_confirms = rvol > RVOL_BREAKOUT_THRESHOLD
        
        if was_lagging and now_neutral and now_leading and volume_confirms:
            alpha_delta = alpha_5d - alpha_20d
            strength = min(100, (alpha_delta / 5.0) * 100)
            return {'pattern': 'breakout', 'strength': strength, 'alpha_change': alpha_delta}
    except Exception as e:
        logger.error(f"Error in breakout detection: {e}")
    return None

def detect_fading_candidates(df: pd.DataFrame, theme_suffix: str) -> bool:
    if df is None or df.empty or len(df) < 10: return False
    try:
        last = df.iloc[-1]
        prev_5 = df.iloc[-6]
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        alpha_5d_now = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_5d_prev = prev_5.get(f"Alpha_Short_{theme_suffix}", 0)
        price = last.get('Close', 0)
        sma50 = last.get('Sma50', 0)
        
        was_very_strong = alpha_20d > ALPHA_FADING_THRESHOLD
        still_positive = 0 < alpha_5d_now < ALPHA_DIP_BUY_THRESHOLD
        declining = alpha_5d_now < alpha_5d_prev
        not_broken = price > sma50 if sma50 > 0 else True
        return was_very_strong and still_positive and declining and not_broken
    except Exception as e:
        logger.error(f"Error in fader detection: {e}")
        return False

def detect_relative_strength_divergence(df: pd.DataFrame, theme_suffix: str, lookback: int = 20) -> Optional[str]:
    if df is None or df.empty or len(df) < lookback: return None
    try:
        recent = df.tail(lookback)
        alpha_col = f"Alpha_Short_{theme_suffix}"
        if alpha_col not in recent.columns: return None
        
        price_high = recent['Close'].max()
        price_low = recent['Close'].min()
        alpha_high = recent[alpha_col].max()
        alpha_low = recent[alpha_col].min()
        
        current_price = recent['Close'].iloc[-1]
        current_alpha = recent[alpha_col].iloc[-1]
        
        if current_price >= price_high * 0.98 and current_alpha < alpha_high * 0.7:
            return 'bearish_divergence'
        if current_price <= price_low * 1.05 and current_alpha > alpha_low * 1.3:
            return 'bullish_divergence'
    except Exception as e:
        logger.error(f"Error in divergence detection: {e}")
    return None

def _score_to_grade(score: float) -> str:
    if score >= 80: return 'A'
    if score >= 70: return 'B'
    if score >= 60: return 'C'
    if score >= 50: return 'D'
    return 'F'

def calculate_comprehensive_stock_score(df: pd.DataFrame, theme_suffix: str, theme_quadrant: str, sector_score: float = 70, sector_stage: str = "Established") -> Optional[Dict]:
    """Calculates comprehensive stock score based on setup, timing, and risk/reward."""
    if df is None or df.empty or len(df) < 20: return None
    try:
        last = df.iloc[-1]
        score_breakdown = {}
        
        alpha_5d = last.get(f"Alpha_Short_{theme_suffix}", 0)
        alpha_10d = last.get(f"Alpha_Med_{theme_suffix}", 0)
        alpha_20d = last.get(f"Alpha_Long_{theme_suffix}", 0)
        
        price = last.get('Close', 0)
        ema21 = last.get('Ema21', 0)
        sma50 = last.get('Sma50', 0)
        sma200 = last.get('Sma200', 0)
        rvol_5d = last.get('RVOL_Short', 0)
        rvol_10d = last.get('RVOL_Med', 0)
        rvol_20d = last.get('RVOL_Long', 0)
        
        days_positive = 0
        for i in range(min(20, len(df))):
            row_alpha = df.iloc[-(i+1)].get(f"Alpha_Short_{theme_suffix}", 0)
            if row_alpha > 0: days_positive += 1
        
        # Setup Quality
        setup_score = 0
        setup_reason = ""
        if alpha_5d > 0 and sector_score >= 60:
            setup_score += 12; setup_reason = "Positive in decent sector"
        elif alpha_5d > 0:
            setup_score += 8; setup_reason = "Positive alpha"
        if 0 <= alpha_5d <= 3.0:
            setup_score += 15; setup_reason = "Fresh entry range (0-3% alpha)"
            if days_positive <= 7: setup_score += 5; setup_reason += ", very fresh"
        if alpha_5d > alpha_10d and alpha_10d > alpha_20d:
            setup_score += 10; setup_reason += ", accelerating"
        elif alpha_5d > alpha_20d:
            setup_score += 6; setup_reason += ", improving"
        if alpha_5d > 5.0:
            setup_score -= 15; setup_reason = "Extended (>5% alpha)"
        elif alpha_5d > 3.5 and days_positive >= 15:
            setup_score -= 10; setup_reason = "Extended (mature + high alpha)"
        score_breakdown['setup_quality'] = max(0, min(40, setup_score))
        
        # Timing
        timing_score = 0
        dist_from_ema21 = (price - ema21) / ema21 if ema21 > 0 else 999
        dist_from_sma50 = (price - sma50) / sma50 if sma50 > 0 else 999
        if price > sma200 and sma200 > 0: timing_score += 5
        if abs(dist_from_ema21) <= 0.03 and price > sma200 and sma200 > 0: timing_score += 10
        elif 0 < dist_from_sma50 <= 0.08 and rvol_5d >= 1.2: timing_score += 10
        elif -0.05 <= dist_from_ema21 <= 0.08: timing_score += 5
        elif dist_from_ema21 > 0.12 or dist_from_sma50 > 0.18: timing_score -= 10
        score_breakdown['timing_price'] = timing_score
        
        # Volume
        volume_score = 0
        if rvol_5d >= 1.3: volume_score += 15
        elif rvol_5d >= 1.0: volume_score += 12
        elif rvol_5d >= 0.85: volume_score += 8
        else: volume_score += 3
        if rvol_5d > rvol_10d > rvol_20d: volume_score += 3
        score_breakdown['timing_volume'] = min(15, volume_score)
        
        # Risk/Reward
        rr_score = 0
        sector_implied_alpha = 3.5 if sector_score >= 80 else (2.5 if sector_score >= 70 else 1.5)
        upside_gap = sector_implied_alpha - alpha_5d
        if sector_score >= 70: rr_score += 5
        if upside_gap >= 2.5 and sector_score >= 70: rr_score += 10
        elif upside_gap >= 0.5: rr_score += 7
        elif upside_gap >= -1.0: rr_score += 4
        if price > sma50 > sma200: rr_score += 5
        elif price > sma200: rr_score += 3
        score_breakdown['risk_reward'] = min(20, rr_score)
        
        # Sector Context
        context_score = 0
        if sector_stage == "Early" and sector_score >= 70: context_score += 10
        elif sector_stage == "Established" and sector_score >= 65: context_score += 7
        elif sector_stage == "Established" and sector_score >= 55: context_score += 4
        elif sector_stage == "Topping" or sector_score < 50: context_score -= 10
        score_breakdown['sector_context'] = context_score
        
        total_score = sum(score_breakdown.values())
        
        # Pattern Bonuses
        pattern_label = ""
        pattern_score = 0
        fb_matches = sum([1 for c in [-2.5<=alpha_20d<=0.5, 0<=alpha_5d<=3.0, rvol_5d>=1.2, days_positive<=8] if c])
        if fb_matches >= 3:
            pattern_score = 10 if fb_matches == 4 else 7
            pattern_label = "🚀 Fresh Breakout"
        elif alpha_20d >= 2.5 and -0.5 <= alpha_5d <= 2.0:
            db_matches = 1 + (1 if price>ema21 else 0) + (1 if abs(dist_from_ema21)<=0.05 else 0)
            if db_matches >= 2:
                pattern_score = 8 if db_matches == 3 else 5
                pattern_label = "💎 Dip Buy"
        elif sector_stage == "Early" and -1.5<=alpha_20d<=2.0 and 0.5<=alpha_5d<=4.0:
            pattern_score = 12
            pattern_label = "🎯 Sector Rotation"
        
        # Penalties
        extended_count = sum([1 for c in [alpha_5d>5.0, alpha_5d>3.5 and days_positive>=13, dist_from_ema21>0.12] if c])
        if extended_count >= 2:
            pattern_score = -12
            pattern_label = "⚡ Extended"
            
        divergence = detect_relative_strength_divergence(df, theme_suffix)
        if divergence == 'bearish_divergence':
            pattern_score -= 8
            pattern_label = pattern_label + " + 📉" if pattern_label else "📉 Bear Div"
            
        total_score = min(100, max(0, total_score + pattern_score))
        
        category = "⚠️ Caution"
        if total_score >= 75 and alpha_5d <= 3.5 and days_positive <= 12: category = "🎯 Optimal Entry"
        elif total_score >= 70 and alpha_20d >= 3.0 and alpha_5d <= 2.0: category = "💎 Pullback Buy"
        elif total_score >= 75 and alpha_5d > 4.0: category = "⚖️ Established Winner"
        elif total_score >= 60: category = "⚖️ Selective"
        
        return {
            'total_score': total_score, 'breakdown': score_breakdown, 'grade': _score_to_grade(total_score),
            'category': category, 'pattern_label': pattern_label, 'days_positive': days_positive
        }
    except Exception as e:
        logger.error(f"Error calculating stock score: {e}")
        return None

def calculate_theme_score(df: pd.DataFrame, view_key: str = 'Short') -> Optional[Dict]:
    if df is None or df.empty or len(df) < 4: return None
    try:
        col_ratio = f"RRG_Ratio_{view_key}"
        col_mom = f"RRG_Mom_{view_key}"
        if col_ratio not in df.columns or col_mom not in df.columns: return None
        
        recent = df.tail(4)
        last = recent.iloc[-1]
        ratio, mom = last[col_ratio], last[col_mom]
        score_breakdown = {}
        
        # Position
        quadrant_base = 30 if ratio>=100 and mom>=100 else 20 if ratio<100 and mom>=100 else 15 if ratio>=100 and mom<100 else 5
        distance = abs(ratio-100) + abs(mom-100)
        magnitude_bonus = 10 if distance>10 else 5 if distance>5 else 0
        score_breakdown['position'] = min(40, quadrant_base + magnitude_bonus)
        
        # Momentum Quality
        mom_short, mom_med, mom_long = last.get('RRG_Mom_Short',0), last.get('RRG_Mom_Med',0), last.get('RRG_Mom_Long',0)
        pos_count = sum([mom_short>100, mom_med>100, mom_long>100])
        consistency = 15 if pos_count==3 else 10 if pos_count==2 else 5 if pos_count==1 else 0
        acceleration = 15 if mom_short>mom_med>mom_long else 8 if mom_short>mom_med or mom_short>mom_long else 0
        score_breakdown['momentum_quality'] = consistency + acceleration
        
        # Trajectory
        traj_data = [{'r': recent.iloc[i][col_ratio], 'm': recent.iloc[i][col_mom]} for i in range(4)]
        quad_rank = lambda r,m: 3 if r>=100 and m>=100 else 2 if r<100 and m>=100 else 1 if r>=100 and m<100 else 0
        path = [quad_rank(t['r'], t['m']) for t in traj_data]
        
        if path[-1] > path[0]: imp = 10 if all(path[i]<=path[i+1] for i in range(3)) else 5
        elif path[-1] == path[0]: imp = 3
        else: imp = 0
        
        mom_change = traj_data[-1]['m'] - traj_data[0]['m']
        mom_acc = 10 if mom_change>5 else 5 if mom_change>2 else 3 if mom_change>0 else 0
        score_breakdown['trajectory'] = imp + mom_acc
        
        # Stability
        unique_quads = len(set([quad_rank(t['r'], t['m']) for t in traj_data]))
        stability = 10 if unique_quads==1 else 7 if (path==sorted(path) or path==sorted(path, reverse=True)) else 3 if unique_quads==2 else 0
        score_breakdown['stability'] = stability
        
        total_score = min(100, max(0, sum(score_breakdown.values())))
        
        # Freshness
        current_quad = quad_rank(ratio, mom)
        days = 1
        for i in range(len(traj_data)-2, -1, -1):
            if quad_rank(traj_data[i]['r'], traj_data[i]['m']) == current_quad: days+=1
            else: break
            
        freshness = "🆕 Fresh" if days<=2 else "⭐ Early" if days==3 else "🕐 Established"
        quad_name = "🟢 Leading" if ratio>=100 and mom>=100 else "🔵 Improving" if ratio<100 and mom>=100 else "🟡 Weakening" if ratio>=100 and mom<100 else "🔴 Lagging"
        traj_label = "🚀 Rising" if path[-1]>path[0] else "➡️ Stable" if path[-1]==path[0] and stability>=7 else "⚡ Choppy" if path[-1]==path[0] else "📉 Falling"
        
        return {
            'total_score': total_score, 'grade': _score_to_grade(total_score), 'quadrant': quad_name,
            'freshness': freshness, 'freshness_detail': f"Day {days}", 'trajectory_label': traj_label,
            'breakdown': score_breakdown, 'days_in_quadrant': days, 'stability': "Smooth" if stability>=7 else "Choppy"
        }
    except Exception as e:
        logger.error(f"Error calculating theme score: {e}")
        return None

def calculate_consensus_theme_score(df: pd.DataFrame) -> Optional[Dict]:
    if df is None or df.empty: return None
    try:
        s5 = calculate_theme_score(df, 'Short')
        s10 = calculate_theme_score(df, 'Med')
        s20 = calculate_theme_score(df, 'Long')
        if not all([s5, s10, s20]): return None
        
        consensus = (s5['total_score']*0.2 + s10['total_score']*0.3 + s20['total_score']*0.5)
        bullish_q = ['🟢 Leading', '🔵 Improving']
        bullish_count = sum([1 for s in [s5, s10, s20] if s['quadrant'] in bullish_q])
        
        align_bonus = 10 if bullish_count==3 else 0 if bullish_count==2 else -10 if bullish_count==1 else -15
        consensus = min(100, max(0, consensus + align_bonus))
        
        conviction = "High" if bullish_count==3 else "Medium" if bullish_count==2 else "Low" if bullish_count==1 else "None"
        emoji = "✅" if bullish_count==3 else "🎯" if bullish_count==2 else "⚠️" if bullish_count==1 else "🚫"
        
        return {
            'consensus_score': round(consensus, 1), 'grade': _score_to_grade(consensus),
            'conviction': conviction, 'conviction_emoji': emoji,
            'timeframes': {'5d': s5, '10d': s10, '20d': s20},
            'freshness': s20['freshness'], 'freshness_detail': s20['freshness_detail']
        }
    except Exception: return None

# ==========================================
# 6. ANALYSIS FUNCTIONS (NEWLY ADDED)
# ==========================================
def analyze_stocks_batch(
    etf_data_cache: Dict,
    stock_theme_pairs: List[Tuple[str, str]],
    show_divergences: bool,
    show_mkt_caps: bool,
    show_biotech: bool,
    theme_category_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Perform heavy lifting for stock analysis: MC fetching, Divergence scanning, and Data extraction.
    MOVED FROM MAIN to keep main clean.
    """
    if not stock_theme_pairs:
        return pd.DataFrame()
    
    unique_tickers = list(set([pair[0] for pair in stock_theme_pairs]))
    
    # 1. Fetch Market Caps
    mc_map = {}
    if show_mkt_caps:
        with st.spinner("Fetching Market Caps..."):
            mc_map = ud.fetch_market_caps_batch(unique_tickers)

    # 2. Scan Divergences
    div_map = {}
    if show_divergences:
        with st.spinner("Scanning Divergences..."):
            def process_div_single(stock):
                sdf = etf_data_cache.get(stock)
                if sdf is None or sdf.empty or len(sdf) < 20: return stock, "—"
                try:
                    d_d, _ = ud.prepare_data(sdf.copy())
                    if d_d is not None and not d_d.empty:
                        divs = ud.find_divergences(
                            d_d, stock, 'Daily', min_n=0,
                            periods_input=ud.DIV_CSV_PERIODS_DAYS,
                            optimize_for='PF',
                            lookback_period=ud.DIV_LOOKBACK_DEFAULT,
                            price_source=ud.DIV_SOURCE_DEFAULT,
                            strict_validation=(ud.DIV_STRICT_DEFAULT == "Yes"),
                            recent_days_filter=ud.DIV_DAYS_SINCE_DEFAULT,
                            rsi_diff_threshold=ud.DIV_RSI_DIFF_DEFAULT
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

    # 3. Process Single Stock Helper
    def process_single_stock(stock, stock_theme):
        if stock_theme == "Biotech" and not show_biotech: return None
        sdf = etf_data_cache.get(stock)
        if sdf is None or sdf.empty or len(sdf) < 20: return None

        # Volume Filter
        try:
            recent_vol = sdf['Volume'].values[-20:]
            recent_close = sdf['Close'].values[-20:]
            if (recent_vol.mean() * recent_close.mean()) < MIN_DOLLAR_VOLUME: return None
        except: return None

        try:
            last = sdf.iloc[-1]
            return {
                "Ticker": stock,
                "Theme": stock_theme,
                "Theme Category": theme_category_map.get(stock_theme, "Unknown"),
                "Price": last['Close'],
                "Market Cap (B)": mc_map.get(stock, 0) / 1e9,
                "Beta": last.get(f"Beta_{stock_theme}", 1.0),
                "Alpha 5d": last.get(f"Alpha_Short_{stock_theme}", 0),
                "Alpha 10d": last.get(f"Alpha_Med_{stock_theme}", 0),
                "Alpha 20d": last.get(f"Alpha_Long_{stock_theme}", 0),
                "RVOL 5d": last.get('RVOL_Short', 0),
                "RVOL 10d": last.get('RVOL_Med', 0),
                "RVOL 20d": last.get('RVOL_Long', 0),
                "Div": div_map.get(stock, "—"),
                "8 EMA": get_ma_signal(last['Close'], last.get('Ema8', 0)),
                "21 EMA": get_ma_signal(last['Close'], last.get('Ema21', 0)),
                "50 MA": get_ma_signal(last['Close'], last.get('Sma50', 0)),
                "200 MA": get_ma_signal(last['Close'], last.get('Sma200', 0)),
            }
        except Exception: return None

    # 4. Execute Extraction
    stock_data = []
    with st.spinner(f"Processing {len(stock_theme_pairs)} stocks..."):
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_stock = {executor.submit(process_single_stock, s, t): s for s, t in stock_theme_pairs}
            for future in as_completed(future_to_stock):
                result = future.result()
                if result: stock_data.append(result)
                
    return pd.DataFrame(stock_data)

def apply_stock_filters(df_stocks: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
    """
    Applies the list of dictionary filters (created by the UI) to the stocks dataframe.
    MOVED FROM MAIN to keep main clean.
    """
    if df_stocks.empty or not filters:
        return df_stocks

    df_filtered = df_stocks.copy()
    
    # Separate numeric and categorical
    numeric_filters = [f for f in filters if f['value_type'] in ['Number', 'Column']]
    categorical_filters = [f for f in filters if f['value_type'] == 'Categorical']
    
    # Build Numeric (AND logic)
    numeric_conditions = []
    for f in numeric_filters:
        col, op = f['column'], f['operator']
        if f['value_type'] == 'Number':
            val = f['value']
            cond = (df_filtered[col] >= val) if op == '>=' else (df_filtered[col] <= val)
        else:
            val_col = f['value_column']
            cond = (df_filtered[col] >= df_filtered[val_col]) if op == '>=' else (df_filtered[col] <= df_filtered[val_col])
        numeric_conditions.append(cond)
        
    # Build Categorical (Mixed Logic)
    categorical_conditions = []
    for i, f in enumerate(categorical_filters):
        col, val = f['column'], f['value_categorical']
        cond = (df_filtered[col] == val)
        logic = f.get('logic', 'AND') if i > 0 else None
        categorical_conditions.append((cond, logic))
        
    # Combine conditions
    final_condition = None
    
    # Merge numeric
    if numeric_conditions:
        final_condition = numeric_conditions[0]
        for c in numeric_conditions[1:]: final_condition = final_condition & c
        
    # Merge categorical
    if categorical_conditions:
        cat_combined = categorical_conditions[0][0]
        for i in range(1, len(categorical_conditions)):
            cond, logic = categorical_conditions[i]
            if logic == 'OR': cat_combined = cat_combined | cond
            else: cat_combined = cat_combined & cond
        
        final_condition = (final_condition & cat_combined) if final_condition is not None else cat_combined
        
    if final_condition is not None:
        df_filtered = df_filtered[final_condition]
        
    return df_filtered

# ==========================================
# 7. CORE LOGIC (CATEGORIES & DAYS)
# ==========================================
def calculate_days_in_category(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or df.empty or len(df) < 4: return {'days': 0, 'category': 'Unknown'}
    try:
        recent_days = df.tail(20)
        daily_categories = []
        for i in range(len(recent_days)):
            if i < 3: continue
            window = recent_days.iloc[max(0, i-3):i+1]
            if len(window) < 4: continue
            
            positions_10d = [{'ratio': window.iloc[j].get('RRG_Ratio_Med', 100), 'momentum': window.iloc[j].get('RRG_Mom_Med', 100)} for j in range(4)]
            avg_r = sum(p['ratio'] for p in positions_10d[:3]) / 3
            avg_m = sum(p['momentum'] for p in positions_10d[:3]) / 3
            
            perf = "outperforming" if positions_10d[3]['ratio'] > avg_r else "underperforming"
            mom = "gaining" if positions_10d[3]['momentum'] > avg_m else "losing"
            daily_categories.append(f"{mom}_{perf}")
            
        if not daily_categories: return {'days': 0, 'category': 'Unknown'}
        
        current_cat = daily_categories[-1]
        days = 1
        for i in range(len(daily_categories) - 2, -1, -1):
            if daily_categories[i] == current_cat: days += 1
            else: break
            
        return {'days': days, 'category': current_cat}
    except Exception as e:
        logger.error(f"Error calculating days: {e}")
        return {'days': 0, 'category': 'Unknown'}

def get_momentum_performance_categories(etf_data_cache: Dict, theme_map: Dict) -> Dict[str, List[Dict]]:
    """Categorize themes. Includes shortened display names."""
    categories = {
        'gaining_mom_outperforming': [],
        'gaining_mom_underperforming': [],
        'losing_mom_outperforming': [],
        'losing_mom_underperforming': []
    }
    
    for theme, ticker in theme_map.items():
        df = etf_data_cache.get(ticker)
        if df is None or df.empty or len(df) < 4: continue
        
        try:
            recent_4 = df.tail(4)
            pos_10d = [{'r': recent_4.iloc[i].get('RRG_Ratio_Med', 100), 'm': recent_4.iloc[i].get('RRG_Mom_Med', 100)} for i in range(4)]
            
            avg_r = sum(p['r'] for p in pos_10d[:3]) / 3
            avg_m = sum(p['m'] for p in pos_10d[:3]) / 3
            today_r, today_m = pos_10d[3]['r'], pos_10d[3]['m']
            
            perf_dir = "Outperforming" if today_r > avg_r else "Underperforming"
            mom_dir = "Gaining Momentum" if today_m > avg_m else "Losing Momentum"
            
            bucket_key = f"{'gaining_mom' if today_m > avg_m else 'losing_mom'}_{'outperforming' if today_r > avg_r else 'underperforming'}"
            arrow = "⬈" if bucket_key == 'gaining_mom_outperforming' else "⬉" if bucket_key == 'gaining_mom_underperforming' else "⬊" if bucket_key == 'losing_mom_outperforming' else "⬋"
            
            full_cat_name = f"{mom_dir} & {perf_dir}"
            short_cat_name = _shorten_category_name(full_cat_name)
            
            r_5d, m_5d = recent_4.iloc[-1].get('RRG_Ratio_Short', 100), recent_4.iloc[-1].get('RRG_Mom_Short', 100)
            if r_5d > today_r and m_5d > today_m: confirmation = "5d accelerating ahead"
            elif r_5d > today_r or m_5d > today_m: confirmation = "5d confirming trend"
            else: confirmation = "5d lagging behind"
            
            theme_info = {
                'theme': theme,
                'category': full_cat_name,
                'display_category': arrow + " " + short_cat_name,
                'arrow': arrow,
                'quadrant_5d': get_quadrant_name(r_5d, m_5d),
                'quadrant_10d': get_quadrant_name(today_r, today_m),
                'quadrant_20d': get_quadrant_name(recent_4.iloc[-1].get('RRG_Ratio_Long', 100), recent_4.iloc[-1].get('RRG_Mom_Long', 100)),
                'reason': f"10d: {mom_dir.lower()} & {perf_dir.lower()}, {confirmation}",
                'days_in_category': calculate_days_in_category(df)['days']
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
# 8. VISUALIZATION & CLASSIFICATION
# ==========================================
def classify_setup(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty: return None
    try:
        last = df.iloc[-1]
        m5, m10, m20 = last.get("RRG_Mom_Short",0), last.get("RRG_Mom_Med",0), last.get("RRG_Mom_Long",0)
        ratio_20 = last.get("RRG_Ratio_Long", 100)
        
        if m20 < 100 and m5 > 100 and m5 > (m20 + JHOOK_MIN_SHIFT): return "🪝 J-Hook"
        if ratio_20 > 100 and m5 > 100 and m5 > m10: return "🚩 Bull Flag"
        if m5 > m10 > m20 and m20 > 100: return "🚀 Rocket"
        if m5 < m10 < m20 and m20 < 100: return "💥 Breakdown"
        if m20 < 100 and m5 > 100 and m5 < (m20 + JHOOK_MIN_SHIFT): return "🐱 Dead Cat"
        if ratio_20 > 100 and m20 > 100 and m10 > 100 and m5 > 100: return "⚡ Power"
        if ratio_20 > 100 and m5 < 100 and m5 < m10: return "📉 Fading"
    except Exception: pass
    return None

def get_quadrant_status(df: pd.DataFrame, timeframe_key: str) -> str:
    if df is None or df.empty: return "N/A"
    try:
        r = df[f"RRG_Ratio_{timeframe_key}"].iloc[-1]
        m = df[f"RRG_Mom_{timeframe_key}"].iloc[-1]
        return get_quadrant_name(r, m)
    except: return "N/A"

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
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers+text', name=theme, text=[""]*(n-1)+[theme], customdata=[theme]*n,
            textposition="top center", marker=dict(size=[MARKER_SIZE_TRAIL]*(n-1)+[MARKER_SIZE_CURRENT], color=color, opacity=[TRAIL_OPACITY]*(n-1)+[CURRENT_OPACITY], line=dict(width=1, color='white')),
            line=dict(color=color, width=1 if show_trails else 0, shape='spline', smoothing=1.3),
            hoverinfo='text+name', hovertext=[f"{theme}<br>Trend: {x:.1f}<br>Mom: {y:.1f}" for x, y in zip(x_vals, y_vals)]
        ))

    limit_x = max(max([abs(x - 100) for x in all_x]) * 1.1, 2.0) if all_x else 2.0
    limit_y = max(max([abs(y - 100) for y in all_y]) * 1.1, 2.0) if all_y else 2.0
    
    fig.add_hline(y=100, line_width=1, line_color="gray", line_dash="dash")
    fig.add_vline(x=100, line_width=1, line_color="gray", line_dash="dash")
    
    labels = [("LEADING", 100+limit_x*0.5, 100+limit_y*0.5, "rgba(0, 255, 0, 0.7)"),
              ("IMPROVING", 100-limit_x*0.5, 100+limit_y*0.5, "rgba(0, 100, 255, 0.7)"),
              ("WEAKENING", 100+limit_x*0.5, 100-limit_y*0.5, "rgba(255, 165, 0, 0.7)"),
              ("LAGGING", 100-limit_x*0.5, 100-limit_y*0.5, "rgba(255, 0, 0, 0.7)")]
    
    for txt, x, y, col in labels: fig.add_annotation(x=x, y=y, text=f"<b>{txt}</b>", showarrow=False, font=dict(color=col, size=20))
    
    fig.update_layout(xaxis=dict(title="Relative Trend", showgrid=False, range=[100-limit_x, 100+limit_x]), 
                      yaxis=dict(title="Relative Momentum", showgrid=False, range=[100-limit_y, 100+limit_y]),
                      height=750, showlegend=False, template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    return fig

# ==========================================
# 9. ORCHESTRATOR
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_and_process_universe(benchmark_ticker: str = "SPY"):
    """HEAVILY OPTIMIZED data pipeline."""
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
    if 'Date' in master_df.columns: master_df = master_df.set_index(pd.to_datetime(master_df['Date'])).sort_index()

    needed = set(tickers) | set(theme_map.values()) | {benchmark_ticker}
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