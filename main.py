# main.py
import streamlit as st
import pandas as pd
from datetime import date

# --- MODULE IMPORTS ---
import main_options  # Database, Rankings, Pivots, Strike Zones
import main_prices   # Divergences, RSI, Seasonality, EMA
import main_sector   # Sector Rotation
import utils_options as uo # For loading global data
import utils_prices as up  # For config checks

# --- 0. PAGE CONFIG ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

# --- 1. CSS ---
st.markdown("""<style>
.block-container{padding-top:3.5rem;padding-bottom:1rem;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}
.zone-wrapper{flex-grow: 1; position: relative; height: 24px; background-color: rgba(0,0,0,0.03); border-radius: 4px; overflow: hidden;}
.zone-bar{position: absolute; left: 0; top: 0; bottom: 0; z-index: 1; border-radius: 3px; opacity: 0.65;}
.zone-bull{background-color: #71d28a;} .zone-bear{background-color: #f29ca0;}
.zone-value{position: absolute; right: 8px; top: 0; bottom: 0; display: flex; align-items: center; z-index: 2; font-size: 12px; font-weight: 700; color: #1f1f1f; text-shadow: 0 0 4px rgba(255,255,255,0.8);}
.price-divider { display: flex; align-items: center; justify-content: center; position: relative; margin: 24px 0; width: 100%; }
.price-divider::before, .price-divider::after { content: ""; flex-grow: 1; height: 2px; background: #66b7ff; opacity: 0.4; }
.price-badge { background: rgba(102, 183, 255, 0.1); color: #66b7ff; border: 1px solid rgba(102, 183, 255, 0.5); border-radius: 16px; padding: 6px 14px; font-weight: 800; font-size: 12px; margin: 0 12px; z-index: 1; }
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
.badge{background: rgba(128, 128, 128, 0.08); border: 1px solid rgba(128, 128, 128, 0.2); border-radius:18px; padding:6px 10px; font-weight:700}
.price-badge-header{background: rgba(102, 183, 255, 0.1); border: 1px solid #66b7ff; border-radius:18px; padding:6px 10px; font-weight:800}
.light-note { opacity: 0.7; font-size: 14px; margin-bottom: 10px; }
</style>""", unsafe_allow_html=True)

# --- 2. GLOBAL DATA LOADING ---
try:
    # Load Options DB (Using Options Utils)
    sheet_url = st.secrets["GSHEET_URL"]
    df_global = uo.load_and_clean_data(sheet_url)
    
    db_date = "No Data"
    if not df_global.empty and "Trade Date" in df_global.columns:
        db_date = df_global["Trade Date"].max().strftime("%d %b %y")
        
    # Price History Status (Using Prices Utils)
    price_status = "Offline"
    try:
        cfg = up.get_parquet_config()
        if cfg: price_status = "Linked"
    except: pass

    # --- 3. NAVIGATION ---
    pg = st.navigation([
        # OPTIONS APPS (Main Options)
        st.Page(lambda: main_options.run_database_app(df_global), title="Database", icon="📂", url_path="options_db", default=True),
        st.Page(lambda: main_options.run_rankings_app(df_global), title="Rankings", icon="🏆", url_path="rankings"),
        st.Page(lambda: main_options.run_pivot_tables_app(df_global), title="Pivot Tables", icon="🎯", url_path="pivot_tables"),
        st.Page(lambda: main_options.run_strike_zones_app(df_global), title="Strike Zones", icon="📊", url_path="strike_zones"),
        
        # PRICE APPS (Main Prices)
        st.Page(lambda: main_prices.run_price_divergences_app(df_global), title="Price Divergences", icon="📉", url_path="price_divergences"),
        st.Page(lambda: main_prices.run_rsi_scanner_app(df_global), title="RSI Scanner", icon="🤖", url_path="rsi_scanner"),
        st.Page(lambda: main_prices.run_seasonality_app(df_global), title="Seasonality", icon="📅", url_path="seasonality"),
        st.Page(lambda: main_prices.run_ema_distance_app(df_global), title="EMA Distance", icon="📏", url_path="ema_distance"),
        
        # SECTOR APP
        st.Page(lambda: main_sector.run_sector_rotation_app(df_global), title="Sector Rotation", icon="🔄", url_path="sector_rotation"),
    ])

    # --- 4. SIDEBAR ---
    st.sidebar.caption("🖥️ Wide monitor & light mode.")
    st.sidebar.caption(f"💾 **JB Database:** {db_date}")
    
    with st.sidebar.expander("🏥 Data Health Check", expanded=False):
        tm_url = st.secrets.get("URL_TICKER_MAP", "")
        if not tm_url: st.markdown("❌ **Ticker Map**: Missing")
        else: st.markdown("✅ **Ticker Map**: Connected")

        health_config = up.get_parquet_config()
        for name, key in health_config.items():
            if key in st.secrets: st.markdown(f"✅ **{name}**: Linked")
            else: st.markdown(f"❌ **{name}**: Missing Secret")
    
    pg.run()
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
except Exception as e: 
    st.error(f"Error initializing dashboard: {e}")
