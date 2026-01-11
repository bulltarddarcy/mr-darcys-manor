# --- IMPORTS ---
import streamlit as st
import pandas as pd
from datetime import date

# --- MODULE IMPORTS ---
import main_darcy
import main_sector
import main_prices  # <--- NEW IMPORT
import utils_darcy as ud

# --- 0. PAGE CONFIGURATION ---
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

# --- 1. CSS STYLING ---
st.markdown("""<style>
.block-container{padding-top:3.5rem;padding-bottom:1rem;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}
.zone-wrapper{
    flex-grow: 1; 
    position: relative; 
    height: 24px; 
    background-color: rgba(0,0,0,0.03);
    border-radius: 4px;
    overflow: hidden;
}
.zone-bar{
    position: absolute;
    left: 0; 
    top: 0; 
    bottom: 0; 
    z-index: 1;
    border-radius: 3px;
    opacity: 0.65;
}
.zone-bull{background-color: #71d28a;}
.zone-bear{background-color: #f29ca0;}
.zone-value{
    position: absolute; 
    right: 8px; 
    top: 0; 
    bottom: 0; 
    line-height: 24px; 
    font-size: 11px; 
    font-weight: 600; 
    color: #444; 
    z-index: 2;
}
.price-divider{
    display: flex; 
    align-items: center; 
    margin: 12px 0 12px 100px; 
    border-top: 2px dashed #ccc; 
    position: relative;
    height: 1px;
}
.price-badge{
    position: absolute; 
    right: 0; 
    top: -10px; 
    background: #fff; 
    padding: 0 8px; 
    font-size: 12px; 
    font-weight: 700; 
    color: #333; 
    border: 1px solid #ccc; 
    border-radius: 10px;
}
.metric-row{display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;}
.price-badge-header{
    font-size: 18px; 
    font-weight: 700; 
    background: #e3f2fd; 
    color: #1565c0; 
    padding: 4px 12px; 
    border-radius: 6px;
}
.badge{
    font-size: 14px; 
    background: #f1f3f4; 
    color: #333; 
    padding: 6px 10px; 
    border-radius: 6px; 
    border: 1px solid #ddd;
}
.light-note{font-size: 0.85rem; color: #888; font-style: italic;}
</style>""", unsafe_allow_html=True)

# --- 2. SESSION STATE INIT ---
if "data_global" not in st.session_state:
    st.session_state.data_global = pd.DataFrame()

# --- 3. DATA LOADING & CACHE ---
# We load the main dataset once here and pass it down.
@st.cache_data(ttl=600)
def load_data():
    return ud.load_and_clean_data()

# --- 4. NAVIGATION SETUP ---
# Define the pages for the app
pages = [
    st.Page(main_sector.run_sector_rotation_app, title="Sector Rotation", icon="🔄", default=True),
    st.Page(main_darcy.run_database_app, title="Database", icon="📂"),
    st.Page(main_darcy.run_rankings_app, title="Rankings", icon="🏆"),
    st.Page(main_darcy.run_pivot_tables_app, title="Flow Pivot", icon="🎯"),
    st.Page(main_darcy.run_strike_zones_app, title="Strike Zones", icon="📊"),
    
    # --- UPDATED REFERENCES TO MAIN_PRICES ---
    st.Page(main_prices.run_price_divergences_app, title="Price Divergences", icon="📉"),
    st.Page(main_prices.run_rsi_scanner_app, title="RSI Scanner", icon="🤖"),
    st.Page(main_prices.run_seasonality_app, title="Seasonality", icon="📅"),
    st.Page(main_prices.run_ema_distance_app, title="EMA Distance", icon="📏"),
]

pg = st.navigation(pages)

# --- 5. GLOBAL DATA SYNC ---
# We fetch data only if the selected page needs the global CSV (Database/Rankings/Pivot/Zones)
# The "Price" apps use their own Parquet loaders, but passing the df doesn't hurt.

df_global = pd.DataFrame()
if pg.title in ["Database", "Rankings", "Flow Pivot", "Strike Zones"]:
    df_global = load_data()
    
    if df_global.empty:
        st.error("Global Data Load Failed. Check Google Drive connection.")
    else:
        # Side Bar Info
        with st.sidebar:
            st.success("Data Connected")
            st.caption(f"Rows: {len(df_global):,}")
            last_date = df_global["Trade Date"].max().strftime('%Y-%m-%d')
            st.caption(f"Last Trade: {last_date}")
            
            # Health Checks (Only show if specifically requested or debug mode)
            with st.expander("System Health"):
                tm_key = "URL_TICKER_MAP"
                tm_url = st.secrets.get(tm_key, "")
                if not tm_url:
                    st.markdown(f"❌ **Ticker Map**: Secret Missing")
                elif "drive.google.com" in tm_url:
                     st.markdown(f"✅ **Ticker Map**: Connected")

                health_config = ud.get_parquet_config()
                for name, key in health_config.items():
                    url = st.secrets.get(key, "")
                    if url and "drive.google.com" in url:
                        st.markdown(f"✅ **{name}**: Linked")
                    else:
                        st.markdown(f"❌ **{name}**: Missing/Invalid")

# --- 6. RUN ---
# We pass the global dataframe to the page function
try:
    pg.run(df_global)
except TypeError:
    # Handle apps that don't accept args (like Sector Rotation might not yet)
    pg.run()

# Global padding
st.markdown("<br><br>", unsafe_allow_html=True)