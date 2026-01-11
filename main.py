{
type: uploaded file
fileName: mr-darcys-manor-main (15).zip/mr-darcys-manor-main/main.py
fullContent:
import streamlit as st
import pandas as pd
from functools import partial
from datetime import date

# --- MODULE IMPORTS ---
import main_darcy
import main_sector
import utils_darcy as ud  # For global data loading & health checks

# --- 0. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Trading Toolbox",
    layout="wide",
    page_icon="💎"
)

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
.zone-marker{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background-color: rgba(0,0,0,0.5);
    z-index: 2;
}
.stRadio [data-testid="stMarkdownContainer"] > p {font-size: 14px;}
div[data-testid="stMetricValue"] {font-size: 24px;}
</style>
""", unsafe_allow_html=True)

# --- 2. GLOBAL DATA LOADING ---
@st.cache_data(ttl=600)
def load_global_data():
    """
    Loads the main trade log (GSHEET_URL) once at the app level.
    This allows passing the data down to sub-apps like Sector Rotation.
    """
    try:
        url = st.secrets.get("GSHEET_URL")
        if url:
            # utilizing the robust loader from utils_darcy
            return ud.load_and_clean_data(url)
    except Exception as e:
        print(f"Data Load Error: {e}")
    return pd.DataFrame()

# Load data once
df_global = load_global_data()

# --- 3. HEALTH CHECKS ---
with st.sidebar:
    st.header("System Status")
    
    with st.expander("🔌 Data Connections", expanded=False):
        # A. Check Ticker Map
        tm_key = "URL_TICKER_MAP"
        tm_url = st.secrets.get(tm_key, "")
        if not tm_url:
            st.markdown(f"❌ **Ticker Map**: Secret Missing")
        elif "drive.google.com" not in tm_url:
            st.markdown(f"⚠️ **Ticker Map**: Invalid URL")
        else:
             st.markdown(f"✅ **Ticker Map**: Connected")

        # B. Check Parquet Files (using Darcy Utils config)
        health_config = ud.get_parquet_config()
        all_good = True
        
        for name, key in health_config.items():
            url = st.secrets.get(key, "")
            
            if not url:
                st.markdown(f"❌ **{name}**: Secret Missing")
                all_good = False
            elif "drive.google.com" not in url:
                 st.markdown(f"⚠️ **{name}**: Invalid URL Format")
                 all_good = False
            else:
                status_icon = "✅"
                note = ""
                if "usp=drive_link" in url:
                    status_icon = "⚠️" 
                    note = "(drive_link)"
                st.markdown(f"{status_icon} **{name}**: Linked {note}")
        
        if all_good and tm_url:
            st.caption("All configurations look valid.")
        else:
            st.error("Configuration errors detected.")
            
    st.markdown("---")

# --- 4. NAVIGATION SETUP ---
# define pages
# Note: We use partial to pass the df_global to main_sector, 
# while main_darcy runs with its standard signature.
pages = {
    "Apps": [
        st.Page(main_darcy.run_darcy_app, title="Price Divergences", icon="📉"),
        st.Page(partial(main_sector.run_sector_rotation_app, df_global=df_global), title="Sector Rotation", icon="🔄"),
    ]
}

pg = st.navigation(pages)

# --- 5. RUN ---
pg.run()

# Global padding
st.markdown("<br><br>", unsafe_allow_html=True)
}
