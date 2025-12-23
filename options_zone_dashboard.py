import warnings
warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import yfinance as yf
import math
import streamlit_authenticator as stauth

# --- 1. AUTHENTICATION SETUP ---
credentials = {
    "usernames": {
        "admin": {"name": "admin", "password": "admindarcy"},
        "friend1": {"name": "mister", "password": "darcy"}
    }
}

authenticator = stauth.Authenticate(credentials, "cookie_name", "signature_key", 30)
name, authentication_status, username = authenticator.login("main")

# --- 2. THE DASHBOARD (Only runs if logged in) ---
if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    
    # This pulls the URL from the secure cloud settings
    # Ensure you set this up in the Streamlit Cloud Settings!
    DEFAULT_SHEET_CSV_URL = st.secrets["GSHEET_URL"]

    st.set_page_config(page_title="Options Strike Zones Dashboard", layout="wide")

    # ---------- Styles ----------
    st.markdown("""
    <style>
    :root{
      --bg:#1f1f22; --panel:#2a2d31; --panel2:#24272b; --text:#e7e7ea;
      --green:#71d28a; --red:#f29ca0; --line:#66b7ff; --ema8:#b689ff; --ema21:#ffb86b; --sma200:#ffffff; --price:#bfe7ff;
    }
    html,body,[class*="css"]{color:var(--text)!important;background-color:var(--bg)!important;}
    .block-container{padding-top:1.2rem;padding-bottom:1rem;}
    .control-box{padding:14px 0; border-radius:10px;}
    .zones-panel{padding:14px 0; border-radius:10px;}
    .zone-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
    .zone-label{width:220px;font-weight:700;color:#fff}
    .zone-bar{height:22px;border-radius:6px;min-width:6px}
    .zone-bull{background:linear-gradient(90deg,var(--green),#60c57b)}
    .zone-bear{background:linear-gradient(90deg,var(--red),#e4878d)}
    .zone-value{min-width:220px;font-variant-numeric:tabular-nums}
    .price-divider{position:relative;margin:16px 0 12px 0;text-align:center}
    .price-divider .line{height:2px;background:var(--line);opacity:.9}
    .price-badge{position:absolute;left:50%;transform:translate(-50%,-50%);top:0;background:#2b3a45;color:#bfe7ff;
      border:1px solid #56b6ff;border-radius:16px;padding:6px 12px;font-weight:800;font-size:12px;letter-spacing:.3px;
      box-shadow:0 2px 8px rgba(0,0,0,.35)}
    .metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:.35rem 0 .75rem 0}
    .badge{background:#2b3a45;border:1px solid #3b5566;color:#cde8ff;border-radius:18px;padding:6px 10px;font-weight:700}
    .price-badge-header{background:#2b3a45;border:1px solid #56b6ff;color:#bfe7ff;border-radius:18px;padding:6px 10px;font-weight:800}
    th,td{border:1px solid #3a3f45;padding:8px} th{background:#343a40;text-align:left}
    [data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0px; }
    [data-testid="stSidebar"] .stCheckbox { margin-bottom: -10px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Options Strike Zones Dashboard")

    # ---------- ALL REMAINING DASHBOARD CODE ----------
    # (The rest of your script logic goes here, exactly as you wrote it)
    
    # ... [Rest of your code from 'c1, c2, c3...' down to the end] ...
    # Make sure all the code below this is INDENTED to stay inside the 'if' block!

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')