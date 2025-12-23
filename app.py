import streamlit as st
from streamlit_gsheets import GSheetsConnection

# Page Config for a professional look (Image 2 style)
st.set_page_config(page_title="Bulltard Option Flow", layout="wide")

st.title("Option Flow")

# Establish Connection
conn = st.connection("gsheets", type=GSheetsConnection)

# --- TAB 1: SEARCHABLE DATABASE (Your 2nd Image) ---
url = "https://docs.google.com/spreadsheets/d/1wejcjgk6_lq86gQKl4iqT_cZVLy-RN-oNyUSR6HbupQ/edit?usp=sharing"
df_raw = conn.read(spreadsheet=url, worksheet="Database_Tab_Name")

# Add Search Filters in a row
col1, col2, col3 = st.columns(3)
with col1:
    symbol_search = st.text_input("Symbol", value="AMZN")
with col2:
    order_type = st.multiselect("Order Type", options=df_raw['Order Type'].unique(), default=["Calls Bought", "Puts Bought"])

# Filter the data
filtered_df = df_raw[(df_raw['Symbol'] == symbol_search) & (df_raw['Order Type'].isin(order_type))]
st.dataframe(filtered_df, use_container_width=True)

---

# --- TAB 2: PIVOT TABLES (Your 3rd Image) ---
st.divider()
st.header("Market Summary")

# Pull data directly from your Pivot Table tab in Google Sheets
df_pivot = conn.read(spreadsheet=url, worksheet="trades")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Calls Bought")
    st.table(df_pivot) # Use .table for a clean, non-interactive look