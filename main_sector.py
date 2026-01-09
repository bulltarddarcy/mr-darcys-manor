import streamlit as st
import pandas as pd
from utils_sector import (
    load_sector_universe, fetch_sector_data, process_market_data, 
    plot_rrg_chart, highlight_alpha, TIMEFRAMES
)

def run_sector_page():
    st.title("🔄 Sector Rotation & Themes")
    
    # --- 1. CONFIGURATION & LOADING ---
    with st.expander("⚙️ Settings & Data", expanded=False):
        col1, col2 = st.columns([1,3])
        with col1:
            if st.button("🔄 Refresh Data", type="primary"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            st.info("Data is cached for 1 hour. Click Refresh to force an update.")

    # Load Universe
    uni_df, tickers, theme_map = load_sector_universe()
    
    if uni_df.empty:
        st.error("Universe not loaded. Check 'SECTOR_UNIVERSE' secret.")
        return

    # Load & Process Data
    with st.spinner(f"Analyzing {len(tickers)} tickers..."):
        raw_data = fetch_sector_data(tickers)
        summary_df = process_market_data(raw_data, theme_map, uni_df)
    
    if summary_df.empty:
        st.warning("No data processed.")
        return

    # --- 2. MAIN TABS ---
    tab_rrg, tab_explore = st.tabs(["📈 Rotation Graph", "🔎 Theme Explorer"])

    # --- TAB 1: RRG CHART ---
    with tab_rrg:
        c1, c2 = st.columns([1, 4])
        with c1:
            st.markdown("### View")
            view_mode = st.radio("Timeframe", options=["Short", "Med", "Long"], index=1, 
                                 help="Short=5d, Med=10d, Long=20d")
            
            st.markdown("### Filter")
            show_etfs = st.checkbox("Show Theme ETFs", value=True)
            show_stocks = st.checkbox("Show Stocks", value=False)
            
            sel_themes = st.multiselect("Filter Themes", options=sorted(uni_df['Theme'].unique()), default=[])
        
        with c2:
            # Filter Data for Chart
            mask = pd.Series([False] * len(summary_df), index=summary_df.index)
            
            if show_etfs:
                mask = mask | (summary_df['Role'] == 'ETF')
            if show_stocks:
                mask = mask | (summary_df['Role'] == 'Stock')
                
            chart_df = summary_df[mask].copy()
            
            if sel_themes:
                chart_df = chart_df[chart_df['Theme'].isin(sel_themes)]
            
            if chart_df.empty:
                st.info("No data matches filters.")
            else:
                fig = plot_rrg_chart(chart_df, view=view_mode)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top Movers below chart
                st.markdown(f"**Top Movers ({view_mode})**")
                cols = st.columns(4)
                
                # Sort by Alpha of current view
                alpha_col = f"Alpha_{view_mode}"
                sorted_df = chart_df.sort_values(by=alpha_col, ascending=False)
                
                for i in range(min(4, len(sorted_df))):
                    row = sorted_df.iloc[i]
                    with cols[i]:
                        st.metric(
                            label=f"{row['Ticker']} ({row['Theme']})", 
                            value=f"${row['Price']:.2f}",
                            delta=f"{row[alpha_col]:.2f}% vs SPY"
                        )

    # --- TAB 2: EXPLORER ---
    with tab_explore:
        st.markdown("### Deep Dive")
        
        # Search / Filter
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            txt_search = st.text_input("Search Ticker", placeholder="e.g. NVDA").upper()
        with f_col2:
            filter_theme = st.selectbox("Filter by Theme", ["All"] + sorted(uni_df['Theme'].unique().tolist()))

        # Apply Filters
        disp_df = summary_df.copy()
        if txt_search:
            disp_df = disp_df[disp_df['Ticker'].str.contains(txt_search)]
        if filter_theme != "All":
            disp_df = disp_df[disp_df['Theme'] == filter_theme]
            
        # Display Configuration
        # We show columns for all timeframes or just the selected one? 
        # Let's show a comprehensive view.
        
        cols_to_show = ['Ticker', 'Theme', 'Price', 'RVOL', 'Alpha_Short', 'Alpha_Med', 'Alpha_Long']
        
        # Apply Pandas Styler
        styled_df = disp_df[cols_to_show].style.format({
            'Price': "${:.2f}",
            'RVOL': "{:.1f}x",
            'Alpha_Short': "{:+.2f}%",
            'Alpha_Med': "{:+.2f}%",
            'Alpha_Long': "{:+.2f}%"
        }).applymap(highlight_alpha, subset=['Alpha_Short', 'Alpha_Med', 'Alpha_Long'])
        
        st.dataframe(
            styled_df, 
            use_container_width=True, 
            height=600,
            column_config={
                "Ticker": st.column_config.TextColumn("Symbol", help="Stock Ticker"),
                "Alpha_Short": st.column_config.NumberColumn("Alpha (5d)", help="Performance vs SPY over 5 days"),
                "Alpha_Med": st.column_config.NumberColumn("Alpha (10d)", help="Performance vs SPY over 10 days"),
                "Alpha_Long": st.column_config.NumberColumn("Alpha (20d)", help="Performance vs SPY over 20 days"),
            }
        )

# This block allows running this file directly for testing, 
# but main.py will import the run_sector_page function.
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Sector Rotation")
    run_sector_page()