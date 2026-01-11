"""
Sector Rotation App - REFACTORED VERSION
With multi-theme support, smart filters, and comprehensive scoring.
"""

import streamlit as st
import pandas as pd
import utils_sector as us

# ==========================================
# UI HELPERS
# ==========================================
def get_ma_signal(price: float, ma_val: float) -> str:
    """
    Return emoji based on price vs moving average.
    
    Args:
        price: Current price
        ma_val: Moving average value
        
    Returns:
        Emoji indicator
    """
    if pd.isna(ma_val) or ma_val == 0:
        return "⚠️"
    return "✅" if price > ma_val else "❌"

# ==========================================
# MAIN PAGE FUNCTION
# ==========================================
def run_sector_rotation_app(df_global=None):
    """
    Main entry point for Sector Rotation application.
    
    Features:
    - RRG quadrant analysis
    - Multi-timeframe views
    - Stock-level alpha analysis
    - Smart pattern filters
    - Comprehensive scoring
    """
    st.title("🔄 Sector Rotation")
    
    # --- 0. BENCHMARK CONTROL ---
    if "sector_benchmark" not in st.session_state:
        st.session_state.sector_benchmark = "SPY"

    # --- 1. DATA FETCH (CACHED) ---
    with st.spinner(f"Syncing Sector Data ({st.session_state.sector_benchmark})..."):
        etf_data_cache, missing_tickers, theme_map, uni_df, stock_themes = \
            us.fetch_and_process_universe(st.session_state.sector_benchmark)

    if uni_df.empty:
        st.warning("⚠️ SECTOR_UNIVERSE secret is missing or empty.")
        return

    # --- 2. MISSING DATA CHECK ---
    if missing_tickers:
        with st.expander(f"⚠️ Missing Data for {len(missing_tickers)} Tickers", expanded=False):
            st.caption("These tickers were in your Universe but not found in the parquet file.")
            st.write(", ".join(missing_tickers))

    # --- 3. SESSION STATE INITIALIZATION ---
    if "sector_view" not in st.session_state:
        st.session_state.sector_view = "5 Days"
    if "sector_trails" not in st.session_state:
        st.session_state.sector_trails = False
    
    all_themes = sorted(list(theme_map.keys()))
    if not all_themes:
        st.error("No valid themes found. Check data sources.")
        return

    if "sector_target" not in st.session_state or st.session_state.sector_target not in all_themes:
        st.session_state.sector_target = "All"  # Default to All instead of first theme
    
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # --- 4. RRG QUADRANT GRAPHIC ---
    st.subheader("Rotation Quadrant Graphic")

    # User Guide
    with st.expander("🗺️ Graphic User Guide", expanded=False):
        st.markdown(f"""
        **🧮 How It Works (The Math)**
        This chart shows **Relative Performance** against **{st.session_state.sector_benchmark}** (not absolute price).
        
        * **X-Axis (Trend):** Are we beating the benchmark?
            * `> 100`: Outperforming {st.session_state.sector_benchmark}
            * `< 100`: Underperforming {st.session_state.sector_benchmark}
        * **Y-Axis (Momentum):** How fast is the trend changing?
            * `> 100`: Gaining speed (Acceleration)
            * `< 100`: Losing speed (Deceleration)
        
        *Calculations use Weighted Regression (recent days weighted 3x more)*
        
        **📊 Quadrant Guide**
        * 🟢 **LEADING (Top Right):** Strong trend + accelerating. The winners.
        * 🟡 **WEAKENING (Bottom Right):** Strong trend but losing steam. Take profits.
        * 🔴 **LAGGING (Bottom Left):** Weak trend + decelerating. The losers.
        * 🔵 **IMPROVING (Top Left):** Weak trend but momentum building. Turnarounds.
        """)

    # Controls
    with st.expander("⚙️ Chart Inputs & Filters", expanded=False):
        col_inputs, col_filters = st.columns([1, 1])
        
        # --- LEFT: TIMEFRAME & BENCHMARK ---
        with col_inputs:
            st.markdown("**Benchmark Ticker**")
            new_benchmark = st.radio(
                "Benchmark",
                ["SPY", "QQQ"],
                horizontal=True,
                index=["SPY", "QQQ"].index(st.session_state.sector_benchmark) 
                    if st.session_state.sector_benchmark in ["SPY", "QQQ"] else 0,
                key="sector_benchmark_radio",
                label_visibility="collapsed"
            )
            
            if new_benchmark != st.session_state.sector_benchmark:
                st.session_state.sector_benchmark = new_benchmark
                st.cache_data.clear()
                st.rerun()

            st.markdown("---")
            st.markdown("**Timeframe Window**")
            st.session_state.sector_view = st.radio(
                "Timeframe Window",
                ["5 Days", "10 Days", "20 Days"],
                horizontal=True,
                key="timeframe_radio",
                label_visibility="collapsed"
            )
            
            st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)
            st.session_state.sector_trails = st.checkbox(
                "Show 3-Day Trails",
                value=st.session_state.sector_trails
            )
            
            # Display last data date
            if st.session_state.sector_benchmark in etf_data_cache:
                bench_df = etf_data_cache[st.session_state.sector_benchmark]
                if not bench_df.empty:
                    last_dt = bench_df.index[-1].strftime("%Y-%m-%d")
                    st.caption(f"📅 Data Date: {last_dt}")

        # --- RIGHT: SECTOR FILTERS ---
        with col_filters:
            st.markdown("**Sectors Shown**")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("➕ Everything", use_container_width=True):
                    st.session_state.sector_theme_filter_widget = all_themes
                    st.rerun()

            with btn_col2:
                if st.button("⭐ Big 11", use_container_width=True):
                    big_11 = [
                        "Communications", "Consumer Discretionary", "Consumer Staples",
                        "Energy", "Financials", "Healthcare", "Industrials",
                        "Materials", "Real Estate", "Technology", "Utilities"
                    ]
                    valid = [t for t in big_11 if t in all_themes]
                    st.session_state.sector_theme_filter_widget = valid
                    st.rerun()

            with btn_col3:
                if st.button("➖ Clear", use_container_width=True):
                    st.session_state.sector_theme_filter_widget = []
                    st.rerun()
            
            sel_themes = st.multiselect(
                "Select Themes",
                all_themes,
                key="sector_theme_filter_widget",
                label_visibility="collapsed"
            )
    
    filtered_map = {k: v for k, v in theme_map.items() if k in sel_themes}
    timeframe_map = {"5 Days": "Short", "10 Days": "Med", "20 Days": "Long"}
    view_key = timeframe_map[st.session_state.sector_view]

    # --- 6. RRG CHART ---
    
    # Get categories for filtering
    categories = us.get_momentum_performance_categories(etf_data_cache, theme_map)
    
    # Category filter buttons
    st.markdown("**Filter Chart by Category:**")
    
    # Row 1: All Themes button
    if st.button("🎯 All Themes", use_container_width=False, key="filter_all"):
        st.session_state.chart_filter = "all"
        st.rerun()
    
    # Row 2: Category buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬈ Gaining Momentum & Outperforming", use_container_width=True, key="filter_gain_out"):
            st.session_state.chart_filter = "gaining_mom_outperforming"
            st.rerun()
    with col2:
        if st.button("⬉ Gaining Momentum & Underperforming", use_container_width=True, key="filter_gain_under"):
            st.session_state.chart_filter = "gaining_mom_underperforming"
            st.rerun()
    
    # Row 3: Category buttons
    col3, col4 = st.columns(2)
    with col3:
        if st.button("⬊ Losing Momentum & Outperforming", use_container_width=True, key="filter_lose_out"):
            st.session_state.chart_filter = "losing_mom_outperforming"
            st.rerun()
    with col4:
        if st.button("⬋ Losing Momentum & Underperforming", use_container_width=True, key="filter_lose_under"):
            st.session_state.chart_filter = "losing_mom_underperforming"
            st.rerun()
    
    # Initialize filter if not set
    if 'chart_filter' not in st.session_state:
        st.session_state.chart_filter = "all"
    
    # Apply filter to theme map
    if st.session_state.chart_filter == "all":
        filtered_map_chart = filtered_map
        st.caption(f"Showing all {len(filtered_map_chart)} themes")
    else:
        # Get themes in selected category
        selected_themes = [t['theme'] for t in categories.get(st.session_state.chart_filter, [])]
        filtered_map_chart = {k: v for k, v in filtered_map.items() if k in selected_themes}
        
        # Get category name for display
        category_names = {
            'gaining_mom_outperforming': '⬈ Gaining Momentum & Outperforming',
            'gaining_mom_underperforming': '⬉ Gaining Momentum & Underperforming',
            'losing_mom_outperforming': '⬊ Losing Momentum & Outperforming',
            'losing_mom_underperforming': '⬋ Losing Momentum & Underperforming'
        }
        st.caption(f"Showing {len(filtered_map_chart)} themes in {category_names.get(st.session_state.chart_filter, 'category')}")
    
    # Display chart with filtered themes
    chart_placeholder = st.empty()
    with chart_placeholder:
        fig = us.plot_simple_rrg(etf_data_cache, filtered_map_chart, view_key, st.session_state.sector_trails)
        chart_event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points"
        )
    
    # Handle chart selection
    if chart_event and chart_event.selection and chart_event.selection.points:
        point = chart_event.selection.points[0]
        if "customdata" in point:
            st.session_state.sector_target = point["customdata"]
        elif "text" in point:
            st.session_state.sector_target = point["text"]
    
    st.divider()

    # --- 7. SECTOR OVERVIEW ---
    st.subheader("📊 Sector Overview")
    
    # Help section
    col_help1, col_help2 = st.columns([4, 1])
    with col_help2:
        with st.popover("📖 How Categories Work", use_container_width=True):
            st.markdown("""
            ### Understanding Momentum & Performance Categories
            
            Sectors are categorized based on their **10-day trend direction**:
            
            **⬈ Gaining Momentum & Outperforming**
            - Moving up AND right on RRG chart
            - Both accelerating AND outperforming benchmark
            → **Best opportunity** - sector gaining strength
            
            **⬉ Gaining Momentum & Underperforming**
            - Moving up but still on left side
            - Accelerating but still behind benchmark
            → **Potential reversal** - watch for breakout
            
            **⬊ Losing Momentum & Outperforming**
            - Moving down but still on right side
            - Decelerating but still ahead of benchmark
            → **Topping** - take profits, avoid new entries
            
            **⬋ Losing Momentum & Underperforming**
            - Moving down AND left on RRG chart
            - Both decelerating AND underperforming
            → **Avoid** - sector in decline
            
            ---
            
            **5-Day Confirmation** shows if short-term trend supports the 10-day direction:
            - "5d accelerating ahead" = Very strong ⭐⭐⭐
            - "5d confirming trend" = Strong ⭐⭐
            - "5d lagging behind" = Weak ⭐
            """)
            st.markdown("---")
            if st.button("📖 View All Possible Combinations", use_container_width=True):
                st.session_state.show_full_guide = True
                st.rerun()
    
    # Show full combinations guide if requested
    if st.session_state.get('show_full_guide', False):
        with st.expander("📖 All 12 Possible Combinations", expanded=True):
            if st.button("✖️ Close Guide"):
                st.session_state.show_full_guide = False
                st.rerun()
            
            st.markdown("""
            ## Complete Category Guide
            
            Each of the 4 main categories can have 3 confirmation states from the 5-day window.
            
            ### 1. ⬈ Gaining Momentum & Outperforming
            
            **Best case - sector improving on both axes**
            
            - **1a. 5d accelerating ahead** ⭐⭐⭐
              - 10d: Moving up-right
              - 5d: Even MORE up-right
              - **Action:** Strong buy - momentum building fast
              - **Example:** Tech sector breaking out with volume
            
            - **1b. 5d confirming trend** ⭐⭐
              - 10d: Moving up-right
              - 5d: Also up-right, tracking 10d
              - **Action:** Buy - steady improvement
              - **Example:** Tech in consistent uptrend
            
            - **1c. 5d lagging behind** ⭐
              - 10d: Moving up-right
              - 5d: Behind 10d (pullback)
              - **Action:** Caution - might be losing steam
              - **Example:** Tech taking a breather
            
            ---
            
            ### 2. ⬉ Gaining Momentum & Underperforming
            
            **Bottoming - picking up speed but still behind benchmark**
            
            - **2a. 5d accelerating ahead** 🔄⭐
              - 10d: Moving up but left
              - 5d: Accelerating faster
              - **Action:** Watch closely - reversal starting
              - **Example:** Beaten-down sector showing life
            
            - **2b. 5d confirming trend** 🔄
              - 10d: Moving up but left
              - 5d: Also moving up-left
              - **Action:** Early reversal stage
              - **Example:** Weak sector starting to improve
            
            - **2c. 5d lagging behind** 🔄
              - 10d: Moving up but left
              - 5d: Not keeping pace
              - **Action:** False start - not ready
              - **Example:** Weak sector with brief bounce
            
            ---
            
            ### 3. ⬊ Losing Momentum & Outperforming
            
            **Topping - still ahead of benchmark but decelerating**
            
            - **3a. 5d accelerating ahead** ⚠️
              - 10d: Moving right but down
              - 5d: Ahead of 10d
              - **Action:** Possible last push up
              - **Example:** Leader showing one more surge
            
            - **3b. 5d confirming trend** ⚠️⚠️
              - 10d: Moving right but down
              - 5d: Also moving right-down
              - **Action:** Take profits - top is forming
              - **Example:** Strong sector losing steam
            
            - **3c. 5d lagging behind** ⚠️⚠️⚠️
              - 10d: Moving right but down
              - 5d: Even weaker
              - **Action:** Avoid - topping accelerating
              - **Example:** Leader rolling over
            
            ---
            
            ### 4. ⬋ Losing Momentum & Underperforming
            
            **Worst case - decline on both axes**
            
            - **4a. 5d accelerating ahead** ❌
              - 10d: Moving down-left
              - 5d: Less bad than 10d
              - **Action:** Still avoid, but may bottom soon
              - **Example:** Downtrend slowing
            
            - **4b. 5d confirming trend** ❌❌
              - 10d: Moving down-left
              - 5d: Also down-left
              - **Action:** Avoid - consistent weakness
              - **Example:** Weak sector staying weak
            
            - **4c. 5d lagging behind** ❌❌❌
              - 10d: Moving down-left
              - 5d: Even worse
              - **Action:** Avoid strongly - accelerating lower
              - **Example:** Sector in free fall
            
            ---
            
            ## Key Insights
            
            **Best Setups:**
            - ⬈ with 5d accelerating = Strongest momentum
            - ⬉ with 5d accelerating = Early reversal catch
            
            **Profit-Taking Signals:**
            - ⬊ with any 5d = Momentum fading
            
            **Stay Away:**
            - ⬋ with any 5d = Both metrics declining
            """)
    
    # Get momentum/performance categories
    categories = us.get_momentum_performance_categories(etf_data_cache, theme_map)
    
    # --- CATEGORY 1: Gaining Momentum & Outperforming ---
    if categories['gaining_mom_outperforming']:
        st.success(f"⬈ **GAINING MOMENTUM & OUTPERFORMING** ({len(categories['gaining_mom_outperforming'])} sectors)")
        st.caption("✅ **Best Opportunities** - Sectors accelerating with momentum building. 🆕 Day 1 = Fresh entry!")
        
        data = []
        for theme_info in categories['gaining_mom_outperforming']:
            # Highlight fresh entries (Day 1-2)
            days = theme_info['days_in_category']
            if days == 1:
                days_display = "🆕 Day 1"
            elif days == 2:
                days_display = "⭐ Day 2"
            else:
                days_display = f"Day {days}"
            
            data.append({
                "Sector": theme_info['theme'],
                "Days": days_display,
                "Category": theme_info['arrow'] + " " + theme_info['category'],
                "5d": theme_info['quadrant_5d'],
                "10d": theme_info['quadrant_10d'],
                "20d": theme_info['quadrant_20d'],
                "Why Selected": theme_info['reason']
            })
        
        # Sort by days (fresh first)
        df_display = pd.DataFrame(data)
        df_display['_days_sort'] = df_display['Days'].str.extract(r'(\d+)').astype(int)
        df_display = df_display.sort_values('_days_sort').drop('_days_sort', axis=1)
        
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Days": st.column_config.TextColumn("Days", help="Consecutive days in this category", width="small")
            }
        )
    else:
        st.info("⬈ **GAINING MOMENTUM & OUTPERFORMING** - No sectors currently in this category")
    
    # --- CATEGORY 2: Gaining Momentum & Underperforming ---
    if categories['gaining_mom_underperforming']:
        st.info(f"⬉ **GAINING MOMENTUM & UNDERPERFORMING** ({len(categories['gaining_mom_underperforming'])} sectors)")
        st.caption("🔄 **Potential Reversals** - Sectors bottoming, watch for breakout. 🆕 Day 1 = Fresh reversal!")
        
        data = []
        for theme_info in categories['gaining_mom_underperforming']:
            days = theme_info['days_in_category']
            if days == 1:
                days_display = "🆕 Day 1"
            elif days == 2:
                days_display = "⭐ Day 2"
            else:
                days_display = f"Day {days}"
            
            data.append({
                "Sector": theme_info['theme'],
                "Days": days_display,
                "Category": theme_info['arrow'] + " " + theme_info['category'],
                "5d": theme_info['quadrant_5d'],
                "10d": theme_info['quadrant_10d'],
                "20d": theme_info['quadrant_20d'],
                "Why Selected": theme_info['reason']
            })
        
        df_display = pd.DataFrame(data)
        df_display['_days_sort'] = df_display['Days'].str.extract(r'(\d+)').astype(int)
        df_display = df_display.sort_values('_days_sort').drop('_days_sort', axis=1)
        
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Days": st.column_config.TextColumn("Days", help="Consecutive days in this category", width="small")
            }
        )
    else:
        st.info("⬉ **GAINING MOMENTUM & UNDERPERFORMING** - No sectors currently in this category")
    
    # --- CATEGORY 3: Losing Momentum & Outperforming ---
    if categories['losing_mom_outperforming']:
        st.warning(f"⬊ **LOSING MOMENTUM & OUTPERFORMING** ({len(categories['losing_mom_outperforming'])} sectors)")
        st.caption("⚠️ **Topping** - Take profits, avoid new entries. 🆕 Day 1 = Just started losing steam")
        
        data = []
        for theme_info in categories['losing_mom_outperforming']:
            days = theme_info['days_in_category']
            if days == 1:
                days_display = "🆕 Day 1"
            elif days == 2:
                days_display = "⭐ Day 2"
            else:
                days_display = f"Day {days}"
            
            data.append({
                "Sector": theme_info['theme'],
                "Days": days_display,
                "Category": theme_info['arrow'] + " " + theme_info['category'],
                "5d": theme_info['quadrant_5d'],
                "10d": theme_info['quadrant_10d'],
                "20d": theme_info['quadrant_20d'],
                "Why Selected": theme_info['reason']
            })
        
        df_display = pd.DataFrame(data)
        df_display['_days_sort'] = df_display['Days'].str.extract(r'(\d+)').astype(int)
        df_display = df_display.sort_values('_days_sort').drop('_days_sort', axis=1)
        
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Days": st.column_config.TextColumn("Days", help="Consecutive days in this category", width="small")
            }
        )
    else:
        st.info("⬊ **LOSING MOMENTUM & OUTPERFORMING** - No sectors currently in this category")
    
    # --- CATEGORY 4: Losing Momentum & Underperforming ---
    if categories['losing_mom_underperforming']:
        st.error(f"⬋ **LOSING MOMENTUM & UNDERPERFORMING** ({len(categories['losing_mom_underperforming'])} sectors)")
        st.caption("❌ **Avoid** - Sectors declining on both metrics")
        
        data = []
        for theme_info in categories['losing_mom_underperforming']:
            days = theme_info['days_in_category']
            if days == 1:
                days_display = "🆕 Day 1"
            elif days == 2:
                days_display = "⭐ Day 2"
            else:
                days_display = f"Day {days}"
            
            data.append({
                "Sector": theme_info['theme'],
                "Days": days_display,
                "Category": theme_info['arrow'] + " " + theme_info['category'],
                "5d": theme_info['quadrant_5d'],
                "10d": theme_info['quadrant_10d'],
                "20d": theme_info['quadrant_20d'],
                "Why Selected": theme_info['reason']
            })
        
        df_display = pd.DataFrame(data)
        df_display['_days_sort'] = df_display['Days'].str.extract(r'(\d+)').astype(int)
        df_display = df_display.sort_values('_days_sort').drop('_days_sort', axis=1)
        
        st.dataframe(
            df_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Days": st.column_config.TextColumn("Days", help="Consecutive days in this category", width="small")
            }
        )
    else:
        st.info("⬋ **LOSING MOMENTUM & UNDERPERFORMING** - No sectors currently in this category")
    
    st.markdown("---")
    
    st.subheader(f"📊 Stock Analysis")
    
    # Theme selector with "All" option (unique key)
    all_themes = ["All"] + sorted(theme_map.keys())
    
    # Initialize sector_target if not exists
    if 'sector_target' not in st.session_state:
        st.session_state.sector_target = "All"
    
    selected_theme = st.selectbox(
        "Select Theme",
        all_themes,
        index=all_themes.index(st.session_state.sector_target) if st.session_state.sector_target in all_themes else 0,
        key="stock_theme_selector_unique"
    )
    
    # Update session state
    st.session_state.sector_target = selected_theme
    
    # Get momentum/performance categories for theme categorization
    categories = us.get_momentum_performance_categories(etf_data_cache, theme_map)
    
    # Build theme -> category mapping
    theme_category_map = {}
    for theme_info in categories.get('gaining_mom_outperforming', []):
        theme_category_map[theme_info['theme']] = "⬈ Gaining Momentum & Outperforming"
    for theme_info in categories.get('gaining_mom_underperforming', []):
        theme_category_map[theme_info['theme']] = "⬉ Gaining Momentum & Underperforming"
    for theme_info in categories.get('losing_mom_outperforming', []):
        theme_category_map[theme_info['theme']] = "⬊ Losing Momentum & Outperforming"
    for theme_info in categories.get('losing_mom_underperforming', []):
        theme_category_map[theme_info['theme']] = "⬋ Losing Momentum & Underperforming"
    
    # Filter stocks for selected theme(s)
    if selected_theme == "All":
        # Get all stocks and their themes
        stock_theme_pairs = []
        for _, row in uni_df[uni_df['Role'] == 'Stock'].iterrows():
            stock_theme_pairs.append((row['Ticker'], row['Theme']))
    else:
        # Get stocks for selected theme
        stock_theme_pairs = []
        for _, row in uni_df[(uni_df['Theme'] == selected_theme) & (uni_df['Role'] == 'Stock')].iterrows():
            stock_theme_pairs.append((row['Ticker'], row['Theme']))
    
    if not stock_theme_pairs:
        st.info(f"No stocks found")
        return
    
    # Build data for all stock-theme combinations
    stock_data = []
    
    with st.spinner(f"Loading {len(stock_theme_pairs)} stock-theme combinations..."):
        for stock, stock_theme in stock_theme_pairs:
            sdf = etf_data_cache.get(stock)
            
            if sdf is None or sdf.empty or len(sdf) < 20:
                continue
            
            try:
                # Volume filter
                avg_vol = sdf['Volume'].tail(20).mean()
                avg_price = sdf['Close'].tail(20).mean()
                
                if (avg_vol * avg_price) < us.MIN_DOLLAR_VOLUME:
                    continue
                
                last = sdf.iloc[-1]
                
                # Get theme-specific metrics
                alpha_5d = last.get(f"Alpha_Short_{stock_theme}", 0)
                alpha_10d = last.get(f"Alpha_Med_{stock_theme}", 0)
                alpha_20d = last.get(f"Alpha_Long_{stock_theme}", 0)
                beta = last.get(f"Beta_{stock_theme}", 1.0)
                
                stock_data.append({
                    "Ticker": stock,
                    "Theme": stock_theme,
                    "Theme Category": theme_category_map.get(stock_theme, "Unknown"),
                    "Price": last['Close'],
                    "Beta": beta,
                    "Alpha 5d": alpha_5d,
                    "Alpha 10d": alpha_10d,
                    "Alpha 20d": alpha_20d,
                    "RVOL 5d": last.get('RVOL_Short', 0),
                    "RVOL 10d": last.get('RVOL_Med', 0),
                    "RVOL 20d": last.get('RVOL_Long', 0),
                    "8 EMA": get_ma_signal(last['Close'], last.get('Ema8', 0)),
                    "21 EMA": get_ma_signal(last['Close'], last.get('Ema21', 0)),
                    "50 MA": get_ma_signal(last['Close'], last.get('Sma50', 0)),
                    "200 MA": get_ma_signal(last['Close'], last.get('Sma200', 0)),
                })
                
            except Exception as e:
                continue

    if not stock_data:
        st.info(f"No stocks found (or filtered by volume).")
        return
    
    df_stocks = pd.DataFrame(stock_data)
    
    # --- FILTER BUILDER ---
    st.markdown("### 🔍 Custom Filters")
    st.caption("Build up to 5 filters. Filters apply automatically as you change them.")
    
    # Filterable columns (numeric and categorical)
    numeric_columns = ["Alpha 5d", "Alpha 10d", "Alpha 20d", "RVOL 5d", "RVOL 10d", "RVOL 20d"]
    categorical_columns = ["Theme", "Theme Category"]
    all_filter_columns = numeric_columns + categorical_columns
    
    # Get unique values for categorical columns
    unique_themes = sorted(df_stocks['Theme'].unique().tolist())
    unique_categories = sorted(df_stocks['Theme Category'].unique().tolist())
    
    # Initialize default filters on first load
    if 'default_filters_set' not in st.session_state:
        st.session_state.default_filters_set = True
        st.session_state.filter_defaults = {
            0: {'column': 'Alpha 5d', 'operator': '>=', 'type': 'Number', 'value': 3.0},
            1: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Number', 'value': 1.3},
            2: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Column', 'value_column': 'RVOL 10d'},
            3: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬈ Gaining Momentum & Outperforming', 'logic': 'OR'},
            4: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬉ Gaining Momentum & Underperforming'}
        }
    
    # Create 5 filter rows (always visible)
    filters = []
    
    for i in range(5):
        cols = st.columns([0.20, 0.08, 0.22, 0.35, 0.15])
        
        # Get default for this filter if exists
        default = st.session_state.filter_defaults.get(i, {})
        default_column = default.get('column')
        default_operator = default.get('operator', '>=')
        default_type = default.get('type', 'Number')
        default_value = default.get('value', 0.0)
        default_value_column = default.get('value_column', 'Alpha 10d')
        default_value_cat = default.get('value_cat', '')
        default_logic = default.get('logic', 'AND')
        
        with cols[0]:
            # Set default index
            if default_column and default_column in all_filter_columns:
                default_index = all_filter_columns.index(default_column) + 1
            else:
                default_index = 0
            
            column = st.selectbox(
                f"Filter {i+1} Column",
                [None] + all_filter_columns,
                index=default_index,
                key=f"filter_{i}_column",
                label_visibility="collapsed",
                placeholder="Select column..."
            )
        
        # Determine if column is numeric or categorical
        is_numeric = column in numeric_columns
        is_categorical = column in categorical_columns
        
        if is_numeric:
            with cols[1]:
                operator = st.selectbox(
                    "Operator",
                    [">=", "<="],
                    index=0 if default_operator == '>=' else 1,
                    key=f"filter_{i}_operator",
                    label_visibility="collapsed",
                    disabled=column is None
                )
            
            with cols[2]:
                value_type = st.radio(
                    "Type",
                    ["Number", "Column"],
                    index=0 if default_type == 'Number' else 1,
                    key=f"filter_{i}_type",
                    horizontal=True,
                    label_visibility="collapsed",
                    disabled=column is None
                )
            
            with cols[3]:
                if value_type == "Number":
                    value = st.number_input(
                        "Value",
                        value=default_value,
                        step=0.1,
                        format="%.2f",
                        key=f"filter_{i}_value",
                        label_visibility="collapsed",
                        disabled=column is None
                    )
                    value_column = None
                    value_categorical = None
                else:  # Column
                    # Get index for default column
                    if default_value_column in numeric_columns:
                        col_index = numeric_columns.index(default_value_column)
                    else:
                        col_index = 0
                    
                    value_column = st.selectbox(
                        "Compare to",
                        numeric_columns,
                        index=col_index,
                        key=f"filter_{i}_value_column",
                        label_visibility="collapsed",
                        disabled=column is None
                    )
                    value = None
                    value_categorical = None
        
        elif is_categorical:
            # For categorical columns, show = operator and dropdown
            with cols[1]:
                operator = st.selectbox(
                    "Operator",
                    ["="],
                    key=f"filter_{i}_operator_cat",
                    label_visibility="collapsed",
                    disabled=column is None
                )
            
            with cols[2]:
                st.write("")  # Placeholder
            
            with cols[3]:
                if column == "Theme":
                    # Get index for default
                    if default_value_cat in unique_themes:
                        cat_index = unique_themes.index(default_value_cat)
                    else:
                        cat_index = 0
                    
                    value_categorical = st.selectbox(
                        "Select Theme",
                        unique_themes,
                        index=cat_index,
                        key=f"filter_{i}_value_theme",
                        label_visibility="collapsed"
                    )
                elif column == "Theme Category":
                    # Get index for default
                    if default_value_cat in unique_categories:
                        cat_index = unique_categories.index(default_value_cat)
                    else:
                        cat_index = 0
                    
                    value_categorical = st.selectbox(
                        "Select Category",
                        unique_categories,
                        index=cat_index,
                        key=f"filter_{i}_value_category",
                        label_visibility="collapsed"
                    )
                else:
                    value_categorical = None
                
                value = None
                value_column = None
                value_type = "Categorical"
        
        else:
            # No column selected
            with cols[1]:
                st.write("")
            with cols[2]:
                st.write("")
            with cols[3]:
                st.write("")
            operator = None
            value = None
            value_column = None
            value_categorical = None
            value_type = None
        
        with cols[4]:
            # Logic connector (except for last filter)
            if i < 4 and column is not None:
                logic = st.radio(
                    "Logic",
                    ["AND", "OR"],
                    index=0 if default_logic == 'AND' else 1,
                    key=f"filter_{i}_logic",
                    horizontal=True,
                    label_visibility="collapsed"
                )
            else:
                logic = None
        
        # Store filter config (only if column is selected)
        if column is not None:
            filters.append({
                'column': column,
                'operator': operator,
                'value_type': value_type,
                'value': value,
                'value_column': value_column,
                'value_categorical': value_categorical,
                'logic': logic
            })
    
    # Clear filters button
    if st.button("🗑️ Clear All Filters"):
        # Clear all filter widget states
        keys_to_clear = []
        for i in range(5):
            keys_to_clear.extend([
                f"filter_{i}_column",
                f"filter_{i}_operator",
                f"filter_{i}_operator_cat",
                f"filter_{i}_type",
                f"filter_{i}_value",
                f"filter_{i}_value_column",
                f"filter_{i}_value_theme",
                f"filter_{i}_value_category",
                f"filter_{i}_logic"
            ])
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset to defaults
        st.session_state.default_filters_set = False
        st.rerun()
    
    # Apply filters automatically
    df_filtered = df_stocks.copy()
    
    if filters:
        # Separate numeric and categorical filters (keeping track of indices)
        numeric_filters = []
        categorical_filters = []
        
        for i, f in enumerate(filters):
            if f['value_type'] in ['Number', 'Column']:
                numeric_filters.append(f)
            elif f['value_type'] == 'Categorical':
                categorical_filters.append(f)
        
        # Build numeric conditions (all combined with AND)
        numeric_conditions = []
        for f in numeric_filters:
            col = f['column']
            op = f['operator']
            
            if f['value_type'] == 'Number':
                val = f['value']
                if op == '>=':
                    condition = df_filtered[col] >= val
                else:
                    condition = df_filtered[col] <= val
            else:  # Column
                val_col = f['value_column']
                if op == '>=':
                    condition = df_filtered[col] >= df_filtered[val_col]
                else:
                    condition = df_filtered[col] <= df_filtered[val_col]
            
            numeric_conditions.append(condition)
        
        # Build categorical conditions (using logic from PREVIOUS categorical filter)
        categorical_conditions = []
        for i, f in enumerate(categorical_filters):
            col = f['column']
            val_cat = f['value_categorical']
            condition = df_filtered[col] == val_cat
            
            # For first categorical filter, use None
            # For subsequent, use the previous categorical filter's logic
            if i == 0:
                logic = None
            else:
                logic = categorical_filters[i-1].get('logic', 'AND')
            
            categorical_conditions.append((condition, logic))
        
        # Combine all numeric conditions with AND
        final_condition = None
        if numeric_conditions:
            numeric_combined = numeric_conditions[0]
            for cond in numeric_conditions[1:]:
                numeric_combined = numeric_combined & cond
            final_condition = numeric_combined
        
        # Combine all categorical conditions with their logic
        if categorical_conditions:
            cat_combined = categorical_conditions[0][0]
            
            for i in range(1, len(categorical_conditions)):
                condition, logic = categorical_conditions[i]
                if logic == 'OR':
                    cat_combined = cat_combined | condition
                else:  # AND (default)
                    cat_combined = cat_combined & condition
            
            # Combine numeric and categorical with AND
            if final_condition is not None:
                final_condition = final_condition & cat_combined
            else:
                final_condition = cat_combined
        
        # Apply the filter
        if final_condition is not None:
            df_filtered = df_filtered[final_condition]
    
    # Display results
    st.markdown("---")
    st.caption(f"**Showing {len(df_filtered)} of {len(df_stocks)} stocks**")
    
    # Column configuration
    column_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Theme": st.column_config.TextColumn("Theme", width="medium"),
        "Theme Category": st.column_config.TextColumn("Theme Category", width="medium"),
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
        "Alpha 5d": st.column_config.NumberColumn("Alpha 5d", format="%+.2f%%"),
        "Alpha 10d": st.column_config.NumberColumn("Alpha 10d", format="%+.2f%%"),
        "Alpha 20d": st.column_config.NumberColumn("Alpha 20d", format="%+.2f%%"),
        "RVOL 5d": st.column_config.NumberColumn("RVOL 5d", format="%.2fx"),
        "RVOL 10d": st.column_config.NumberColumn("RVOL 10d", format="%.2fx"),
        "RVOL 20d": st.column_config.NumberColumn("RVOL 20d", format="%.2fx"),
        "8 EMA": st.column_config.TextColumn("8 EMA", width="small"),
        "21 EMA": st.column_config.TextColumn("21 EMA", width="small"),
        "50 MA": st.column_config.TextColumn("50 MA", width="small"),
        "200 MA": st.column_config.TextColumn("200 MA", width="small"),
    }
    
    st.dataframe(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )
