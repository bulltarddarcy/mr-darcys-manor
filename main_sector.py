"""
Sector Rotation App - REFACTORED VERSION
With multi-theme support, smart filters, and comprehensive scoring.
"""

import streamlit as st
import pandas as pd
import utils_sector as us

# ==========================================
# MAIN PAGE FUNCTION
# ==========================================
def run_theme_momentum_app(df_global=None):
    """
    Main entry point for Sector Rotation application.
    Features: RRG quadrant analysis, Multi-timeframe views, Stock-level alpha analysis.
    """
    st.title("🔄 Theme Momentum")
    
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
        st.session_state.sector_target = "All"
    
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
    
    # --- GLOBAL FILTER APPLICATION ---
    filtered_map = {k: v for k, v in theme_map.items() if k in sel_themes}
    timeframe_map = {"5 Days": "Short", "10 Days": "Med", "20 Days": "Long"}
    view_key = timeframe_map[st.session_state.sector_view]

    # --- 6. RRG CHART ---
    categories = us.get_momentum_performance_categories(etf_data_cache, filtered_map)
    
    st.markdown("**Filter Chart by Category:**")
    btn_cols = st.columns(5)
    filter_config = [
        (0, "🎯 All", "all"),
        (1, "⬈ Gain/Out", "gaining_mom_outperforming"),
        (2, "⬉ Gain/Under", "gaining_mom_underperforming"),
        (3, "⬊ Lose/Out", "losing_mom_outperforming"),
        (4, "⬋ Lose/Under", "losing_mom_underperforming")
    ]

    for col_idx, label, filter_key in filter_config:
        with btn_cols[col_idx]:
            if st.button(label, use_container_width=True, key=f"filter_{filter_key}"):
                st.session_state.chart_filter = filter_key
                st.rerun()
    
    if 'chart_filter' not in st.session_state:
        st.session_state.chart_filter = "all"
    
    if st.session_state.chart_filter == "all":
        filtered_map_chart = filtered_map
        st.caption(f"Showing all {len(filtered_map_chart)} themes")
    else:
        selected_themes = [t['theme'] for t in categories.get(st.session_state.chart_filter, [])]
        filtered_map_chart = {k: v for k, v in filtered_map.items() if k in selected_themes}
        st.caption(f"Showing {len(filtered_map_chart)} themes in {st.session_state.chart_filter}")
    
    # Display chart
    chart_placeholder = st.empty()
    with chart_placeholder:
        fig = us.plot_simple_rrg(etf_data_cache, filtered_map_chart, view_key, st.session_state.sector_trails)
        chart_event = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points"
        )
    
    if chart_event and chart_event.selection and chart_event.selection.points:
        point = chart_event.selection.points[0]
        if "customdata" in point:
            st.session_state.sector_target = point["customdata"]
        elif "text" in point:
            st.session_state.sector_target = point["text"]
    
    st.divider()

    # --- 7. SECTOR OVERVIEW (Refactored Loop) ---
    st.subheader("📊 Theme Categories")
    
    col_guide1, col_guide2 = st.columns([1, 1], gap="small")
    with col_guide1:
        with st.expander("📖 How Categories Work", expanded=False):
            st.markdown("""
            ### Understanding Momentum & Performance Categories
            Sectors are categorized based on their **10-day trend direction**:
            
            **⬈ Gain Mom & Outperf**
            - Moving up AND right on RRG chart
            - Both accelerating AND outperforming benchmark
            → **Best opportunity** - sector gaining strength
            
            **⬉ Gain Mom & Underperf**
            - Moving up but still on left side
            - Accelerating but still behind benchmark
            → **Potential reversal** - watch for breakout
            
            **⬊ Lose Mom & Outperf**
            - Moving down but still on right side
            - Decelerating but still ahead of benchmark
            → **Topping** - take profits, avoid new entries
            
            **⬋ Lose Mom & Underperf**
            - Moving down AND left on RRG chart
            - Both decelerating AND underperforming
            → **Avoid** - sector in decline
            
            ---
            **5-Day Confirmation** shows if short-term trend supports the 10-day direction:
            - "5d accelerating ahead" = Very strong ⭐⭐⭐
            - "5d confirming trend" = Strong ⭐⭐
            - "5d lagging behind" = Weak ⭐
            """)
            
    with col_guide2:
        if st.button("📖 View All Possible Combinations", use_container_width=True):
            st.session_state.show_full_guide = True
            st.rerun()
    
    if st.session_state.get('show_full_guide', False):
        with st.expander("📖 All 12 Possible Combinations", expanded=True):
            if st.button("✖️ Close Guide"):
                st.session_state.show_full_guide = False
                st.rerun()
            st.markdown("""
            ## Complete Category Guide
            ### 1. ⬈ Gain Mom & Outperf (Best case)
            - **1a. 5d accelerating ahead** ⭐⭐⭐ (Strong buy)
            - **1b. 5d confirming trend** ⭐⭐ (Buy)
            - **1c. 5d lagging behind** ⭐ (Caution)
            
            ### 2. ⬉ Gain Mom & Underperf (Bottoming)
            - **2a. 5d accelerating ahead** 🔄⭐ (Watch closely)
            - **2b. 5d confirming trend** 🔄 (Early reversal)
            - **2c. 5d lagging behind** 🔄 (False start)
            
            ### 3. ⬊ Lose Mom & Outperf (Topping)
            - **3a. 5d accelerating ahead** ⚠️ (Possible last push)
            - **3b. 5d confirming trend** ⚠️⚠️ (Take profits)
            - **3c. 5d lagging behind** ⚠️⚠️⚠️ (Avoid - accelerating down)
            
            ### 4. ⬋ Lose Mom & Underperf (Worst case)
            - **4a. 5d accelerating ahead** ❌ (Still avoid)
            - **4b. 5d confirming trend** ❌❌ (Avoid)
            - **4c. 5d lagging behind** ❌❌❌ (Avoid strongly)
            """)

    # --- CATEGORY DISPLAY LOOP ---
    # Define metadata for each quadrant to avoid repetition
    quadrant_meta = [
        ('gaining_mom_outperforming', '⬈ GAIN MOM & OUTPERF', 'success', '✅ Best Opportunities - Sectors accelerating with momentum building. 🆕 Day 1 = Fresh entry!'),
        ('gaining_mom_underperforming', '⬉ GAIN MOM & UNDERPERF', 'info', '🔄 Potential Reversals - Sectors bottoming, watch for breakout. 🆕 Day 1 = Fresh reversal!'),
        ('losing_mom_outperforming', '⬊ LOSE MOM & OUTPERF', 'warning', '⚠️ Topping - Take profits, avoid new entries. 🆕 Day 1 = Just started losing steam'),
        ('losing_mom_underperforming', '⬋ LOSE MOM & UNDERPERF', 'error', '❌ Avoid - Sectors declining on both metrics')
    ]

    for key, title, style_func_name, caption in quadrant_meta:
        items = categories.get(key, [])
        style_func = getattr(st, style_func_name)
        
        if items:
            style_func(f"**{title}** ({len(items)} sectors)")
            st.caption(caption)
            
            data = []
            for theme_info in items:
                days = theme_info['days_in_category']
                days_display = "🆕 Day 1" if days == 1 else "⭐ Day 2" if days == 2 else f"Day {days}"
                
                data.append({
                    "Sector": theme_info['theme'],
                    "Days": days_display,
                    "Category": theme_info['display_category'], # Used shortened name from utils
                    "5d": theme_info['quadrant_5d'],
                    "10d": theme_info['quadrant_10d'],
                    "20d": theme_info['quadrant_20d'],
                    "Why Selected": theme_info['reason']
                })
            
            df_display = pd.DataFrame(data)
            df_display['_days_sort'] = df_display['Days'].str.extract(r'(\d+)').astype(int)
            df_display = df_display.sort_values('_days_sort').drop('_days_sort', axis=1)
            
            st.dataframe(
                df_display, hide_index=True, use_container_width=True,
                column_config={"Days": st.column_config.TextColumn("Days", help="Consecutive days in this category", width="small")}
            )
        else:
            style_func(f"**{title}** - No sectors currently in this category")

    st.markdown("---")
    
    st.subheader(f"📊 Stock Analysis")

    # ==========================================
    # 1. DEFINE CALLBACKS & INITIALIZE (MOVED UP)
    # ==========================================
    # We must define these and run initialization BEFORE widgets are drawn
    # to avoid the "cannot be modified after instantiation" error.

    def cb_set_defaults():
        """Resets filters and settings to safe defaults."""
        # Update settings checkboxes
        st.session_state.opt_show_divergences = False
        st.session_state.opt_show_mkt_caps = False
        st.session_state.opt_show_biotech = False
        
        # Update filter flags
        st.session_state.filters_were_cleared = False
        st.session_state.default_filters_set = True
        
        # Set Default Filters
        st.session_state.filter_defaults = {
            0: {'column': 'Alpha 5d', 'operator': '>=', 'type': 'Number', 'value': 3.0},
            1: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Number', 'value': 1.3},
            2: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Column', 'value_column': 'RVOL 10d'},
            3: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬈ Gain Mom & Outperf', 'logic': 'OR'},
            4: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬉ Gain Mom & Underperf'},
            5: {}, 6: {}, 7: {}
        }
        # Clear custom user inputs (keys starting with filter_)
        for k in [k for k in st.session_state if k.startswith('filter_') and k != 'filter_defaults']:
            del st.session_state[k]

    def cb_darcy_special():
        """Applies Darcy Special settings."""
        # Enable settings
        st.session_state.opt_show_divergences = True
        st.session_state.opt_show_mkt_caps = True
        
        # Reset flags
        st.session_state.filters_were_cleared = False
        st.session_state.default_filters_set = True
        
        # Set Darcy Filters
        st.session_state.filter_defaults = {
            0: {'column': 'Alpha 5d', 'operator': '>=', 'type': 'Number', 'value': 1.2},
            1: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Number', 'value': 1.2},
            2: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Column', 'value_column': 'RVOL 10d'},
            3: {'column': 'Market Cap (B)', 'operator': '>=', 'type': 'Number', 'value': 5.0},
            4: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬈ Gain Mom & Outperf', 'logic': 'OR'},
            5: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬉ Gain Mom & Underperf', 'logic': 'OR'},
            6: {'column': 'Div', 'operator': '=', 'type': 'Categorical', 'value_cat': '🟢 Bullish'},
            7: {}
        }

    def cb_clear_filters():
        """Clears all filters."""
        for k in [k for k in st.session_state if k.startswith('filter_') or k in ['filter_defaults', 'default_filters_set']]:
            del st.session_state[k]
        st.session_state.filters_were_cleared = True

    # --- RUN INITIALIZATION NOW ---
    # This must happen before the checkboxes below are rendered!
    if 'filter_defaults' not in st.session_state:
        st.session_state.filter_defaults = {i: {} for i in range(8)}
    
    if 'default_filters_set' not in st.session_state:
        if not st.session_state.get('filters_were_cleared', False):
            cb_set_defaults() # This is safe now because widgets aren't drawn yet

    # ==========================================
    # 2. RENDER UI (CONTROLS)
    # ==========================================
    
    all_themes = ["All"] + sorted(filtered_map.keys())
    if 'sector_target' not in st.session_state: st.session_state.sector_target = "All"
    
    col_sel, col_opt = st.columns([1, 1])
    with col_sel:
        st.session_state.sector_target = st.selectbox(
            "Select Theme", all_themes,
            index=all_themes.index(st.session_state.sector_target) if st.session_state.sector_target in all_themes else 0,
            key="stock_theme_selector_unique"
        )
    with col_opt:
        st.caption("Additional Settings")
        c_opt1, c_opt2, c_opt3 = st.columns(3)
        # These widgets will now pick up the state set by cb_set_defaults above
        show_divergences = c_opt1.checkbox("Show Divergences", key="opt_show_divergences", help="Slower: Scans RSI history for divergences.")
        show_mkt_caps = c_opt2.checkbox("Show Market Caps", key="opt_show_mkt_caps", help="Slower: Fetches live Market Cap.")
        show_biotech = c_opt3.checkbox("Show Biotech", key="opt_show_biotech", value=False, help="Include Biotech theme.")

    # ==========================================
    # 3. PROCESS DATA
    # ==========================================

    # Create a map of Theme -> Display Category for the table
    theme_cat_map = {}
    for cat_list in categories.values():
        for t in cat_list:
            theme_cat_map[t['theme']] = t['display_category']

    selected_theme = st.session_state.sector_target
    
    # Get Pairs
    if selected_theme == "All":
        stock_theme_pairs = [(row['Ticker'], row['Theme']) for _, row in uni_df[uni_df['Role'] == 'Stock'].iterrows() if row['Theme'] in filtered_map]
    else:
        stock_theme_pairs = [(row['Ticker'], row['Theme']) for _, row in uni_df[(uni_df['Theme'] == selected_theme) & (uni_df['Role'] == 'Stock')].iterrows()]
    
    if not stock_theme_pairs:
        st.info(f"No stocks found")
        return

    # --- CALL UTILS FOR HEAVY LIFTING ---
    df_stocks = us.analyze_stocks_batch(
        etf_data_cache, stock_theme_pairs, show_divergences, 
        show_mkt_caps, show_biotech, theme_cat_map
    )

    if df_stocks.empty:
        st.info(f"No stocks found (or filtered by volume/Biotech setting).")
        return
    
    # ==========================================
    # 4. FILTER BUILDER UI
    # ==========================================
    st.markdown("### 🔍 Custom Filters")
    with st.expander("ℹ️ How are RVOL and Alpha calculated?"):
        st.markdown(r"""
        ### **1. RVOL (Relative Volume) - 5, 10, & 20 Days**
        The calculation establishes a daily relative volume ratio and averages it over specific timeframes.
        **Daily Calculation:** $\text{Daily RVOL} = \frac{\text{Volume}}{\text{Avg Volume (Last 20 Days)}}$
        **Timeframes:** Average of Daily RVOL over 5/10/20 days.
        > An RVOL of **1.3** means 130% of normal volume.
        
        ### **2. Alpha - 5, 10, & 20 Days**
        Excess return relative to Sector ETF, adjusted for Beta.
        1. **Beta ($\beta$):** 60-day rolling correlation.
        2. **Expected Return:** $\text{Sector \%} \times \beta$
        3. **Alpha:** $\text{Stock \%} - \text{Expected Return}$
        > Alpha 5d of **3.0** means outperformance by 3% over the week.
        """)
    
    st.caption("Build up to 8 filters. Filters apply automatically as you change them.")
    
    # Filter Buttons
    col_d, col_s, col_c = st.columns(3)
    # Note: We don't need to define callbacks here anymore, they are defined at the top
    col_d.button("↺ Set to Defaults", type="secondary", use_container_width=True, on_click=cb_set_defaults)
    col_s.button("✨ Darcy Special", type="primary", use_container_width=True, on_click=cb_darcy_special)
    col_c.button("🗑️ Clear Filters", type="secondary", use_container_width=True, on_click=cb_clear_filters)
    
    # Columns for filtering
    numeric_columns = ["Price", "Market Cap (B)", "Beta", "Alpha 5d", "Alpha 10d", "Alpha 20d", "RVOL 5d", "RVOL 10d", "RVOL 20d"]
    categorical_columns = ["Theme", "Theme Category", "Div", "8 EMA", "21 EMA", "50 MA", "200 MA"]
    all_filter_columns = numeric_columns + categorical_columns
    
    # Filter Construction Loop
    filters = []
    for i in range(8):
        cols = st.columns(5)
        default = st.session_state.filter_defaults.get(i, {})
        
        # Column Select
        idx = all_filter_columns.index(default.get('column')) + 1 if default.get('column') in all_filter_columns else 0
        column = cols[0].selectbox(f"F{i+1}", [None] + all_filter_columns, index=idx, key=f"filter_{i}_column", label_visibility="collapsed", placeholder="Column...")
        
        if column:
            is_numeric = column in numeric_columns
            
            # Operator
            ops = [">=", "<="] if is_numeric else ["="]
            def_op_idx = 0 if default.get('operator', '>=') == '>=' else 1
            operator = cols[1].selectbox("Op", ops, index=def_op_idx, key=f"filter_{i}_operator", label_visibility="collapsed")
            
            val_type, val, val_col, val_cat = "Number", None, None, None
            
            if is_numeric:
                # Type (Number/Column)
                t_idx = 0 if default.get('type', 'Number') == 'Number' else 1
                val_type = cols[2].radio("T", ["Number", "Column"], index=t_idx, key=f"filter_{i}_type", horizontal=True, label_visibility="collapsed")
                
                if val_type == "Number":
                    val = cols[3].number_input("V", value=default.get('value', 0.0), step=0.1, format="%.2f", key=f"filter_{i}_value", label_visibility="collapsed")
                else:
                    def_vc = default.get('value_column')
                    vc_idx = numeric_columns.index(def_vc) if def_vc in numeric_columns else 0
                    val_col = cols[3].selectbox("C", numeric_columns, index=vc_idx, key=f"filter_{i}_val_col", label_visibility="collapsed")
            else:
                # Categorical Value
                val_type = "Categorical"
                uniques = sorted(df_stocks[column].astype(str).unique())
                def_cat = default.get('value_cat')
                cat_idx = uniques.index(def_cat) if def_cat in uniques else 0
                val_cat = cols[3].selectbox("V", uniques, index=cat_idx, key=f"filter_{i}_val_cat", label_visibility="collapsed")

            # Logic (AND/OR)
            logic = None
            if i < 7:
                l_idx = 0 if default.get('logic', 'AND') == 'AND' else 1
                logic = cols[4].radio("L", ["AND", "OR"], index=l_idx, key=f"filter_{i}_logic", horizontal=True, label_visibility="collapsed")
            
            filters.append({
                'column': column, 'operator': operator, 'value_type': val_type,
                'value': val, 'value_column': val_col, 'value_categorical': val_cat,
                'logic': logic
            })

    # Apply Filters via Utils
    df_filtered = us.apply_stock_filters(df_stocks, filters)
    
    # Display Results
    st.markdown("---")
    st.caption(f"**Showing {len(df_filtered)} of {len(df_stocks)} stocks**")
    
    column_config = {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Theme": st.column_config.TextColumn("Theme", width="medium"),
        "Theme Category": st.column_config.TextColumn("Theme Category", width="medium"),
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Market Cap (B)": st.column_config.NumberColumn("Mkt Cap", format="$%.1fB"),
        "Beta": st.column_config.NumberColumn("Beta", format="%.2f"),
        "Alpha 5d": st.column_config.NumberColumn("Alpha 5d", format="%+.2f%%"),
        "Alpha 10d": st.column_config.NumberColumn("Alpha 10d", format="%+.2f%%"),
        "Alpha 20d": st.column_config.NumberColumn("Alpha 20d", format="%+.2f%%"),
        "RVOL 5d": st.column_config.NumberColumn("RVOL 5d", format="%.2fx"),
        "RVOL 10d": st.column_config.NumberColumn("RVOL 10d", format="%.2fx"),
        "RVOL 20d": st.column_config.NumberColumn("RVOL 20d", format="%.2fx"),
        "Div": st.column_config.TextColumn("Div", width="small"),
        "8 EMA": st.column_config.TextColumn("8 EMA", width="small"),
        "21 EMA": st.column_config.TextColumn("21 EMA", width="small"),
        "50 MA": st.column_config.TextColumn("50 MA", width="small"),
        "200 MA": st.column_config.TextColumn("200 MA", width="small"),
    }
    
    st.dataframe(df_filtered, use_container_width=True, hide_index=True, column_config=column_config)
    
    if not df_filtered.empty:
        st.caption("Copy tickers:")
        st.code(", ".join(df_filtered['Ticker'].unique().tolist()), language="text")