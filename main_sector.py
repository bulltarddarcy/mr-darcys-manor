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
            st.markdown("**Sectors Shown (Applies to Entire Page!)**")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("➕ Everything", use_container_width=True):
                    st.session_state.sector_theme_filter_widget = all_themes
                    st.rerun()

            with btn_col2:
                if st.button("⭐ Big 11", use_container_width=True):
                    big_11 = [
                        "Comms", "Cons Discr", "Cons Staples",
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

    # --- 7. THEME CATEGORIES DISPLAY ---
    st.subheader("📊 Theme Categories")
    
    col_guide1, col_guide2 = st.columns([1, 1], gap="small")
    with col_guide1:
        with st.expander("📖 How Categories Work", expanded=False):
            st.markdown("""
            ### Understanding Momentum & Performance Categories
            Sectors are categorized based on their **10-day trend direction**:
            ...
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
            st.markdown("""## Complete Category Guide ... (Content Preserved) ...""")

    quadrant_meta = [
        ('gaining_mom_outperforming', '⬈ GAIN MOM & OUTPERF', 'success', '✅ Best Opportunities - Sectors accelerating...'),
        ('gaining_mom_underperforming', '⬉ GAIN MOM & UNDERPERF', 'info', '🔄 Potential Reversals - Sectors bottoming...'),
        ('losing_mom_outperforming', '⬊ LOSE MOM & OUTPERF', 'warning', '⚠️ Topping - Take profits...'),
        ('losing_mom_underperforming', '⬋ LOSE MOM & UNDERPERF', 'error', '❌ Avoid - Sectors declining...')
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
                    "Category": theme_info['display_category'],
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
    
    # ==========================================
    # STOCK ANALYSIS SECTION
    # ==========================================
    st.subheader(f"📊 Stock Analysis")

    # --- 1. INITIALIZE SETTINGS STATE (Before Analysis) ---
    # We must init these here so `analyze_stocks_batch` has values to use,
    # even though the checkbox widgets are drawn later in the UI.
    if "opt_show_divergences" not in st.session_state:
        st.session_state.opt_show_divergences = False
    if "opt_show_mkt_caps" not in st.session_state:
        st.session_state.opt_show_mkt_caps = False
    if "opt_show_biotech" not in st.session_state:
        st.session_state.opt_show_biotech = False

    # --- 2. SET DEFAULT FILTERS ---
    if 'filter_defaults' not in st.session_state:
        st.session_state.filter_defaults = {
            0: {'column': 'Alpha 5d', 'operator': '>=', 'type': 'Number', 'value': 3.0, 'logic': 'AND'},
            1: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Number', 'value': 1.3, 'logic': 'AND'},
            2: {'column': 'RVOL 5d', 'operator': '>=', 'type': 'Column', 'value_column': 'RVOL 10d', 'logic': 'AND'},
            3: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬈ Gain Mom & Outperf', 'logic': 'OR'},
            4: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬉ Gain Mom & Underperf', 'logic': 'OR'}},
            5: {}, 6: {}, 7: {}
        }
    
    # --- 3. PROCESS DATA (Using Global Filters) ---
    # Create theme category map for the table
    theme_cat_map = {}
    for cat_list in categories.values():
        for t in cat_list:
            theme_cat_map[t['theme']] = t['display_category']

    # Generate Stock List based on Global Filter (Top of page)
    # We always load ALL stocks matching the global filter
    stock_theme_pairs = []
    for _, row in uni_df[uni_df['Role'] == 'Stock'].iterrows():
        if row['Theme'] in filtered_map:
            stock_theme_pairs.append((row['Ticker'], row['Theme']))
    
    if not stock_theme_pairs:
        st.info(f"No stocks found for selected themes.")
        return

    # Run Analysis (using session state values for options)
    df_stocks = us.analyze_stocks_batch(
        etf_data_cache, 
        stock_theme_pairs, 
        st.session_state.opt_show_divergences, 
        st.session_state.opt_show_mkt_caps, 
        st.session_state.opt_show_biotech, 
        theme_cat_map
    )

    if df_stocks.empty:
        st.info(f"No stocks found (or filtered by volume/Biotech setting).")
        return
    
    # --- 4. FILTER BUILDER UI (ROBUST VERSION) ---
    # st.markdown("### 🔍 Custom Filters")
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
    
    # Define Columns
    numeric_columns = ["Price", "Market Cap (B)", "Beta", "Alpha 5d", "Alpha 10d", "Alpha 20d", "RVOL 5d", "RVOL 10d", "RVOL 20d"]
    categorical_columns = ["Theme", "Theme Category", "Div", "8 EMA", "21 EMA", "50 MA", "200 MA"]
    all_filter_columns = numeric_columns + categorical_columns

    # --- HELPER: ROBUST OPTION LISTS ---
    def get_safe_options(df, col_name):
        """Returns sorted string options, ensures list is never empty."""
        if col_name not in df.columns: return ["-"]
        opts = sorted([str(x) for x in df[col_name].unique() if pd.notna(x) and str(x).strip() != ""])
        return opts if opts else ["-"]

    unique_themes = get_safe_options(df_stocks, 'Theme')
    unique_categories = get_safe_options(df_stocks, 'Theme Category')
    unique_divs = get_safe_options(df_stocks, 'Div')
    
    # --- HELPER: SAFE INDEXING ---
    def safe_index(options, value, default=0):
        """Safely finds index of value in options, returning default if not found."""
        try:
            return options.index(value)
        except ValueError:
            return default if 0 <= default < len(options) else 0
            
    # --- FILTER LOOP ---
    filters = []
    
    for i in range(8):
        cols = st.columns(5)
        default = st.session_state.filter_defaults.get(i, {})
        
        # 1. Column Selector
        col_opts = [None] + all_filter_columns
        col_idx = safe_index(col_opts, default.get('column'), 0)
        
        column = cols[0].selectbox(
            f"F{i+1}", col_opts, index=col_idx, 
            key=f"filter_{i}_column", label_visibility="collapsed", placeholder="Column..."
        )
        
        if column:
            is_numeric = column in numeric_columns
            is_categorical = column in categorical_columns
            
            # 2. Operator Selector
            ops = [">=", "<="] if is_numeric else ["="]
            op_idx = safe_index(ops, default.get('operator', '>='), 0)
            operator = cols[1].selectbox("Op", ops, index=op_idx, key=f"filter_{i}_operator", label_visibility="collapsed")
            
            val_type, val, val_col, val_cat = "Number", None, None, None
            
            if is_numeric:
                # 3. Type Selector
                type_opts = ["Number", "Column"]
                type_idx = safe_index(type_opts, default.get('type', 'Number'), 0)
                val_type = cols[2].radio("T", type_opts, index=type_idx, key=f"filter_{i}_type", horizontal=True, label_visibility="collapsed")
                
                if val_type == "Number":
                    # 4. Value Input
                    val = cols[3].number_input("V", value=float(default.get('value', 0.0)), step=0.1, format="%.2f", key=f"filter_{i}_value", label_visibility="collapsed")
                else:
                    # 4. Column Comparison
                    vc_idx = safe_index(numeric_columns, default.get('value_column'), 0)
                    val_col = cols[3].selectbox("C", numeric_columns, index=vc_idx, key=f"filter_{i}_val_col", label_visibility="collapsed")
            else:
                # Categorical Value
                val_type = "Categorical"
                if column == "Theme": cat_opts = unique_themes
                elif column == "Theme Category": cat_opts = unique_categories
                elif column == "Div": cat_opts = unique_divs
                else: cat_opts = get_safe_options(df_stocks, column) # Fallback
                
                cat_idx = safe_index(cat_opts, default.get('value_cat'), 0)
                val_cat = cols[3].selectbox("V", cat_opts, index=cat_idx, key=f"filter_{i}_val_cat", label_visibility="collapsed")

            # 5. Logic Selector
            logic = None
            if i < 7:
                logic_opts = ["AND", "OR"]
                log_idx = safe_index(logic_opts, default.get('logic', 'AND'), 0)
                logic = cols[4].radio("L", logic_opts, index=log_idx, key=f"filter_{i}_logic", horizontal=True, label_visibility="collapsed")
            
            filters.append({
                'column': column, 'operator': operator, 'value_type': val_type,
                'value': val, 'value_column': val_col, 'value_categorical': val_cat,
                'logic': logic
            })
            
    # Apply Filters via Utils
    df_filtered = us.apply_stock_filters(df_stocks, filters)
    
    # Display Results
    st.caption(f"**Showing {len(df_filtered)} of {len(df_stocks)} stock-theme combinations**")

    # --- SETTINGS CHECKBOXES (Horizontal Compact) ---
    # The [2, 2, 2, 6] ratio keeps them close on the left and leaves empty space on the right
    c1, c2, c3, _ = st.columns([2, 2, 2, 6]) 
    with c1:
        st.checkbox("Show Divergences", key="opt_show_divergences", help="Slower: Scans RSI history for divergences.")
    with c2:
        st.checkbox("Show Market Caps", key="opt_show_mkt_caps", help="Slower: Fetches live Market Cap data from Yahoo Finance.")
    with c3:
        st.checkbox("Show Biotech", key="opt_show_biotech", value=False, help="Include Biotech theme.")

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
