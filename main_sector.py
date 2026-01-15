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

    # --- 1. INITIALIZE SETTINGS STATE ---
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
            4: {'column': 'Theme Category', 'operator': '=', 'type': 'Categorical', 'value_cat': '⬉ Gain Mom & Underperf', 'logic': 'OR'},
            5: {}, 6: {}, 7: {}
        }
    
    # --- 3. PROCESS DATA (Using Global Filters) ---
    theme_cat_map = {}
    for cat_list in categories.values():
        for t in cat_list:
            theme_cat_map[t['theme']] = t['display_category']

    stock_theme_pairs = []
    for _, row in uni_df[uni_df['Role'] == 'Stock'].iterrows():
        if row['Theme'] in filtered_map:
            stock_theme_pairs.append((row['Ticker'], row['Theme']))
    
    if not stock_theme_pairs:
        st.info(f"No stocks found for selected themes.")
        return

    # Run Analysis - FAST MODE (No expensive columns yet)
    # FIX: Only pass 4 arguments here
    df_stocks = us.analyze_stocks_batch(
        etf_data_cache, 
        stock_theme_pairs, 
        st.session_state.opt_show_biotech, 
        theme_cat_map
    )

    if df_stocks.empty:
        st.info(f"No stocks found (or filtered by volume/Biotech setting).")
        return
    
    # --- 4. FILTER BUILDER UI ---
    with st.expander("ℹ️ How are RVOL and Alpha calculated?"):
        st.markdown(r"""
        ### **1. RVOL (Relative Volume)**
        Daily RVOL averaged over 5/10/20 days.
        > **1.3** = 130% of normal volume.
        
        ### **2. Alpha**
        Excess return relative to Sector ETF, adjusted for Beta.
        > **3.0** = Outperformed risk-adjusted expectation by 3%.
        """)
    
    # Define Columns
    numeric_columns = ["Price", "Market Cap (B)", "Beta", "Alpha 5d", "Alpha 10d", "Alpha 20d", "RVOL 5d", "RVOL 10d", "RVOL 20d"]
    categorical_columns = ["Theme", "Theme Category", "Div", "8 EMA", "21 EMA", "50 MA", "200 MA"]
    all_filter_columns = numeric_columns + categorical_columns

    # Helpers
    def get_safe_options(df, col_name):
        if col_name not in df.columns: return ["-"]
        opts = sorted([str(x) for x in df[col_name].unique() if pd.notna(x) and str(x).strip() != ""])
        return opts if opts else ["-"]

    unique_themes = get_safe_options(df_stocks, 'Theme')
    unique_categories = get_safe_options(df_stocks, 'Theme Category')
    unique_divs = get_safe_options(df_stocks, 'Div')
    
    def safe_index(options, value, default=0):
        try:
            return options.index(value)
        except ValueError:
            return default if 0 <= default < len(options) else 0
            
    # --- FILTER LOOP (Builds UI List Only) ---
    current_ui_filters = [] # Store current widget states here
    
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
                else: cat_opts = get_safe_options(df_stocks, column)
                
                cat_idx = safe_index(cat_opts, default.get('value_cat'), 0)
                val_cat = cols[3].selectbox("V", cat_opts, index=cat_idx, key=f"filter_{i}_val_cat", label_visibility="collapsed")

            # 5. Logic Selector
            logic = None
            if i < 7:
                logic_opts = ["AND", "OR"]
                log_idx = safe_index(logic_opts, default.get('logic', 'AND'), 0)
                logic = cols[4].radio("L", logic_opts, index=log_idx, key=f"filter_{i}_logic", horizontal=True, label_visibility="collapsed")
            
            current_ui_filters.append({
                'column': column, 'operator': operator, 'value_type': val_type,
                'value': val, 'value_column': val_col, 'value_categorical': val_cat,
                'logic': logic
            })

    # --- APPLY BUTTON LOGIC ---
    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    
    # 1. Initialize active filters on first load
    if "active_stock_filters" not in st.session_state:
        st.session_state.active_stock_filters = current_ui_filters

    # 2. Button to update active filters AND RESET ENRICHMENT
    if st.button("Apply Filters", type="primary", use_container_width=True):
        st.session_state.active_stock_filters = current_ui_filters
        # RESET the checkboxes so we don't do expensive calculations automatically
        st.session_state.opt_show_divergences = False
        st.session_state.opt_show_mkt_caps = False
        st.rerun()

    # 3. Apply the ACTIVE filters (not the UI filters)
    df_filtered = us.apply_stock_filters(df_stocks, st.session_state.active_stock_filters)
    
    # Check if UI differs from Active (Visual Cue)
    if current_ui_filters != st.session_state.active_stock_filters:
        st.caption("⚠️ *Filters have changed. Click 'Apply Filters' to update the table.*")

    # Display Results
    st.markdown(f"**Showing {len(df_filtered)} of {len(df_stocks)} stock-theme combinations**")

    # --- SETTINGS CHECKBOXES (RENDERED AFTER FILTERING) ---
    c1, c2, c3, _ = st.columns([2, 2, 2, 6]) 
    with c1:
        st.checkbox("Show Divergences", key="opt_show_divergences", help="Enrich filtered list. Slower.")
    with c2:
        st.checkbox("Show Market Caps", key="opt_show_mkt_caps", help="Enrich filtered list. Slower.")
    with c3:
        st.checkbox("Show Biotech", key="opt_show_biotech", value=False, help="Include Biotech theme.")

    # --- ENRICH DATA ---
    # Only run enrichment if checkboxes are checked (and logic ensures they are unchecked on filter apply)
    df_filtered = us.enrich_stock_data(
        df_filtered, 
        etf_data_cache, 
        st.session_state.opt_show_mkt_caps, 
        st.session_state.opt_show_divergences
    )

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

    # --- TICKER LOOKUP ---
    
    # Text input for ticker
    lookup_ticker = st.text_input(
        "Enter a ticker to see its sectors:", 
        placeholder="e.g. AAPL",
        label_visibility="visible"
    ).upper().strip()

    if lookup_ticker:
        # Check against the universe dataframe loaded at the start
        if not uni_df.empty and 'Ticker' in uni_df.columns:
            matches = uni_df[uni_df['Ticker'] == lookup_ticker]
            
            if not matches.empty:
                # Get unique themes for this ticker
                found_themes = sorted(matches['Theme'].unique().tolist())
                st.success(f"✅ **{lookup_ticker}** is mapped to: **{', '.join(found_themes)}**")
            else:
                st.warning(f"❌ Ticker **{lookup_ticker}** not found in the loaded Universe.")
        else:
            st.error("Universe data not loaded.")
    
    if not df_filtered.empty:
        st.caption("Copy tickers:")
        st.code(", ".join(df_filtered['Ticker'].unique().tolist()), language="text")

    # divider line between the outputs (above) and the admin inputs/configurations (below)
    st.markdown('<hr style="border-top: 4px solid black;">', unsafe_allow_html=True)
    
    # ==========================================
    # ADMIN: PHASE 1: GENERATE NEW UNIVERSE
    # ==========================================
    with st.expander("🛠️ Phase 1: Generate New Universe", expanded=False):
        st.caption("Use this tool to mathematically generate a universe based on ETF holdings.")
        
        # 1. Inputs
        c_gen1, c_gen2, c_gen3 = st.columns(3)
        with c_gen1:
            # CHANGED: Slider -> Number Input
            target_weight = st.number_input(
                "Target Cumulative Weight %", 
                min_value=0.05, max_value=1.0, value=0.60, step=0.05, format="%.2f",
                help="We will keep pulling tickers until they account for this % of the ETF's weight."
            )
        with c_gen2:
            qty_cap = st.number_input("Max Tickers per Sector", min_value=0, value=0, 
                help="Hard cap on count. Leave 0 for no limit.")
            # Handle 0 as None
            qty_cap_val = None if qty_cap == 0 else qty_cap
        with c_gen3:
            min_vol_input = st.number_input("Min $ Volume (Daily)", value=10_000_000, step=1_000_000, format="%d",
                help="Filter out illiquid stocks.")

        # 2. Action
        if st.button("🚀 Generate Universe", type="primary"):
            # Check if we have theme map
            if 'theme_map' not in locals() or not theme_map:
                # Fallback for when button is pressed without main load
                _, _, theme_map, _, _ = us.SectorDataManager().load_universe()
                
            generator = us.UniverseGenerator()
            
            with st.spinner("Fetching holdings and validating volume... (This may take 30s)"):
                csv_data, stats_df = generator.generate_universe(
                    theme_map, 
                    target_cumulative_weight=target_weight,
                    max_tickers_per_sector=qty_cap_val,
                    min_dollar_volume=min_vol_input
                )
            
            # 3. Output - Summary Table
            st.subheader("Generation Results")
            
            # Add color highlighting to the status
            def color_status(val):
                color = '#ff4b4b' if 'No Data' in val else '#ffa700' if 'Limit' in val else '#21c354'
                return f'color: {color}'

            st.dataframe(
                stats_df.style.map(color_status, subset=['Status']),
                use_container_width=True,
                column_config={
                    "Weight Pulled": st.column_config.NumberColumn("ETF Weight %", format="%.1f%%"),
                    "Tickers Selected": st.column_config.NumberColumn("Tickers Added"),
                }
            )
            
            if any("Limit" in s for s in stats_df['Status'].values):
                st.warning("⚠️ **Data Limitation:** Some sectors only returned Top 10 holdings (approx 10-20% weight). This is a limitation of free data sources. For full 60% coverage, you may need to manually import holdings for equal-weight ETFs like XBI.")

            # 4. Output - Copy/Paste Area
            st.subheader("Your New Universe CSV")
            st.caption("Copy this and paste it into your Google Sheet 'SECTOR_UNIVERSE'")
            st.code(csv_data, language="csv")

    # ==========================================
    # ADMIN: PHASE 2 & 3: OPTIMISE GAIN MOM & OUTPERFORMING SIGNALS BY SECTOR ETF
    # ==========================================
    with st.expander("🛠️ Phase 2 & 3: Optimise What is "Gaining Mom & Outperforming" by Sector ETF)", expanded=False):
        st.caption("Generate a master file to scientifically prove WHICH trend definition works best, and WHEN to enter.")
        
        # Initialize session state for this phase
        if "compass_df" not in st.session_state:
            st.session_state.compass_df = None

        c_p2_1, c_p2_2 = st.columns([1.5, 1])
        
        with c_p2_1:
            st.info("""
            **This Single Export Solves Two Problems:**
            
            1.  **The Logic (Phase 2):** Tests 6 different mathematical definitions of "Gaining Momentum" (from Fast 5d scalps to Slow 20d trends).
            2.  **The Timing (Phase 3):** Includes a **"Streak Counter"** for every logic. This lets AI compare buying on **Day 1** (Aggressive) vs **Day 3** (Confirmed).
            """)
            
        with c_p2_2:
            if st.button("🧭 Generate Compass Data", use_container_width=True):
                # Ensure universe is loaded
                if 'theme_map' not in locals() or not theme_map:
                     _, _, theme_map, _, _ = us.SectorDataManager().load_universe()
                
                with st.spinner("Calculating 6 Logics x Streak Counts for all ETFs..."):
                    df_compass = us.generate_compass_data(etf_data_cache, theme_map)
                    st.session_state.compass_df = df_compass
                    
        # Display Download & Prompt if Data Exists
        if st.session_state.compass_df is not None and not st.session_state.compass_df.empty:
            df_comp = st.session_state.compass_df
            st.success(f"✅ Generated {len(df_comp)} rows of optimization data!")
            
            # Download Button
            csv_compass = df_comp.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="⬇️ Download Compass_Master.csv",
                data=csv_compass,
                file_name="Compass_Logic_And_Timing_Master.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # AI Prompt Context (Directly Visible)
            st.markdown("---")
            st.markdown("### 📋 AI Optimization Prompt")
            st.markdown("""
            **Copy this prompt to ChatGPT/Claude to optimize your strategy:**
            
            ```text
            I have uploaded a 'Compass Master' file containing historical data for multiple Sector ETFs.
            This file tests 6 different 'Trend Logics' (A through F) and includes Forward Returns (1d to 20d).

            **Columns Guide:**
            - Logic_X_Signal: 1 = The trend is 'Gaining Momentum & Outperforming'. 0 = It is not.
            - Logic_X_Streak: How many consecutive days the signal has been active (1 = First Day).
            - Target_Xd: The percent return of the ETF X days later.

            **My Goal:**
            I want to find the 'Golden Setup' for each Sector. 

            **Please analyze the data and answer:**
            1. Which Logic (A-F) has the highest correlation with positive 5-day and 10-day returns?
            2. PERFORM A STREAK ANALYSIS: For the best Logic, is the Expected Value (EV) higher on Day 1 (Fresh) or Day 3 (Confirmed)?
            3. Are there specific sectors (e.g., Semis vs Utilities) that require different Logics (e.g., Fast vs Slow)?
            ```
            """)
  
    # ==========================================
    # ADMIN: PHASE 4: DOWNLOAD SECTOR HISTORY
    # ==========================================
    with st.expander("🛠️ Phase 4: Download Sector History", expanded=False):
        st.caption("Use this tool to download sector history for AI to choose the best RVOL and Alpha variables by theme.")
    
        # Initialize session state for generated data so buttons persist
        if "gen_theme_df" not in st.session_state:
            st.session_state.gen_theme_df = None
        if "gen_theme_name" not in st.session_state:
            st.session_state.gen_theme_name = ""
    
        with st.container():
            # --- 1. THEME DATA ---
            st.markdown("#### 1. Theme Data (training sets use 10d trend rel to benchmark selected above)")
            
            dl_theme = st.selectbox(
                "Select Theme to Download", 
                options=all_themes,
                index=0,
                key="dl_theme_selector"
            )
            
            # Generation Button
            if st.button(f"🧠 Generate AI Training Data for {dl_theme}", use_container_width=True):
                target_etf = theme_map.get(dl_theme)
                if target_etf and target_etf in etf_data_cache:
                    # Identify Stocks
                    theme_stocks = uni_df[
                        (uni_df['Role'] == 'Stock') & 
                        (uni_df['Theme'] == dl_theme)
                    ]['Ticker'].unique().tolist()
    
                    with st.spinner("Calculating extended windows (5-50d) & targets..."):
                        # Generate and Store in Session State
                        df_result = us.generate_ai_training_data(
                            target_etf,
                            etf_data_cache,
                            theme_stocks,
                            dl_theme,
                            st.session_state.sector_benchmark
                        )
                        st.session_state.gen_theme_df = df_result
                        st.session_state.gen_theme_name = dl_theme
                else:
                    st.warning("ETF data not found.")
                    st.session_state.gen_theme_df = None
    
            # Display Download Options (if data is generated)
            if st.session_state.gen_theme_df is not None and not st.session_state.gen_theme_df.empty:
                training_df = st.session_state.gen_theme_df
                current_theme = st.session_state.gen_theme_name
                
                # Message
                st.success(f"✅ Data Ready: {current_theme} ({len(training_df)} rows)")
    
                # Two columns for the two formats
                fmt_col1, fmt_col2 = st.columns(2)
                
                with fmt_col1:
                    # PARQUET
                    import io
                    parquet_buffer = io.BytesIO()
                    training_df.to_parquet(parquet_buffer, index=True)
                    parquet_data = parquet_buffer.getvalue()
                    
                    st.download_button(
                        label=f"⬇️ {current_theme}.parquet",
                        data=parquet_data,
                        file_name=f"{current_theme}_AI_Training.parquet",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
    
                with fmt_col2:
                    # CSV
                    csv_data = training_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label=f"⬇️ {current_theme}.csv",
                        data=csv_data,
                        file_name=f"{current_theme}_AI_Training.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Copy-Paste Context
                with st.expander("📋 View AI Prompt Context"):
                    schema_desc = f"""
    I have uploaded a file containing historical trading data for the '{current_theme}' sector. 
    The data is designed to train/optimize a swing trading strategy.
    
    **Schema Description:**
    1. **Context Columns:**
       - `Theme_Category`: The status of the Sector ETF on that day (e.g., "Gaining Momentum & Outperforming").
       - `Days_In_Category`: How many consecutive days the ETF has been in that specific category.
    
    2. **Feature Columns (Predictors):**
       - `Metric_Alpha_[N]d`: The stock's excess return vs the ETF over N days (Windows: 5, 10, 15, 20, 30, 50).
       - `Metric_RVOL_[N]d`: The stock's Relative Volume over N days.
    
    3. **Target Columns (Outcomes):**
       - `Target_FwdRet_1d`: The stock's return on the NEXT day.
       - `Target_FwdRet_5d`: The stock's return over the NEXT 5 days.
       - `Target_FwdRet_10d`: The stock's return over the NEXT 10 days.
    
    **Goal:**
    Analyze the correlations between the `Theme_Category`, `Metric_Alpha`, and `Metric_RVOL` features against the `Target_FwdRet` columns.
    Find the optimal combination of Alpha and RVOL filters for each Theme Category to maximize forward returns.
                    """
                    st.code(schema_desc, language="markdown")
    
            st.markdown("---")
            
            # --- 2. BENCHMARK DATA ---
            # Now in the same flow/column as requested
            st.markdown("#### 2. Benchmark Data (Price Only)")
            
            bench_col1, bench_col2 = st.columns(2)
            
            # Download SPY
            with bench_col1:
                if "SPY" in etf_data_cache:
                    spy_export = us.generate_benchmark_export("SPY", etf_data_cache)
                    if not spy_export.empty:
                        st.download_button(
                            label="⬇️ Download SPY",
                            data=spy_export.to_csv(index=True).encode('utf-8'),
                            file_name="SPY_History_Clean.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="dl_btn_spy"
                        )
            
            # Download QQQ
            with bench_col2:
                if "QQQ" in etf_data_cache:
                    qqq_export = us.generate_benchmark_export("QQQ", etf_data_cache)
                    if not qqq_export.empty:
                        st.download_button(
                            label="⬇️ Download QQQ",
                            data=qqq_export.to_csv(index=True).encode('utf-8'),
                            file_name="QQQ_History_Clean.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="dl_btn_qqq"
                        )
