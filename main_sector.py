# ==========================================
# APP 1: THE TRADING DASHBOARD
# ==========================================
def run_theme_momentum_app(df_global=None):
    """
    The 'Front End' for traders. 
    Focus: Visualizing momentum, finding setups, analyzing stocks.
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
    if "use_smart_opt" not in st.session_state:
        st.session_state.use_smart_opt = False
    
    all_themes = sorted(list(theme_map.keys()))
    if not all_themes:
        st.error("No valid themes found. Check data sources.")
        return

    if "sector_target" not in st.session_state or st.session_state.sector_target not in all_themes:
        st.session_state.sector_target = "All"
    
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # --- 4. CONFIGURATION & INPUTS ---
    # Moved header out of expander for visibility
    st.subheader("⚙️ Inputs & Filters")
    
    c_in1, c_in2 = st.columns(2)
    
    with c_in1:
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

    with c_in2:
        st.markdown("**Timeframe Window (Chart)**")
        st.session_state.sector_view = st.radio(
            "Timeframe Window",
            ["5 Days", "10 Days", "20 Days"],
            horizontal=True,
            key="timeframe_radio",
            label_visibility="collapsed"
        )
    
    # Toggle Row
    c_tog1, c_tog2 = st.columns(2)
    with c_tog1:
        st.session_state.use_smart_opt = st.checkbox(
            "✨ Use Smart Optimization",
            value=st.session_state.use_smart_opt,
            help="If checked, tables below use the statistically BEST timeframe per sector. If unchecked, they match the Chart Window."
        )
    with c_tog2:
        st.session_state.sector_trails = st.checkbox(
            "Show 3-Day Trails (Chart)",
            value=st.session_state.sector_trails
        )

    # Sector Filter Expander
    with st.expander("🔎 Filter Sectors Shown", expanded=False):
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("➕ Everything", use_container_width=True):
                st.session_state.sector_theme_filter_widget = all_themes
                st.rerun()
        with btn_col2:
            if st.button("⭐ Big 11", use_container_width=True):
                big_11 = ["Comms", "Cons Discr", "Cons Staples", "Energy", "Financials", "Healthcare", "Industrials", "Materials", "Real Estate", "Technology", "Utilities"]
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

    # --- 6. CATEGORIZATION LOGIC ---
    force_tf = None if st.session_state.use_smart_opt else view_key
    categories = us.get_momentum_performance_categories(
        etf_data_cache, 
        filtered_map, 
        force_timeframe=force_tf
    )
    
    st.divider()

    # --- CHART SECTION (IN EXPANDER) ---
    with st.expander("🗺️ Rotation Quadrant Graphic", expanded=False):
        st.markdown("**Filter Chart by Category:**")
        if st.session_state.use_smart_opt:
            st.caption("ℹ️ *Categories below are based on Optimized Timeframes, not the Chart.*")
        
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
        else:
            selected_themes = [t['theme'] for t in categories.get(st.session_state.chart_filter, [])]
            filtered_map_chart = {k: v for k, v in filtered_map.items() if k in selected_themes}
        
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

    # --- 7. THEME CATEGORIES DISPLAY (ONLY GAIN MOM & OUTPERF) ---
    st.subheader("📊 Theme Categories")
    
    # UPDATED: Only showing the first category
    quadrant_meta = [
        ('gaining_mom_outperforming', '⬈ GAIN MOM & OUTPERF', 'success', '✅ Best Opportunities - Sectors accelerating...')
    ]

    for key, title, style_func_name, caption in quadrant_meta:
        items = categories.get(key, [])
        style_func = getattr(st, style_func_name)
        
        if items:
            style_func(f"**{title}** ({len(items)} sectors)")
            # st.caption(caption) # Optional: comment out if you want less clutter
            
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

    # Run Analysis
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
    
    numeric_columns = ["Price", "Market Cap (B)", "Beta", "Alpha 5d", "Alpha 10d", "Alpha 20d", "RVOL 5d", "RVOL 10d", "RVOL 20d"]
    categorical_columns = ["Theme", "Theme Category", "Div", "8 EMA", "21 EMA", "50 MA", "200 MA"]
    all_filter_columns = numeric_columns + categorical_columns

    def get_safe_options(df, col_name):
        if col_name not in df.columns: return ["-"]
        opts = sorted([str(x) for x in df[col_name].unique() if pd.notna(x) and str(x).strip() != ""])
        return opts if opts else ["-"]

    unique_themes = get_safe_options(df_stocks, 'Theme')
    unique_categories = get_safe_options(df_stocks, 'Theme Category')
    unique_divs = get_safe_options(df_stocks, 'Div')
    
    def safe_index(options, value, default=0):
        try: return options.index(value)
        except ValueError: return default if 0 <= default < len(options) else 0
            
    current_ui_filters = [] 
    
    for i in range(8):
        cols = st.columns(5)
        default = st.session_state.filter_defaults.get(i, {})
        col_opts = [None] + all_filter_columns
        col_idx = safe_index(col_opts, default.get('column'), 0)
        
        column = cols[0].selectbox(f"F{i+1}", col_opts, index=col_idx, key=f"filter_{i}_column", label_visibility="collapsed", placeholder="Column...")
        
        if column:
            is_numeric = column in numeric_columns
            ops = [">=", "<="] if is_numeric else ["="]
            op_idx = safe_index(ops, default.get('operator', '>='), 0)
            operator = cols[1].selectbox("Op", ops, index=op_idx, key=f"filter_{i}_operator", label_visibility="collapsed")
            
            val_type, val, val_col, val_cat = "Number", None, None, None
            
            if is_numeric:
                type_opts = ["Number", "Column"]
                type_idx = safe_index(type_opts, default.get('type', 'Number'), 0)
                val_type = cols[2].radio("T", type_opts, index=type_idx, key=f"filter_{i}_type", horizontal=True, label_visibility="collapsed")
                
                if val_type == "Number":
                    val = cols[3].number_input("V", value=float(default.get('value', 0.0)), step=0.1, format="%.2f", key=f"filter_{i}_value", label_visibility="collapsed")
                else:
                    vc_idx = safe_index(numeric_columns, default.get('value_column'), 0)
                    val_col = cols[3].selectbox("C", numeric_columns, index=vc_idx, key=f"filter_{i}_val_col", label_visibility="collapsed")
            else:
                val_type = "Categorical"
                if column == "Theme": cat_opts = unique_themes
                elif column == "Theme Category": cat_opts = unique_categories
                elif column == "Div": cat_opts = unique_divs
                else: cat_opts = get_safe_options(df_stocks, column)
                cat_idx = safe_index(cat_opts, default.get('value_cat'), 0)
                val_cat = cols[3].selectbox("V", cat_opts, index=cat_idx, key=f"filter_{i}_val_cat", label_visibility="collapsed")

            logic = None
            if i < 7:
                logic_opts = ["AND", "OR"]
                log_idx = safe_index(logic_opts, default.get('logic', 'AND'), 0)
                logic = cols[4].radio("L", logic_opts, index=log_idx, key=f"filter_{i}_logic", horizontal=True, label_visibility="collapsed")
            
            current_ui_filters.append({'column': column, 'operator': operator, 'value_type': val_type, 'value': val, 'value_column': val_col, 'value_categorical': val_cat, 'logic': logic})

    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
    
    if "active_stock_filters" not in st.session_state:
        st.session_state.active_stock_filters = current_ui_filters

    if st.button("Apply Filters", type="primary", use_container_width=True):
        st.session_state.active_stock_filters = current_ui_filters
        st.session_state.opt_show_divergences = False
        st.session_state.opt_show_mkt_caps = False
        st.rerun()

    df_filtered = us.apply_stock_filters(df_stocks, st.session_state.active_stock_filters)
    
    if current_ui_filters != st.session_state.active_stock_filters:
        st.caption("⚠️ *Filters have changed. Click 'Apply Filters' to update the table.*")

    st.markdown(f"**Showing {len(df_filtered)} of {len(df_stocks)} stock-theme combinations**")

    c1, c2, c3, _ = st.columns([2, 2, 2, 6]) 
    with c1:
        st.checkbox("Show Divergences", key="opt_show_divergences", help="Enrich filtered list. Slower.")
    with c2:
        st.checkbox("Show Market Caps", key="opt_show_mkt_caps", help="Enrich filtered list. Slower.")
    with c3:
        st.checkbox("Show Biotech", key="opt_show_biotech", value=False, help="Include Biotech theme.")

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
    lookup_ticker = st.text_input("Enter a ticker to see its sectors:", placeholder="e.g. AAPL", label_visibility="visible").upper().strip()

    if lookup_ticker:
        if not uni_df.empty and 'Ticker' in uni_df.columns:
            matches = uni_df[uni_df['Ticker'] == lookup_ticker]
            if not matches.empty:
                found_themes = sorted(matches['Theme'].unique().tolist())
                st.success(f"✅ **{lookup_ticker}** is mapped to: **{', '.join(found_themes)}**")
            else:
                st.warning(f"❌ Ticker **{lookup_ticker}** not found in the loaded Universe.")
        else:
            st.error("Universe data not loaded.")
    
    if not df_filtered.empty:
        st.caption("Copy tickers:")
        st.code(", ".join(df_filtered['Ticker'].unique().tolist()), language="text")
