"""
Sector Rotation App - SPLIT VERSION
Contains two separate apps:
1. run_theme_momentum_app: User-facing dashboard (Charts, Signals, Analysis)
2. run_admin_backtesting: Admin tools (Universe Gen, Compass, AI Data)
"""

import streamlit as st
import pandas as pd
import utils_sector as us

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
    
    all_themes = sorted(list(theme_map.keys()))
    if not all_themes:
        st.error("No valid themes found. Check data sources.")
        return

    if "sector_target" not in st.session_state or st.session_state.sector_target not in all_themes:
        st.session_state.sector_target = "All"
    
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # --- 4. CONFIGURATION ---
    st.subheader("⚙️ Configuration")
    
    # Row 1: Benchmark
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

    st.markdown("") # Spacer

    # Row 2: Sector Filter (Full Width)
    with st.expander("🔎 Select Sectors to View", expanded=False):
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
    
    # Map friendly name to internal key (e.g. "5 Days" -> "Short")
    timeframe_map = {"5 Days": "Short", "10 Days": "Med", "20 Days": "Long"}
    
    # Ensure default
    if st.session_state.sector_view not in timeframe_map:
        st.session_state.sector_view = "5 Days"
        
    view_key = timeframe_map[st.session_state.sector_view]

    st.divider()

    # --- 5. CHART SECTION (IN EXPANDER) ---
    # expanded=True keeps it open when interacting with inner widgets
    with st.expander("🗺️ Rotation Quadrant Graphic", expanded=True):
        
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            st.markdown("**Timeframe Window (Chart Only)**")
            st.session_state.sector_view = st.radio(
                "Timeframe Window",
                ["5 Days", "10 Days", "20 Days"],
                horizontal=True,
                key="timeframe_radio",
                label_visibility="collapsed"
            )
            # Update view_key immediately if changed
            view_key = timeframe_map[st.session_state.sector_view]
            
        with c_chart2:
            st.markdown("**Visual Options**")
            st.session_state.sector_trails = st.checkbox(
                "Show 3-Day Trails",
                value=st.session_state.sector_trails
            )

        st.markdown("---")
        
        # Calculate Categories for CHART filtering (uses Chart Timeframe)
        categories_chart = us.get_momentum_performance_categories(
            etf_data_cache, 
            filtered_map, 
            force_timeframe=view_key
        )
        
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
        else:
            selected_themes = [t['theme'] for t in categories_chart.get(st.session_state.chart_filter, [])]
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

    # --- 6. DUAL TABLES DISPLAY ---
    st.subheader("📊 Theme Categories (Gain Mom & Outperform)")
    
    # 1. Calculate Standard (Matches Chart)
    cats_standard = categories_chart # Already calculated above using view_key
    
    # 2. Calculate Optimized (Uses Config)
    cats_optimized = us.get_momentum_performance_categories(
        etf_data_cache, 
        filtered_map, 
        force_timeframe=None # None triggers Smart Opt lookup
    )
    
    # The bucket we care about
    target_bucket = 'gaining_mom_outperforming'
    
    # Helper to prepare dataframe
    def prepare_display_df(items):
        if not items: return pd.DataFrame()
        data = []
        for theme_info in items:
            days = theme_info['days_in_category']
            days_display = "🆕 Day 1" if days == 1 else "⭐ Day 2" if days == 2 else f"Day {days}"
            data.append({
                "Sector": theme_info['theme'],
                "Days": days_display,
                "5d": theme_info['quadrant_5d'],
                "Reason": theme_info['reason']
            })
        df = pd.DataFrame(data)
        # Sort by Days (Newest first)
        df['_days_sort'] = df['Days'].str.extract(r'(\d+)').astype(float).fillna(0)
        df = df.sort_values('_days_sort').drop('_days_sort', axis=1)
        return df

    # Prepare Data
    df_std = prepare_display_df(cats_standard.get(target_bucket, []))
    df_opt = prepare_display_df(cats_optimized.get(target_bucket, []))
    
    # Calculate Height to remove scrollbar (approx 35px per row + 38px header)
    def get_height(df):
        rows = len(df)
        if rows == 0: return 38
        return (rows + 1) * 35 + 3
    
    col_std, col_opt = st.columns(2)
    
    with col_std:
        st.markdown(f"**Standard ({st.session_state.sector_view})**")
        if not df_std.empty:
            st.dataframe(
                df_std, 
                hide_index=True, 
                use_container_width=True, 
                height=get_height(df_std),
                column_config={
                    "Sector": st.column_config.TextColumn("Sector", width="small"),
                    "Reason": st.column_config.TextColumn("Logic", width="medium"),
                }
            )
        else:
            st.info("No sectors match standard criteria.")

    with col_opt:
        st.markdown("**✨ Smart Optimized**")
        if not df_opt.empty:
            st.dataframe(
                df_opt, 
                hide_index=True, 
                use_container_width=True, 
                height=get_height(df_opt),
                column_config={
                    "Sector": st.column_config.TextColumn("Sector", width="small"),
                    "Reason": st.column_config.TextColumn("Logic", width="medium"),
                }
            )
        else:
            st.warning("No sectors match optimized criteria (or config missing).")

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
    # For Stock Analysis, which category map should we use?
    # Usually safer to use the Standard one to avoid confusion, or offer a choice.
    # Defaulting to Standard (Chart Timeframe) for consistency with the filter labels.
    theme_cat_map = {}
    for cat_list in cats_standard.values():
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


# ==========================================
# APP 2: THE ADMIN & BACKTESTING SUITE
# ==========================================
def run_admin_backtesting():
    """
    The 'Back End' for optimization.
    Focus: Generating universe, testing logic (Compass), creating AI datasets.
    """
    st.title("🛠️ Sector Admin & Backtesting")
    
    # --- SHARED DATA FETCH ---
    if "sector_benchmark" not in st.session_state:
        st.session_state.sector_benchmark = "SPY"
        
    st.caption(f"Loaded Universe Benchmark: **{st.session_state.sector_benchmark}**")
    
    new_bench = st.radio("Benchmark Source:", ["SPY", "QQQ"], horizontal=True, key="admin_bench_radio")
    if new_bench != st.session_state.sector_benchmark:
        st.session_state.sector_benchmark = new_bench
        st.cache_data.clear()
        st.rerun()

    with st.spinner(f"Loading Sector Data for Admin Tools..."):
        etf_data_cache, _, theme_map, uni_df, _ = \
            us.fetch_and_process_universe(st.session_state.sector_benchmark)
            
    all_themes = sorted(list(theme_map.keys()))

    # ==========================================
    # PHASE 1: GENERATE NEW UNIVERSE
    # ==========================================
    st.header("Phase 1: Generate New Universe")
    st.caption("Use this tool to mathematically generate a universe based on ETF holdings.")
    
    c_gen1, c_gen2, c_gen3, c_gen4 = st.columns([1, 1, 1, 1])
    
    with c_gen1:
        target_weight_input = st.number_input(
            "Target Cumulative Weight %", 
            min_value=5.0, max_value=100.0, value=60.0, step=5.0, format="%.1f",
            help="We will keep pulling tickers until they account for this % of the ETF's weight."
        )
    with c_gen2:
        qty_cap = st.number_input("Max Tickers per Sector", min_value=0, value=0, 
            help="Hard cap on count. Leave 0 for no limit.")
        qty_cap_val = None if qty_cap == 0 else qty_cap
    with c_gen3:
        min_vol_input = st.number_input("Min $ Volume (Daily)", value=10_000_000, step=1_000_000, format="%d",
            help="Filter out illiquid stocks.")
    
    with c_gen4:
        st.markdown('<div style="margin-top: 28px;"></div>', unsafe_allow_html=True)
        run_gen = st.button("🚀 Generate", use_container_width=True)

    if run_gen:
        generator = us.UniverseGenerator()
        with st.spinner("Fetching holdings and validating volume... (This may take 30s)"):
            csv_data, stats_df = generator.generate_universe(
                theme_map, 
                target_cumulative_weight=(target_weight_input / 100.0),
                max_tickers_per_sector=qty_cap_val,
                min_dollar_volume=min_vol_input
            )
        
        st.subheader("Generation Results")
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
            st.warning("⚠️ **Data Limitation:** Some sectors only returned Top 10 holdings.")

        with st.expander("📄 View/Copy Your New Universe CSV", expanded=False):
            st.code(csv_data, language="csv")

    # ==========================================
    # PHASE 2 & 3: THE COMPASS
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Phase 2 & 3: The Compass (Optimize Logic & Timing)")
    st.caption("Generate a master file to scientifically prove WHICH trend definition works best, and WHEN to enter.")
    
    if "compass_df" not in st.session_state:
        st.session_state.compass_df = None

    st.info("""
    **This export creates a 'Matrix of Truth' testing 30 different strategy combinations per day.**
    
    **The Buckets (Variables Tested):**
    1.  **6 Logic Definitions:**
        * `Logic A`: 10d Trend / 3d Smooth (Standard)
        * `Logic B`: 10d Trend / 5d Smooth (Smoother)
        * `Logic C`: 20d Trend / 3d Smooth (Slow)
        * `Logic D`: 20d Trend / 5d Smooth (Slowest)
        * `Logic E`: 5d Trend / 2d Smooth (Fast Scalp)
        * `Logic F`: 5d Trend / 3d Smooth (Fast Smooth)
    2.  **Entry Timing:** `Streak Counter` (Test EV of buying Day 1 vs Day 2 vs Day 3).
    3.  **Forward Returns:** `1d`, `3d`, `5d`, `10d`, `20d` (Did the signal work?).
    """)
    
    if st.button("🧭 Generate Compass Data", use_container_width=True):
        with st.spinner("Calculating 6 Logics x Streak Counts for all ETFs..."):
            df_compass = us.generate_compass_data(etf_data_cache, theme_map)
            st.session_state.compass_df = df_compass
            
    if st.session_state.compass_df is not None and not st.session_state.compass_df.empty:
        df_comp = st.session_state.compass_df
        st.success(f"✅ Compass Generated ({len(df_comp)} rows)")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            if st.button("✨ Optimize & Save Settings", type="primary", use_container_width=True):
                with st.spinner("Finding best logic per ETF and saving to file..."):
                    optimized_dict, msg = us.optimize_compass_settings(df_comp)
                    st.success(f"✅ Success! {len(optimized_dict)} ETFs optimized. The dashboard will now use these settings automatically.")
                    
        with col_opt2:
            csv_compass = df_comp.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_compass,
                file_name="Compass_Logic_And_Timing_Master.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        # AI Prompt Context (RESTORED)
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
        3. Please provide a table with columns: Ticker, Window, Smooth. 
           - If Best Logic is A (10d_3s) -> Window="Med", Smooth=3
           - If Best Logic is E (5d_2s) -> Window="Short", Smooth=2
           - etc.
        ```
        """)
            
    # ==========================================
    # PHASE 4: AI TRAINING DATA
    # ==========================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Phase 4: Download Sector History")
    st.caption("Download deep history for specific themes to train AI on Alpha/RVOL metrics.")

    if "gen_theme_df" not in st.session_state:
        st.session_state.gen_theme_df = None
    if "gen_theme_name" not in st.session_state:
        st.session_state.gen_theme_name = ""

    st.markdown("#### Theme Data (Training Sets)")
    dl_theme = st.selectbox("Select Theme", options=all_themes, index=0, key="dl_theme_selector")
    
    if st.button(f"🧠 Generate AI Training Data for {dl_theme}", use_container_width=True):
        target_etf = theme_map.get(dl_theme)
        if target_etf and target_etf in etf_data_cache:
            theme_stocks = uni_df[(uni_df['Role']=='Stock') & (uni_df['Theme']==dl_theme)]['Ticker'].unique().tolist()
            with st.spinner("Calculating extended windows..."):
                df_result = us.generate_ai_training_data(
                    target_etf, etf_data_cache, theme_stocks, dl_theme, st.session_state.sector_benchmark
                )
                st.session_state.gen_theme_df = df_result
                st.session_state.gen_theme_name = dl_theme
        else:
            st.warning("ETF data not found.")
            st.session_state.gen_theme_df = None

    if st.session_state.gen_theme_df is not None and not st.session_state.gen_theme_df.empty:
        training_df = st.session_state.gen_theme_df
        current_theme = st.session_state.gen_theme_name
        st.success(f"✅ Data Ready: {current_theme} ({len(training_df)} rows)")

        fmt_col1, fmt_col2 = st.columns(2)
        with fmt_col1:
            import io
            parquet_buffer = io.BytesIO()
            training_df.to_parquet(parquet_buffer, index=True)
            st.download_button(
                label=f"⬇️ {current_theme}.parquet",
                data=parquet_buffer.getvalue(),
                file_name=f"{current_theme}_AI_Training.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )
        with fmt_col2:
            st.download_button(
                label=f"⬇️ {current_theme}.csv",
                data=training_df.to_csv(index=True).encode('utf-8'),
                file_name=f"{current_theme}_AI_Training.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("### 📋 AI Prompt Context")
        st.markdown(f"""
        **Copy this prompt to ChatGPT/Claude:**
        
        ```text
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
        ```
        """)
