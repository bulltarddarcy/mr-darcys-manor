import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import utils_sector as us

# ==========================================
# UI HELPERS
# ==========================================
def classify_setup(df):
    """Classifies the setup (J-Hook, Bull Flag, Rocket)"""
    if df is None or df.empty: return None
    last = df.iloc[-1]
    if "RRG_Mom_Short" not in last or "RRG_Mom_Long" not in last: return None

    m5 = last["RRG_Mom_Short"]
    m10 = last.get("RRG_Mom_Med", 0)
    m20 = last["RRG_Mom_Long"]
    ratio_20 = last.get("RRG_Ratio_Long", 100)

    if m20 < 100 and m5 > 100 and m5 > (m20 + 2): return "🪝 J-Hook"
    if ratio_20 > 100 and m5 > 100 and m5 > m10: return "🚩 Bull Flag"
    if m5 > m10 and m10 > m20 and m20 > 100: return "🚀 Rocket"
    return None 

def plot_simple_rrg(dm, target_map, view_key, show_trails):
    fig = go.Figure()
    all_x, all_y = [], []
    
    for theme, ticker in target_map.items():
        df = dm.load_ticker_data(ticker)
        if df is None or df.empty: continue
        col_x, col_y = f"RRG_Ratio_{view_key}", f"RRG_Mom_{view_key}"
        if col_x not in df.columns: continue
        
        data_slice = df.tail(3) if show_trails else df.tail(1)
        if data_slice.empty: continue

        x_vals = data_slice[col_x].tolist()
        y_vals = data_slice[col_y].tolist()
        all_x.extend(x_vals); all_y.extend(y_vals)
        
        last_x, last_y = x_vals[-1], y_vals[-1]
        if last_x > 100 and last_y > 100: color = '#00CC96' 
        elif last_x < 100 and last_y > 100: color = '#636EFA'
        elif last_x > 100 and last_y < 100: color = '#FFA15A'
        else: color = '#EF553B'
        
        n = len(x_vals)
        sizes = [8] * (n - 1) + [15]
        opacities = [0.4] * (n - 1) + [1.0]
        texts = [""] * (n - 1) + [theme]
        custom_data = [theme] * n

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers+text', name=theme, text=texts,
            customdata=custom_data, textposition="top center",
            marker=dict(size=sizes, color=color, opacity=opacities, line=dict(width=1, color='white')),
            line=dict(color=color, width=1 if show_trails else 0, shape='spline', smoothing=1.3),
            hoverinfo='text+name',
            hovertext=[f"{theme}<br>Trend: {x:.1f}<br>Mom: {y:.1f}" for x,y in zip(x_vals, y_vals)]
        ))

    # Scaling
    if all_x and all_y:
        limit_x = max(max([abs(x - 100) for x in all_x]) * 1.1, 2.0)
        limit_y = max(max([abs(y - 100) for y in all_y]) * 1.1, 2.0)
        x_range = [100 - limit_x, 100 + limit_x]
        y_range = [100 - limit_y, 100 + limit_y]
    else:
        x_range, y_range = [98, 102], [98, 102]
        limit_x, limit_y = 2, 2

    fig.add_hline(y=100, line_width=1, line_color="gray", line_dash="dash")
    fig.add_vline(x=100, line_width=1, line_color="gray", line_dash="dash")
    
    lbl_x, lbl_y = limit_x * 0.5, limit_y * 0.5
    
    def add_hud_label(x, y, text, color):
        fig.add_annotation(
            x=x, y=y, text=f"<b>{text}</b>", showarrow=False, 
            font=dict(color=color, size=20)
        )

    add_hud_label(100+lbl_x, 100+lbl_y, "LEADING", "rgba(0, 255, 0, 0.7)")
    add_hud_label(100-lbl_x, 100+lbl_y, "IMPROVING", "rgba(0, 100, 255, 0.7)")
    add_hud_label(100+lbl_x, 100-lbl_y, "WEAKENING", "rgba(255, 165, 0, 0.7)")
    add_hud_label(100-lbl_x, 100-lbl_y, "LAGGING", "rgba(255, 0, 0, 0.7)")

    fig.update_layout(
        xaxis=dict(title="Relative Trend", showgrid=False, range=x_range, constrain='domain'),
        yaxis=dict(title="Relative Momentum", showgrid=False, range=y_range),
        height=750, showlegend=False, template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# ==========================================
# MAIN PAGE FUNCTION
# ==========================================
def run_sector_rotation_app(df_global=None):
    st.title("🔄 Sector Rotation")
    
    # 1. Init Data
    dm = us.SectorDataManager()
    uni_df, tickers, theme_map = dm.load_universe()
    
    if uni_df.empty:
        st.warning("⚠️ SECTOR_UNIVERSE secret is missing or empty.")
        return

    # --- GUIDES (EXPANDERS) ---
    with st.expander("📚 Page Information & User Guide", expanded=False):
        st.markdown("""
        **⏱️ Understanding Timeframes**
        * **5 Days (Short):** The Tactical View. (~1 Week). Highly sensitive. Use this for precise entries (buying dips) or exits.
        * **10 Days (Med):** The Balance. (~2 Weeks). A middle ground that smooths out daily noise but reacts faster than the monthly trend.
        * **20 Days (Long):** The Strategic View. (~1 Month). This is your primary trend. Major moves happen here.

        **🧮 How It Works (The Math)**
        This chart does **not** show price. It shows **Relative Performance** against the S&P 500 (SPY).
        * **X-Axis (Trend):** Are we beating the market?
            * `> 100`: Outperforming the S&P 500.
            * `< 100`: Underperforming the S&P 500.
        * **Y-Axis (Momentum):** How fast is the trend changing?
            * `> 100`: Gaining speed (Acceleration).
            * `< 100`: Losing speed (Deceleration).
        
        *Note: Calculations use Weighted Regression, meaning today's price action is weighted 3x more than data from 20 days ago. This eliminates lag.*

        **📊 How to Read the Quadrants**
        * 🟢 **LEADING (Top Right):** Strong Trend + Accelerating Momentum. The Winners.
        * 🟡 **WEAKENING (Bottom Right):** Strong Trend, but losing steam. Often a place to take profits or wait for a pullback.
        * 🔴 **LAGGING (Bottom Left):** Weak Trend + Decelerating. The Losers. Avoid unless hunting for bottoms.
        * 🔵 **IMPROVING (Top Left):** Weak Trend, but Momentum is waking up. This is where "Turnarounds" happen.

        **🐍 Using Tails (History)**
        * **The Dot:** Represents Today's score.
        * **The Tail:** Shows the path of the last 3 days.
        * **Long Tail:** = High Velocity. The move is strong and decisive.
        * **Short Tail:** = Indecision. The sector is stuck.
        * **J-Hook:** If the tail is in the bottom left but "Hooks" sharply up and right, smart money is rotating in aggressively.
        """)
    
    with st.expander("🔑 View Setup Key", expanded=False):
        st.markdown("""
        **🪝 J-Hook (Turnaround)**
        * **Definition:** Long-term trend is weak, but Short-term is exploding higher.
        * **Action:** Aggressive bottom fishing.

        **🚩 Bull Flag (Dip Buy)**
        * **Definition:** Long-term trend is strong. Short-term rested and is turning back up.
        * **Action:** Safe entry into leader.

        **🚀 Rocket (Thrust)**
        * **Definition:** Perfect alignment (5 > 10 > 20). Accelerating velocity.
        * **Action:** Chase the strength (FOMO).
        """)

    with st.expander("📂 Category Guide", expanded=False):
        st.markdown("""
        **📈 Increasing:** 5-Day Momentum > 20-Day Momentum.
        * **Meaning:** The move is accelerating (Gas Pedal is down).

        **🔻 Decreasing:** 5-Day Momentum < 20-Day Momentum.
        * **Meaning:** The move is slowing down (Braking).

        **⚖️ Neutral:** Mixed signals or consolidating.
        """)

    # 2. Session State for Controls
    if "sector_view" not in st.session_state: st.session_state.sector_view = "5 Days"
    if "sector_trails" not in st.session_state: st.session_state.sector_trails = False
    if "sector_target" not in st.session_state: st.session_state.sector_target = sorted(list(theme_map.keys()))[0] if theme_map else ""

    # Ensure filter list is init for the widget
    all_themes = sorted(list(theme_map.keys()))
    if "sector_theme_filter_widget" not in st.session_state:
        st.session_state.sector_theme_filter_widget = all_themes

    # 3. CONTROL SECTION
    
    # --- Row 1: Timeframe & Global Actions ---
    col_time, col_btn = st.columns([2, 1])
    with col_time:
        st.markdown("##### ⏱️ Timeframe Window")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.session_state.sector_view = st.radio(
                "Window", ["5 Days", "10 Days", "20 Days"], 
                horizontal=True, label_visibility="collapsed", key="timeframe_radio"
            )
        with c2:
            st.markdown('<div style="margin-top: 5px;"></div>', unsafe_allow_html=True)
            st.session_state.sector_trails = st.checkbox("Trails", value=st.session_state.sector_trails)

    with col_btn:
        st.markdown("##### Actions")
        if st.button("🔄 Update Data", use_container_width=True):
            status = st.empty()
            calc = us.SectorAlphaCalculator()
            calc.run_full_update(status)
            st.rerun()

    timeframe_map = {"5 Days": "Short", "10 Days": "Med", "20 Days": "Long"}
    view_key = timeframe_map[st.session_state.sector_view]

    # --- Row 2: Sector Selection (EXPANDABLE) ---
    with st.expander("👁️ Sectors Shown", expanded=True):
        # Buttons for Add/Remove All
        btn_col1, btn_col2, _ = st.columns([1, 1, 6])
        
        with btn_col1:
            if st.button("➕ Add All"):
                st.session_state.sector_theme_filter_widget = all_themes
                st.rerun()
        with btn_col2:
            if st.button("➖ Remove All"):
                st.session_state.sector_theme_filter_widget = []
                st.rerun()
                
        # The Multiselect
        sel_themes = st.multiselect(
            "Select Themes", 
            all_themes, 
            key="sector_theme_filter_widget", 
            label_visibility="collapsed"
        )
    
    filtered_map = {k: v for k, v in theme_map.items() if k in sel_themes}

    # --- Row 3: Momentum Scans (EXPANDABLE) ---
    with st.expander("🚀 Momentum Scans", expanded=True):
        # Calculate Momentum buckets
        inc_mom, neut_mom, dec_mom = [], [], []
        for theme, ticker in theme_map.items():
            df = dm.load_ticker_data(ticker)
            if df is None or df.empty or "RRG_Mom_Short" not in df.columns: continue
            last = df.iloc[-1]
            m5, m10, m20 = last.get("RRG_Mom_Short",0), last.get("RRG_Mom_Med",0), last.get("RRG_Mom_Long",0)
            setup = classify_setup(df)
            icon = setup.split()[0] if setup else ""
            item = {"theme": theme, "shift": m5-m20, "icon": icon}
            
            # Momentum Logic
            if m5 > m10 > m20: inc_mom.append(item)
            elif m5 < m10 < m20: dec_mom.append(item)
            else: neut_mom.append(item)

        # Display Columns
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1: 
            st.success(f"📈 Increasing ({len(inc_mom)})")
            for i in inc_mom: st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")
            
        with m_col2:
            st.warning(f"⚖️ Neutral / Mixed ({len(neut_mom)})")
            for i in neut_mom: st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")
            
        with m_col3:
            st.error(f"🔻 Decreasing ({len(dec_mom)})")
            for i in dec_mom: st.caption(f"{i['theme']} {i['icon']} **({i['shift']:+.1f})**")

    # 4. RRG CHART
    chart_placeholder = st.empty()
    with chart_placeholder:
        fig = plot_simple_rrg(dm, filtered_map, view_key, st.session_state.sector_trails)
        chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")
    
    # Handle Chart Selection
    if chart_event and chart_event.selection and chart_event.selection.points:
        point = chart_event.selection.points[0]
        if "customdata" in point:
            st.session_state.sector_target = point["customdata"]
        elif "text" in point:
            st.session_state.sector_target = point["text"]
    
    # 5. THEME EXPLORER (Stock List)
    st.subheader(f"🔎 Explorer: {st.session_state.sector_target}")
    
    # Search Bar
    search_c1, search_c2 = st.columns([1, 3])
    with search_c1:
        search_t = st.text_input("Find Ticker", placeholder="NVDA...").strip().upper()
        if search_t:
            matches = uni_df[uni_df['Ticker'] == search_t]
            if not matches.empty:
                found = matches['Theme'].unique()
                if len(found) > 0: st.session_state.sector_target = found[0]
    
    with search_c2:
        # Update Target Dropdown
        all_sorted = sorted(list(theme_map.keys()))
        curr_idx = all_sorted.index(st.session_state.sector_target) if st.session_state.sector_target in all_sorted else 0
        new_target = st.selectbox("Select Theme", all_sorted, index=curr_idx, label_visibility="collapsed")
        if new_target != st.session_state.sector_target:
            st.session_state.sector_target = new_target

    # Quadrant Stats for ETF
    etf_ticker = theme_map.get(st.session_state.sector_target)
    if etf_ticker:
        etf_df = dm.load_ticker_data(etf_ticker)
        if etf_df is not None and not etf_df.empty:
            last = etf_df.iloc[-1]
            q_cols = st.columns(3)
            for i, (l, k) in enumerate([("Short (5d)", "Short"), ("Med (10d)", "Med"), ("Long (20d)", "Long")]):
                r = last.get(f"RRG_Ratio_{k}", 100)
                m = last.get(f"RRG_Mom_{k}", 100)
                if r >= 100 and m >= 100: txt, clr, icn = "Leading", "green", "🟢"
                elif r < 100 and m >= 100: txt, clr, icn = "Improving", "blue", "🔵"
                elif r < 100 and m < 100: txt, clr, icn = "Lagging", "red", "🔴"
                else: txt, clr, icn = "Weakening", "orange", "🟡"
                with q_cols[i]:
                    st.markdown(f"**{l}**: :{clr}[{txt} {icn}] ({m-100:+.1f})")

    # Stock Table
    stock_tickers = uni_df[(uni_df['Theme'] == st.session_state.sector_target) & (uni_df['Role'] == 'Stock')]['Ticker'].tolist()
    ranking_data = []
    
    # DEBUG COUNTERS
    stats = {
        "total_in_theme": len(stock_tickers),
        "files_found": 0,
        "passed_vol_filter": 0,
        "missing_files": []
    }
    
    for stock in stock_tickers:
        sdf = dm.load_ticker_data(stock)
        
        if sdf is None or sdf.empty:
            stats["missing_files"].append(stock)
            continue
            
        stats["files_found"] += 1
        
        try:
            avg_vol = sdf['Volume'].tail(20).mean()
            avg_price = sdf['Close'].tail(20).mean()
            dollar_vol = avg_vol * avg_price
            
            if dollar_vol < us.MIN_DOLLAR_VOLUME:
                continue
                
            stats["passed_vol_filter"] += 1
            
            last = sdf.iloc[-1]
            ranking_data.append({
                "Ticker": stock,
                "Price": last['Close'],
                "Alpha 5d": last.get("True_Alpha_Short", 0),
                "RVOL 5d": last.get("RVOL_Short", 0),
                "Alpha 10d": last.get("True_Alpha_Med", 0),
                "Alpha 20d": last.get("True_Alpha_Long", 0),
                "8 EMA": "✅" if last['Close'] > last.get('EMA_8', 0) else "❌"
            })
        except Exception:
            continue

    if ranking_data:
        df_disp = pd.DataFrame(ranking_data).sort_values(by='Alpha 5d', ascending=False)
        
        def style_rows(row):
            styles = [''] * len(row)
            if row['Alpha 5d'] > 0 and row['RVOL 5d'] > 1.2:
                 return ['background-color: #d4edda; color: black;'] * len(row)
            return styles

        st.dataframe(
            df_disp.style.apply(style_rows, axis=1).format({"Price": "$%.2f", "Alpha 5d": "%+.2f%%", "RVOL 5d": "%.1fx", "Alpha 10d": "%+.2f%%", "Alpha 20d": "%+.2f%%"}),
            hide_index=True, use_container_width=True,
            column_config={"Ticker": st.column_config.TextColumn("Ticker"), "8 EMA": st.column_config.TextColumn("8 EMA", width="small")}
        )
    else:
        st.error(f"⚠️ No stocks displayed for {st.session_state.sector_target}.")
        st.caption(f"**Diagnostics:**")
        st.caption(f"• Stocks listed in Universe CSV: {stats['total_in_theme']}")
        st.caption(f"• Data files found on disk: {stats['files_found']}")
        st.caption(f"• Stocks passed Volume Filter (>${us.MIN_DOLLAR_VOLUME/1_000_000:.1f}M): {stats['passed_vol_filter']}")
        
        if len(stats["missing_files"]) > 0:
            st.caption(f"• **Missing Files:** {', '.join(stats['missing_files'][:10])}...")
        
        if stats['files_found'] == 0:
            st.warning("👉 Suggestion: Click 'Update Data' to download price history.")
        elif stats['passed_vol_filter'] == 0:
            st.warning("👉 Suggestion: All stocks were filtered out due to low dollar volume.")