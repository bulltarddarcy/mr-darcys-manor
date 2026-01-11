# main_prices.py
#This file contains the UI logic for the Price Divergences, RSI Scanner, Seasonality, and EMA Distance apps.

    st.title("📏 EMA Distance Analysis")

    col_in1, col_in2, _ = st.columns([1, 1, 2])
    with col_in1: ticker = st.text_input("Ticker", value="QQQ").upper().strip()
    with col_in2: years_back = st.number_input("Years to Analyze", min_value=1, max_value=20, value=10, step=1)
    
    if not ticker:
        st.warning("Please enter a ticker.")
        return

    def fmt_pct(val):
        if pd.isna(val): return ""
        if val < 0: return f"({abs(val):.1f}%)"
        return f"{val:.1f}%"

    with st.spinner(f"Crunching data for {ticker}..."):
        try:
            t_obj = yf.Ticker(ticker)
            df = t_obj.history(period=f"{years_back}y")
            if df is None or df.empty:
                st.error(f"Could not fetch data for {ticker}.")
                return
            df = df.reset_index()
            df.columns = [c.upper() for c in df.columns]
            date_col = next((c for c in df.columns if 'DATE' in c), "DATE")
            close_col = 'CLOSE' if 'CLOSE' in df.columns else 'Close'
            low_col = 'LOW' if 'LOW' in df.columns else 'Low'
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return

    df['EMA_8'] = df[close_col].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df[close_col].ewm(span=21, adjust=False).mean()
    df['SMA_50'] = df[close_col].rolling(window=50).mean()
    df['SMA_100'] = df[close_col].rolling(window=100).mean()
    df['SMA_200'] = df[close_col].rolling(window=200).mean()
    
    df['Dist_8'] = ((df[close_col] - df['EMA_8']) / df['EMA_8']) * 100
    df['Dist_21'] = ((df[close_col] - df['EMA_21']) / df['EMA_21']) * 100
    df['Dist_50'] = ((df[close_col] - df['SMA_50']) / df['SMA_50']) * 100
    df['Dist_100'] = ((df[close_col] - df['SMA_100']) / df['SMA_100']) * 100
    df['Dist_200'] = ((df[close_col] - df['SMA_200']) / df['SMA_200']) * 100
    df_clean = df.dropna(subset=['EMA_8', 'EMA_21', 'SMA_50', 'SMA_100', 'SMA_200']).copy()
    current_dist_50 = df_clean['Dist_50'].iloc[-1]

    st.subheader(f"{ticker} vs Moving Avgs & Percentiles")
    with st.expander("ℹ️ Table User Guide"):
        st.markdown(f"""
            **1. Key Metrics Tracked.**
            The app calculates the percentage distance (the "Gap") between the current price and five different moving averages.
            **3. Visual Highlighting System.**
            🟢 **Buy Zone (Green):** Triggered if the Gap is ≤ p50 (Median) AND price is > 8-EMA.
            🟡 **Warning Zone (Yellow):** Triggered if the gap is between p50 and p90.
            🔴 **Sell/Trim Zone (Red):** Triggered if the gap is $\ge$ p90.
                    """)

    stats_data = []
    thresholds = {} 
    current_price = df_clean[close_col].iloc[-1]
    current_ema8 = df_clean['EMA_8'].iloc[-1]
    
    metrics = [
        ("Close vs 8-EMA", df_clean['EMA_8'], df_clean['Dist_8']),
        ("Close vs 21-EMA", df_clean['EMA_21'], df_clean['Dist_21']),
        ("Close vs 50-SMA", df_clean['SMA_50'], df_clean['Dist_50']),
        ("Close vs 100-SMA", df_clean['SMA_100'], df_clean['Dist_100']),
        ("Close vs 200-SMA", df_clean['SMA_200'], df_clean['Dist_200']),
    ]
    
    for label, ma_series, dist_series in metrics:
        p_vals = np.percentile(dist_series, [50, 70, 80, 90, 95])
        thresholds[dist_series.name] = { 'p80': p_vals[2], 'p90': p_vals[3] }
        stats_data.append({
            "Metric": label, "Price": current_price, "MA Level": ma_series.iloc[-1],
            "Gap": dist_series.iloc[-1], "Avg": dist_series.mean(),
            "p50": p_vals[0], "p70": p_vals[1], "p80": p_vals[2], "p90": p_vals[3], "p95": p_vals[4]
        })

    df_stats = pd.DataFrame(stats_data)

    def color_combined(row):
        styles = [''] * len(row)
        gap, p50, p90 = row['Gap'], row['p50'], row['p90']
        idx_gap = df_stats.columns.get_loc("Gap")
        if gap >= p90: styles[idx_gap] = 'background-color: #fce8e6; color: #c5221f; font-weight: bold;'
        elif gap <= p50 and (current_price > current_ema8): styles[idx_gap] = 'background-color: #e6f4ea; color: #1e7e34; font-weight: bold;'
        elif gap > p50 and gap < p90: styles[idx_gap] = 'background-color: #fff8e1; color: #d68f00;'
        return styles

    st.dataframe(
        df_stats.style.apply(color_combined, axis=1).format(fmt_pct, subset=["Gap", "Avg", "p50", "p70", "p80", "p90", "p95"]),
        use_container_width=True, hide_index=True,
        column_config={"Price": st.column_config.NumberColumn("Price", format="$%.2f"), "MA Level": st.column_config.NumberColumn("MA Level", format="$%.2f")}
    )
    
    st.subheader("Combo Over-Extension Signals")
    t8_90 = thresholds['Dist_8']['p90']
    t21_80 = thresholds['Dist_21']['p80']
    t50_80 = thresholds['Dist_50']['p80']
    
    m_d = (df_clean['Dist_8'] >= t8_90) & (df_clean['Dist_21'] >= t21_80)
    m_fs = (df_clean['Dist_8'] >= t8_90) & (df_clean['Dist_50'] >= t50_80)
    m_t = (df_clean['Dist_8'] >= t8_90) & (df_clean['Dist_21'] >= t21_80) & (df_clean['Dist_50'] >= t50_80)
    
    # Using UP helper for backtesting
    res_d = up.run_ema_backtest(m_d, df_clean[close_col], df_clean[low_col])
    res_fs = up.run_ema_backtest(m_fs, df_clean[close_col], df_clean[low_col])
    res_t = up.run_ema_backtest(m_t, df_clean[close_col], df_clean[low_col])

    d_active = "✅" if bool(m_d.iloc[-1]) else "❌"
    fs_active = "✅" if bool(m_fs.iloc[-1]) else "❌"
    t_active = "✅" if bool(m_t.iloc[-1]) else "❌"

    combo_rows = [
        {"Combo Rule": "Double EMA", "Triggers": "(8-EMA Gap ≥ p90), (21-EMA Gap ≥ p80)",
         "Occurrences": res_d[0], "Hit Rate (>=8% Draw Down)": res_d[1], "Median Days to Draw Down": f"{int(res_d[2])} days", "Active Today?": d_active, "raw_status": bool(m_d.iloc[-1])},
        {"Combo Rule": "Fast vs Swing", "Triggers": "(8-EMA gap ≥ p90), (50-SMA gap ≥ p80)",
         "Occurrences": res_fs[0], "Hit Rate (>=8% Draw Down)": res_fs[1], "Median Days to Draw Down": f"{int(res_fs[2])} days", "Active Today?": fs_active, "raw_status": bool(m_fs.iloc[-1])},
        {"Combo Rule": "Triple Stack", "Triggers": "(8-EMA gap ≥ p90), (50-SMA gap ≥ p80), (21-EMA gap ≥ p80)",
         "Occurrences": res_t[0], "Hit Rate (>=8% Draw Down)": res_t[1], "Median Days to Draw Down": f"{int(res_t[2])} days", "Active Today?": t_active, "raw_status": bool(m_t.iloc[-1])}
    ]
    
    df_combo = pd.DataFrame(combo_rows)

    def style_combo(row):
        return ['font-weight: bold; color: #c5221f;' if row['raw_status'] else ''] * len(row)

    st.dataframe(
        df_combo.style.apply(style_combo, axis=1).format({"Hit Rate (>=8% Draw Down)": "{:.1f}%"}),
        use_container_width=True, hide_index=True,
        column_config={"Triggers": st.column_config.TextColumn("Trigger Conditions", width="large"), "Active Today?": st.column_config.TextColumn("Active Today?", width="small")},
        column_order=["Combo Rule", "Triggers", "Occurrences", "Hit Rate (>=8% Draw Down)", "Median Days to Draw Down", "Active Today?"]
    )

    st.subheader("Visualizing the % Distance from 50 SMA")
    chart_data = pd.DataFrame({'Date': pd.to_datetime(df_clean[date_col]), 'Distance (%)': df_clean['Dist_50']})
    chart_data = chart_data[chart_data['Date'] >= (chart_data['Date'].max() - timedelta(days=3650))]

    bars = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Date:T', title=None), 
        y=alt.Y('Distance (%)', title='% Dist from 50 SMA'),
        color=alt.condition(alt.datum['Distance (%)'] > 0, alt.value("#71d28a"), alt.value("#f29ca0")),
        tooltip=['Date', 'Distance (%)']
    )
    rule = alt.Chart(pd.DataFrame({'y': [current_dist_50]})).mark_rule(color='#333', strokeDash=[5, 5], strokeWidth=2).encode(y='y:Q')
    st.altair_chart((bars + rule).properties(height=300).interactive(), use_container_width=True)