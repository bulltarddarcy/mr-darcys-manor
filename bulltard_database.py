def run_rsi_percentiles_app():
    st.title("🔢 RSI Percentiles")
    
    st.markdown("""
        <style>
        /* CSS for row hovering effect if needed, though background colors will dominate */
        tr:hover { filter: brightness(95%); }
        </style>
    """, unsafe_allow_html=True)
    
    # Load dataset map
    dataset_map = load_dataset_config()
    options = list(dataset_map.keys())
    
    # 1. Data input sources
    default_opt = options[0] if options else None
    data_option = st.pills("Dataset", options=options, selection_mode="single", default=default_opt, label_visibility="collapsed")
    
    # 2. Collapsible explanation thing
    with st.expander("ℹ️ Strategy Logic & Explanations"):
        st.markdown('<div class="footer-header">📊 PERCENTILE LOGIC</div>', unsafe_allow_html=True)
        st.markdown(f"""
        * **Historical Context**: The algorithm analyzes up to **10 years** of daily price history for each ticker to establish its unique RSI behavior.
        * **Percentile Thresholds**: It calculates the **User-Selected Percentiles** (Default: 10th for Oversold, 90th for Overbought) specific to that stock.
        * **Signal Trigger**: 
            * **Bullish (Leaving Oversold)**: Triggered when the RSI crosses **ABOVE** the low percentile threshold (recovering from extreme lows).
            * **Bearish (Leaving Overbought)**: Triggered when the RSI crosses **BELOW** the high percentile threshold (cooling off from extreme highs).
        * **Scan Window**: The system checks the last **10 trading periods** to catch recent signals.
        * **Expected Value (EV)**: The average historical **PRICE** return (30d and 90d) following these crossover events.
        * **Row Coloring**: 
            * 🟢 **Green**: Historically Profitable Trade (Bullish signal saw price rise, Bearish signal saw price fall).
            * 🔴 **Red**: Historically Unprofitable Trade (Bullish signal saw price fall, Bearish signal saw price rise).
            * ⚪ **White**: Insufficient data (N < 5) to determine historical profitability.
        """)

    # 3. Progress bar
    status_text = st.empty()
    progress_bar = st.progress(0)

    # 4. Percentile inputs
    c1, c2 = st.columns(2)
    with c1:
        oversold_pct_in = st.number_input("Oversold Percentile", min_value=1, max_value=49, value=10, step=1)
    with c2:
        overbought_pct_in = st.number_input("Overbought Percentile", min_value=51, max_value=99, value=90, step=1)
    
    # Auto-run analysis when a dataset is selected
    if data_option:
        results = []
        
        try:
            target_url = st.secrets[dataset_map[data_option]]
            csv_buffer = get_confirmed_gdrive_data(target_url)
            
            if csv_buffer and csv_buffer != "HTML_ERROR":
                master = pd.read_csv(csv_buffer)
                t_col = next((c for c in master.columns if c.strip().upper() in ['TICKER', 'SYMBOL']), None)
                grouped = master.groupby(t_col)
                grouped_list = list(grouped)
                total = len(grouped_list)
                
                for i, (ticker, group) in enumerate(grouped_list):
                    status_text.text(f"Scanning {ticker}...")
                    d_d, _ = prepare_data(group.copy())
                    if d_d is not None:
                        sigs = find_rsi_percentile_signals(
                            d_d, ticker, 
                            oversold_pct=oversold_pct_in/100.0, 
                            overbought_pct=overbought_pct_in/100.0
                        )
                        results.extend(sigs)
                    
                    if i % 10 == 0: progress_bar.progress((i + 1) / total)
                progress_bar.progress(100)
            else:
                st.error("Could not load dataset.")

            status_text.empty()
            
            if results:
                res_df = pd.DataFrame(results)
                res_df = res_df.sort_values(by='Date', ascending=False)
                
                st.subheader(f"Found {len(res_df)} Events (Last 10 Periods)")
                
                # Custom HTML table with row shading
                html_rows = ['<table style="width:100%; border-collapse: collapse;"><thead><tr style="border-bottom: 2px solid #eee; text-align: left;"><th>Date</th><th>Ticker</th><th>RSI</th><th>Threshold</th><th>EV 30d</th><th>EV 90d</th></tr></thead><tbody>']
                
                def fmt_ev(ev_dict):
                    if not ev_dict: return "N/A"
                    val = ev_dict['return'] * 100
                    n = ev_dict['n']
                    color = "green" if val > 0 else "red"
                    # Removed font size reduction for N part
                    return f"<span style='color:{color};'>{val:+.1f}%</span> <span style='color:#666;'>(N={n})</span>"

                for r in res_df.itertuples():
                    # Determine row color based on PROFITABILITY of the 30d signal
                    # Bullish Signal (Long) -> Profitable if EV > 0
                    # Bearish Signal (Short) -> Profitable if EV < 0
                    
                    is_bullish = "Bullish" in r.Type
                    ev_data = r.ev30
                    
                    # Default style (White/Neutral)
                    row_style = "border-bottom: 1px solid #f0f0f0;" 
                    
                    if ev_data:
                        ev_ret = ev_data['return']
                        if is_bullish:
                            if ev_ret > 0:
                                row_style = "background-color: #e6f4ea; border-bottom: 1px solid #fff;" # Green (Profitable Long)
                            else:
                                row_style = "background-color: #fce8e6; border-bottom: 1px solid #fff;" # Red (Unprofitable Long)
                        else: # Bearish
                            if ev_ret < 0:
                                row_style = "background-color: #e6f4ea; border-bottom: 1px solid #fff;" # Green (Profitable Short - Price went down)
                            else:
                                row_style = "background-color: #fce8e6; border-bottom: 1px solid #fff;" # Red (Unprofitable Short - Price went up)
                    
                    ev30_str = fmt_ev(r.ev30)
                    ev90_str = fmt_ev(r.ev90)
                    
                    row_html = f'<tr style="{row_style}"><td style="padding: 10px;">{r.Date}</td><td style="padding: 10px; font-weight:bold;">{r.Ticker}</td><td style="padding: 10px;">{r.RSI:.1f}</td><td style="padding: 10px;">{r.Threshold:.1f}</td><td style="padding: 10px;">{ev30_str}</td><td style="padding: 10px;">{ev90_str}</td></tr>'
                    html_rows.append(row_html)
                
                html_rows.append("</tbody></table>")
                st.markdown("".join(html_rows), unsafe_allow_html=True)
                
            else:
                st.info(f"No tickers found crossing their {oversold_pct_in}th or {overbought_pct_in}th percentile thresholds in the last 10 periods.")
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
