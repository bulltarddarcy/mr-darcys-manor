# ... (keeping existing imports and authentication) ...

def run_pivot_tables_app(df):
    """Exposure analysis with split rows and robust RR pairing"""
    st.title("🎯 Pivot Tables")
    yesterday = date.today() - timedelta(days=1)
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
    with c1: td_start = st.date_input("Trade Start Date", value=yesterday, key="pv_start")
    with c2: td_end = st.date_input("Trade End Date", value=yesterday, key="pv_end")
    with c3: ticker_filter = st.text_input("Ticker (blank=all)", value="", key="pv_ticker").strip().upper()
    
    # Changed default index to 0 (0M) so tables aren't empty by default
    with c4: min_notional = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}[st.selectbox("Min Dollars", options=["0M", "5M", "10M", "50M", "100M"], index=0, key="pv_notional")]
    
    # Changed default index to 0 (0B) so tables aren't empty by default
    with c5: min_mkt_cap = {"0B": 0, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}[st.selectbox("Mkt Cap Min", options=["0B", "100B", "200B", "500B", "1T"], index=0, key="pv_mkt_cap")]
    
    with c6: ema_filter = st.selectbox("Over 21 Day EMA", options=["All", "Yes"], index=0, key="pv_ema_filter")
    st.markdown('</div>', unsafe_allow_html=True)

    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    
    if d_range.empty:
        st.info("No data found for the selected date range.")
        return

    order_type_col = "Order Type" if "Order Type" in d_range.columns else "Order type"
    cb_pool, ps_pool = d_range[d_range[order_type_col] == "Calls Bought"].copy(), d_range[d_range[order_type_col] == "Puts Sold"].copy()
    
    # --- RR MATCHING LOGIC ---
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    cb_pool['occ'], ps_pool['occ'] = cb_pool.groupby(keys).cumcount(), ps_pool.groupby(keys).cumcount()
    
    # Identify matches
    rr_matches = pd.merge(cb_pool, ps_pool, on=keys + ['occ'], suffixes=('_c', '_p'))
    
    # Extract RR Rows
    rr_rows = []
    for idx, row in rr_matches.iterrows():
        rr_rows.append({'Symbol': row['Symbol'], 'Trade Date': row['Trade Date'], 'Expiry_DT': row['Expiry_DT'], 'Contracts': row['Contracts'], 'Dollars': row['Dollars_c'], 'Strike': clean_strike_fmt(row['Strike_c']), 'Pair_ID': idx, 'Pair_Side': 0})
        rr_rows.append({'Symbol': row['Symbol'], 'Trade Date': row['Trade Date'], 'Expiry_DT': row['Expiry_DT'], 'Contracts': row['Contracts'], 'Dollars': row['Dollars_p'], 'Strike': clean_strike_fmt(row['Strike_p']), 'Pair_ID': idx, 'Pair_Side': 1})
    df_rr = pd.DataFrame(rr_rows)

    # REMOVE RR components from the individual pools so they only show in the RR table
    if not rr_matches.empty:
        # We use the 'occ' and keys to identify exactly which rows were paired
        match_keys = keys + ['occ']
        # Helper to filter out rows present in the match list
        def filter_out_matches(pool, matches, suffix):
            # Create a unique key for merging
            temp_matches = matches[match_keys].copy()
            # Mark these as 'to_remove'
            temp_matches['_remove'] = True
            merged = pool.merge(temp_matches, on=match_keys, how='left')
            return merged[merged['_remove'].isna()].drop(columns=['_remove'])

        cb_pool = filter_out_matches(cb_pool, rr_matches, '_c')
        ps_pool = filter_out_matches(ps_pool, rr_matches, '_p')

    def apply_f(data, is_rr=False):
        if data.empty: return data
        f = data.copy()
        if ticker_filter: 
            f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
        
        # Apply quantitative filters
        f = f[f["Dollars"] >= min_notional]
        
        if not f.empty and min_mkt_cap > 0: 
            # Optimization: Only check unique symbols
            valid_symbols = [s for s in f["Symbol"].unique() if get_market_cap(s) >= min_mkt_cap]
            f = f[f["Symbol"].isin(valid_symbols)]
            
        if not f.empty and ema_filter == "Yes": 
            valid_ema_symbols = [s for s in f["Symbol"].unique() if is_above_ema21(s)]
            f = f[f["Symbol"].isin(valid_ema_symbols)]
        return f

    # Apply filters to all pools consistently
    df_cb_f = apply_f(cb_pool)
    df_ps_f = apply_f(ps_pool)
    df_rr_f = apply_f(df_rr, is_rr=True)

    def get_p(data, is_rr=False):
        if data.empty: return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
        
        # Calculate total dollars per symbol for sorting
        sr = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
        
        if is_rr:
            # For RR, we keep the Pair_ID together so the Call and Put of the same RR stay adjacent
            piv = data.merge(sr, on="Symbol").sort_values(by=["Total_Sym_Dollars", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
        else:
            piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().merge(sr, on="Symbol")
            piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
        
        piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
        
        # Logic to only show the Symbol on the first row of its group for cleaner UI
        piv["Symbol_Display"] = piv["Symbol"]
        piv.loc[piv["Symbol"] == piv["Symbol"].shift(1), "Symbol_Display"] = ""
        
        return piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]

    col1, col2, col3 = st.columns(3)
    fmt = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}
    
    with col1:
        st.subheader("Calls Bought")
        tbl = get_p(df_cb_f)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), 
                         use_container_width=True, hide_index=True, 
                         height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
        else:
            st.caption("No individual calls found (or filtered).")

    with col2:
        st.subheader("Puts Sold")
        tbl = get_p(df_ps_f)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), 
                         use_container_width=True, hide_index=True, 
                         height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
        else:
            st.caption("No individual puts found (or filtered).")

    with col3:
        st.subheader("Risk Reversals")
        tbl = get_p(df_rr_f, is_rr=True)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(fmt).map(highlight_expiry, subset=["Expiry_Table"]), 
                         use_container_width=True, hide_index=True, 
                         height=get_table_height(tbl), column_config=COLUMN_CONFIG_PIVOT)
            st.caption("⚠️ RR Table reflects date range only")
        else:
            st.caption("No matched RR pairs found.")

# ... (rest of the app execution) ...
