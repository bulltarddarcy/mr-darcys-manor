def run_strike_zones_app(df):
    # (1) Renamed page to 📊 Strike Zones
    st.title("📊 Strike Zones")
    exp_range_default = (date.today() + timedelta(days=365))
    
    st.markdown('<div class="control-box">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: ticker = st.text_input("Ticker", value="AMZN", key="sz_ticker").strip().upper()
    with c2: td_start = st.date_input("Trade Date (start)", value=None, key="sz_start")
    with c3: td_end = st.date_input("Trade Date (end)", value=None, key="sz_end")
    with c4: exp_end = st.date_input("Exp. Range (end)", value=exp_range_default, key="sz_exp")
    
    # Configuration row
    sc1, sc2, sc3, sc4 = st.columns(4, gap="medium")
    with sc1:
        st.markdown("**View Mode**")
        view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], label_visibility="collapsed")
    with sc2:
        st.markdown("**Zone Width**")
        width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], label_visibility="collapsed")
        fixed_size_choice = 10
        if width_mode == "Fixed": 
            fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=10)
    with sc3:
        st.markdown("**Include Order Type**")
        inc_calls_bought = st.checkbox("Calls Bought", value=True)
        inc_puts_sold    = st.checkbox("Puts Sold", value=True)
        inc_puts_bought  = st.checkbox("Puts Bought", value=True)
    with sc4:
        st.markdown("**Other Options**")
        hide_empty      = st.checkbox("Hide Empty Zones", value=True)
        show_table       = st.checkbox("Show Strike Zone Table", value=True)
    
    # (2) Moved horizontal bar to below the chart options/filters
    st.markdown("---")
    st.markdown('</div>', unsafe_allow_html=True)
        
    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if td_start: f = f[f["Trade Date"].dt.date >= td_start]
    if td_end: f = f[f["Trade Date"].dt.date <= td_end]
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    
    allowed_sz_types = []
    if inc_calls_bought: allowed_sz_types.append("Calls Bought")
    if inc_puts_sold: allowed_sz_types.append("Puts Sold")
    if inc_puts_bought: allowed_sz_types.append("Puts Bought")
    edit_pool_raw = f[f[order_type_col].isin(allowed_sz_types)].copy()
    
    if edit_pool_raw.empty:
        st.warning("No trades match current filters.")
        return
        
    edit_pool = edit_pool_raw.sort_values(by="Trade Date", ascending=False).copy()
    edit_pool["Trade Date Display"] = edit_pool["Trade Date"].dt.strftime("%d %b %y")
    edit_pool["Expiry Display"] = edit_pool["Expiry_DT"].dt.strftime("%d %b %y")
    state_key = f"sz_include_{ticker}"
    
    if state_key not in st.session_state: st.session_state[state_key] = [True] * len(edit_pool)
    if len(st.session_state[state_key]) != len(edit_pool): st.session_state[state_key] = [True] * len(edit_pool)
    edit_pool["Included"] = st.session_state[state_key]
    
    cols_to_show = ["Trade Date Display", order_type_col, "Symbol", "Strike", "Expiry Display", "Contracts", "Dollars", "Included"]
    used = edit_pool[edit_pool["Included"] == True].copy()
    
    @st.cache_data(ttl=300)
    def get_stock_indicators(sym: str):
        try:
            ticker_obj = yf.Ticker(sym)
            h = ticker_obj.history(period="60d", interval="1d")
            if len(h) == 0: return None, None, None, None, None
            close = h["Close"]
            spot_val = float(close.iloc[-1])
            ema8  = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
            ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
            sma200_full = ticker_obj.history(period="2y")["Close"]
            sma200 = float(sma200_full.rolling(window=200).mean().iloc[-1]) if len(sma200_full) >= 200 else None
            return spot_val, ema8, ema21, sma200, h
        except: return None, None, None, None, None

    spot, ema8, ema21, sma200, history = get_stock_indicators(ticker)
    if spot is None: spot = 100.0

    def pct_from_spot(x):
        if x is None or np.isnan(x): return "—"
        return f"{(x/spot-1)*100:+.1f}%"
        
    badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
    if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct_from_spot(ema8)})</span>')
    if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct_from_spot(ema21)})</span>')
    if sma200: badges.append(f'<span class="badge">SMA(200): ${sma200:,.2f} ({pct_from_spot(sma200)})</span>')
    st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

    if used.empty: st.info("No trades included.")
    else:
        used["Signed Dollars"] = used.apply(lambda r: (1 if r[order_type_col] in ("Calls Bought","Puts Sold") else -1) * (r["Dollars"] or 0.0), axis=1)
        fmt_neg = lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

        if view_mode == "Price Zones":
            strike_min, strike_max = float(np.nanmin(used["Strike (Actual)"].values)), float(np.nanmax(used["Strike (Actual)"].values))
            if width_mode == "Auto": zone_w = float(next((s for s in [1, 2, 5, 10, 25, 50, 100] if s >= (max(1e-9, strike_max - strike_min) / 12.0)), 100))
            else: zone_w = float(fixed_size_choice)
            n_dn, n_up = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w)), int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
            lower_edge = spot - n_dn * zone_w
            total = max(1, n_dn + n_up)
            used["ZoneIdx"] = used["Strike (Actual)"].apply(lambda x: min(total - 1, max(0, int(math.floor((x - lower_edge) / zone_w)))))
            agg = used.groupby("ZoneIdx").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
            zone_df = pd.DataFrame([(z, lower_edge + z*zone_w, lower_edge + (z+1)*zone_w) for z in range(total)], columns=["ZoneIdx","Zone_Low","Zone_High"])
            zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
            if hide_empty: zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
            
            # (1) Header "Strike Zones" removed from directly above chart
            st.markdown('<div class="zones-panel">', unsafe_allow_html=True)
            for _, r in zs.sort_values("ZoneIdx", ascending=False).iterrows():
                if r["Zone_Low"] + (zone_w/2) > spot:
                    color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, zs["Net_Dollars"].abs().max()))*420))
                    val_str = fmt_neg(r["Net_Dollars"])
                    st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>', unsafe_allow_html=True)
            for _, r in zs.sort_values("ZoneIdx", ascending=False).iterrows():
                if r["Zone_Low"] + (zone_w/2) < spot:
                    color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, zs["Net_Dollars"].abs().max()))*420))
                    val_str = fmt_neg(r["Net_Dollars"])
                    st.markdown(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            e = used.copy()
            e["Bucket"] = pd.cut((pd.to_datetime(e["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days), bins=[0, 7, 30, 90, 180, 10000], labels=["0-7d", "8-30d", "31-90d", "91-180d", ">180d"], include_lowest=True)
            agg = e.groupby("Bucket").agg(Net_Dollars=("Signed Dollars","sum"), Trades=("Signed Dollars","count")).reset_index()
            # (1) Header removed here as well
            for _, r in agg.iterrows():
                color, w = ("zone-bull" if r["Net_Dollars"]>=0 else "zone-bear"), max(6, int((abs(r['Net_Dollars'])/max(1.0, agg["Net_Dollars"].abs().max()))*420))
                val_str = fmt_neg(r["Net_Dollars"])
                st.markdown(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-bar {color}" style="width:{w}px"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div>', unsafe_allow_html=True)

    if show_table:
        st.subheader("Data Table")
        df_for_editor = edit_pool[cols_to_show].copy()
        df_for_editor["Dollars"] = df_for_editor["Dollars"].apply(lambda x: f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}")
        df_for_editor["Contracts"] = df_for_editor["Contracts"].apply(lambda x: f"{x:,.0f}")
        edited_df = st.data_editor(df_for_editor, column_config={"Trade Date Display": "Trade Date", "Expiry Display": "Expiry", "Contracts": st.column_config.TextColumn("Qty", width=80), "Dollars": st.column_config.TextColumn("Dollars", width=110), "Included": st.column_config.CheckboxColumn(default=True)}, use_container_width=True, hide_index=True, key="strike_zones_editor")
        if not edited_df["Included"].equals(df_for_editor["Included"]): 
            st.session_state[state_key] = edited_df["Included"].tolist()
            st.rerun()
