# main_options.py
# This file contains the UI logic for the Database, Rankings, Pivot Tables, andStrike Zones apps.

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import utils_options as uo

# --- 1. DATABASE APP ---
def run_database_app(df):
    st.title("📂 Database")
    max_date = uo.get_max_trade_date(df)
    uo.initialize_database_state(max_date)

    def save_state(k, sk): st.session_state[sk] = st.session_state[k]
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: st.text_input("Ticker", value=st.session_state.saved_db_ticker, key="db_t", on_change=save_state, args=("db_t", "saved_db_ticker"))
    with c2: st.date_input("Start", value=st.session_state.saved_db_start, key="db_s", on_change=save_state, args=("db_s", "saved_db_start"))
    with c3: st.date_input("End", value=st.session_state.saved_db_end, key="db_e", on_change=save_state, args=("db_e", "saved_db_end"))
    with c4: st.date_input("Expiry Max", value=st.session_state.saved_db_exp, key="db_x", on_change=save_state, args=("db_x", "saved_db_exp"))
    
    o1, o2, o3, _ = st.columns([1.5, 1.5, 1.5, 5.5])
    with o1: st.checkbox("Calls Bought", value=st.session_state.saved_db_inc_cb, key="db_cb", on_change=save_state, args=("db_cb", "saved_db_inc_cb"))
    with o2: st.checkbox("Puts Sold", value=st.session_state.saved_db_inc_ps, key="db_ps", on_change=save_state, args=("db_ps", "saved_db_inc_ps"))
    with o3: st.checkbox("Puts Bought", value=st.session_state.saved_db_inc_pb, key="db_pb", on_change=save_state, args=("db_pb", "saved_db_inc_pb"))
    
    filtered = uo.filter_database_trades(df, st.session_state.saved_db_ticker.strip().upper(), 
                                         st.session_state.saved_db_start, st.session_state.saved_db_end, 
                                         st.session_state.saved_db_exp, 
                                         st.session_state.saved_db_inc_cb, st.session_state.saved_db_inc_ps, st.session_state.saved_db_inc_pb)
    
    if filtered.empty: st.warning("No data found."); return
        
    st.subheader("Non-Expired Trades"); st.caption("⚠️ Check OI to confirm trades are open")
    styled = uo.get_database_styled_view(filtered)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=35 * min(len(filtered), uo.DB_TABLE_MAX_ROWS) + 38)
    st.markdown("<br><br>", unsafe_allow_html=True)

# --- 2. RANKINGS APP ---
def run_rankings_app(df):
    st.title("🏆 Rankings")
    max_date = uo.get_max_trade_date(df)
    uo.initialize_rankings_state(max_date - timedelta(days=uo.RANK_LOOKBACK_DAYS), max_date)

    def save_state(k, sk): st.session_state[sk] = st.session_state[k]
    
    c1, c2, c3, c4 = st.columns([1, 1, 0.7, 1.3], gap="small")
    with c1: st.date_input("Start", value=st.session_state.saved_rank_start, key="rk_s", on_change=save_state, args=("rk_s", "saved_rank_start"))
    with c2: st.date_input("End", value=st.session_state.saved_rank_end, key="rk_e", on_change=save_state, args=("rk_e", "saved_rank_end"))
    with c3: st.number_input("Limit", value=st.session_state.saved_rank_limit, min_value=1, key="rk_l", on_change=save_state, args=("rk_l", "saved_rank_limit"))
    with c4: 
        opts = list(uo.RANK_MC_THRESHOLDS.keys())
        idx = opts.index("10B") if "10B" in opts else 0
        mc_sel = st.selectbox("Min Cap", opts, index=idx, key="rk_mc", on_change=save_state, args=("rk_mc", "saved_rank_mc"))
        ema = st.checkbox("Hide < 8 EMA", value=st.session_state.saved_rank_ema, key="rk_ema", on_change=save_state, args=("rk_ema", "saved_rank_ema"))
        
    filtered = uo.filter_rankings_data(df, st.session_state.saved_rank_start, st.session_state.saved_rank_end)
    if filtered.empty: st.warning("No trades found."); return

    t1, t2, t3 = st.tabs(["🧠 Smart Money", "💡 Top 3", "🤡 Bulltard"])
    mc_val = uo.RANK_MC_THRESHOLDS.get(mc_sel, 1e10)
    bulls, bears, valid = uo.calculate_smart_money_score(df, st.session_state.saved_rank_start, st.session_state.saved_rank_end, mc_val, ema, st.session_state.saved_rank_limit)

    with t1:
        if valid.empty: st.warning("Not enough data.")
        else:
            cfg = {"Symbol": st.column_config.TextColumn("Ticker", width=60), 
                   "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                   "Trade_Count": st.column_config.NumberColumn("Qty", width=50), "Last Trade": st.column_config.TextColumn("Last", width=70)}
            c_a, c_b = st.columns(2, gap="large")
            with c_a: st.markdown("##### Top Bullish"); st.dataframe(bulls[["Symbol","Score","Trade_Count","Last Trade"]], hide_index=True, use_container_width=True, column_config=cfg)
            with c_b: st.markdown("##### Top Bearish"); st.dataframe(bears[["Symbol","Score","Trade_Count","Last Trade"]], hide_index=True, use_container_width=True, column_config=cfg)

    with t2:
        if bulls.empty: st.info("No candidates.")
        else:
            st.caption(f"Analyzing Top {len(bulls)} tickers...")
            ideas = uo.generate_top_ideas(bulls, df)
            cols = st.columns(3)
            for i, idea in enumerate(ideas):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"### #{i+1} {idea['Ticker']}")
                        st.metric("Score", f"{idea['Score']:.1f}", f"${idea['Price']:.2f}")
                        if idea['Suggestions']['Sell Puts']: st.success(f"🛡️ {idea['Suggestions']['Sell Puts']}")
                        if idea['Suggestions']['Buy Calls']: st.info(f"🟢 {idea['Suggestions']['Buy Calls']}")
                        st.divider()
                        for r in idea['Reasons']: st.caption(f"• {r}")

    with t3:
        st.caption("Legacy Volume Ranking")
        v_bull, v_bear = uo.calculate_volume_rankings(filtered, mc_val, ema, st.session_state.saved_rank_limit)
        v1, v2 = st.columns(2)
        with v1: st.markdown("##### Bull Vol"); st.dataframe(v_bull[["Symbol","Trade Count","Last Trade","Score"]], hide_index=True, use_container_width=True)
        with v2: st.markdown("##### Bear Vol"); st.dataframe(v_bear[["Symbol","Trade Count","Last Trade","Score"]], hide_index=True, use_container_width=True)

# --- 3. PIVOT TABLES APP ---
def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_date = uo.get_max_trade_date(df)
    uo.initialize_pivot_state(max_date, max_date)
    def save(k, sk): st.session_state[sk] = st.session_state[k]

    c_filt, c_calc = st.columns([1, 1], gap="medium")
    with c_filt:
        st.markdown("#### Filters")
        f1, f2, f3 = st.columns(3)
        with f1: st.date_input("Start", value=st.session_state.saved_pv_start, key="pv_s", on_change=save, args=("pv_s", "saved_pv_start"))
        with f2: st.date_input("End", value=st.session_state.saved_pv_end, key="pv_e", on_change=save, args=("pv_e", "saved_pv_end"))
        with f3: st.text_input("Ticker", value=st.session_state.saved_pv_ticker, key="pv_t", on_change=save, args=("pv_t", "saved_pv_ticker"))
        
        f4, f5, f6 = st.columns(3)
        with f4: 
            not_opts = list(uo.PIVOT_NOTIONAL_MAP.keys())
            st.selectbox("Min $", not_opts, index=not_opts.index(st.session_state.saved_pv_notional), key="pv_n", on_change=save, args=("pv_n", "saved_pv_notional"))
        with f5: 
            mc_opts = list(uo.PIVOT_MC_MAP.keys())
            st.selectbox("Min Cap", mc_opts, index=mc_opts.index(st.session_state.saved_pv_mkt_cap), key="pv_mc", on_change=save, args=("pv_mc", "saved_pv_mkt_cap"))
        with f6: 
            st.selectbox("> 21 EMA", ["All", "Yes"], index=0 if st.session_state.saved_pv_ema == "All" else 1, key="pv_ema", on_change=save, args=("pv_ema", "saved_pv_ema"))

    with c_calc:
        st.markdown("#### Calculator")
        cc1, cc2, cc3 = st.columns(3)
        with cc1: st.number_input("Strike", value=st.session_state.saved_calc_strike, key="cl_k", on_change=save, args=("cl_k", "saved_calc_strike"))
        with cc2: st.number_input("Prem", value=st.session_state.saved_calc_premium, key="cl_p", on_change=save, args=("cl_p", "saved_calc_premium"))
        with cc3: st.date_input("Exp", value=st.session_state.saved_calc_expiry, key="cl_e", on_change=save, args=("cl_e", "saved_calc_expiry"))
        
        dte = (st.session_state.saved_calc_expiry - date.today()).days
        ret = (st.session_state.saved_calc_premium / st.session_state.saved_calc_strike * 100) if st.session_state.saved_calc_strike else 0
        ann = (ret / dte * 365) if dte > 0 else 0
        
        cc4, cc5, cc6 = st.columns(3)
        with cc4: st.text_input("Ann. Ret", f"{ann:.1f}%", disabled=True)
        with cc5: st.text_input("CoC Ret", f"{ret:.1f}%", disabled=True)
        with cc6: st.text_input("DTE", f"{max(0,dte)}", disabled=True)

    st.markdown('<div class="light-note">ℹ️ Market Cap filtering can be buggy. If empty, reset to 0B.</div>', unsafe_allow_html=True)
    
    d_rng = df[(df["Trade Date"].dt.date >= st.session_state.saved_pv_start) & (df["Trade Date"].dt.date <= st.session_state.saved_pv_end)]
    if d_rng.empty: return
    
    cb, ps, pb, rr = uo.generate_pivot_pools(d_rng)
    min_n = uo.PIVOT_NOTIONAL_MAP[st.session_state.saved_pv_notional]
    min_c = uo.PIVOT_MC_MAP[st.session_state.saved_pv_mkt_cap]
    
    def _draw(df_in, title, is_rr=False):
        st.subheader(title)
        f = uo.filter_pivot_dataframe(df_in, st.session_state.saved_pv_ticker.strip().upper(), min_n, min_c, st.session_state.saved_pv_ema)
        tbl = uo.get_pivot_styled_view(f, is_rr)
        if not tbl.empty:
            st.dataframe(tbl.style.format(uo.PIVOT_TABLE_FMT).map(uo.highlight_expiry, subset=["Expiry_Table"]), 
                         use_container_width=True, hide_index=True, column_config=uo.COLUMN_CONFIG_PIVOT,
                         height=35 * min(len(tbl), 50) + 38)
            
    r1, r2, r3 = st.columns(3)
    with r1: _draw(cb, "Calls Bought")
    with r2: _draw(ps, "Puts Sold")
    with r3: _draw(pb, "Puts Bought")
    
    _draw(rr, "Risk Reversals", True)

# --- 4. STRIKE ZONES APP ---
def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    def_exp = date.today() + timedelta(days=uo.SZ_DEFAULT_EXP_OFFSET)
    uo.initialize_strike_zone_state(def_exp)
    def save(k, sk): st.session_state[sk] = st.session_state[k]

    c_set, c_vis = st.columns([1, 2.5], gap="large")
    with c_set:
        st.text_input("Ticker", value=st.session_state.saved_sz_ticker, key="sz_t", on_change=save, args=("sz_t", "saved_sz_ticker"))
        st.date_input("Start", value=st.session_state.saved_sz_start, key="sz_s", on_change=save, args=("sz_s", "saved_sz_start"))
        st.date_input("End", value=st.session_state.saved_sz_end, key="sz_e", on_change=save, args=("sz_e", "saved_sz_end"))
        st.date_input("Max Exp", value=st.session_state.saved_sz_exp, key="sz_x", on_change=save, args=("sz_x", "saved_sz_exp"))
        
        sb1, sb2 = st.columns(2)
        with sb1: 
            st.radio("View", ["Price Zones", "Expiry Buckets"], index=0 if st.session_state.saved_sz_view=="Price Zones" else 1, key="sz_v", on_change=save, args=("sz_v", "saved_sz_view"))
            wm = st.radio("Width", ["Auto", "Fixed"], index=0 if st.session_state.saved_sz_width_mode=="Auto" else 1, key="sz_wm", on_change=save, args=("sz_wm", "saved_sz_width_mode"))
            if wm == "Fixed": st.select_slider("Size ($)", [1,5,10,25,50,100], value=st.session_state.saved_sz_fixed, key="sz_f", on_change=save, args=("sz_f", "saved_sz_fixed"))
        with sb2:
            st.checkbox("Calls Bought", value=st.session_state.saved_sz_inc_cb, key="sz_cb", on_change=save, args=("sz_cb", "saved_sz_inc_cb"))
            st.checkbox("Puts Sold", value=st.session_state.saved_sz_inc_ps, key="sz_ps", on_change=save, args=("sz_ps", "saved_sz_inc_ps"))
            st.checkbox("Puts Bought", value=st.session_state.saved_sz_inc_pb, key="sz_pb", on_change=save, args=("sz_pb", "saved_sz_inc_pb"))
            
    with c_vis: chart_spot = st.container()
    
    raw = uo.filter_strike_zone_data(df, st.session_state.saved_sz_ticker.strip().upper(), 
                                     st.session_state.saved_sz_start, st.session_state.saved_sz_end, 
                                     st.session_state.saved_sz_exp, 
                                     st.session_state.saved_sz_inc_cb, st.session_state.saved_sz_inc_ps, st.session_state.saved_sz_inc_pb)
    
    if raw.empty: 
        with c_vis: st.warning("No data found."); return
        
    edit_in = raw[["Include", "Trade Date", "Order Type", "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"]].copy()
    edit_in["Dollars"] = pd.to_numeric(edit_in["Dollars"], errors='coerce').fillna(0)
    
    st.subheader("Selection Table")
    edited = st.data_editor(edit_in, hide_index=True, use_container_width=True, key="sz_ed", 
                            column_config={"Include": st.column_config.CheckboxColumn("Include", default=True),
                                           "Dollars": st.column_config.NumberColumn(format="$%d"),
                                           "Expiry_DT": st.column_config.DateColumn("Exp")})
    
    final = raw[edited["Include"]].copy()
    
    with chart_spot:
        if final.empty: st.info("No rows selected.")
        else:
            spot, e8, e21, s200 = uo.get_strike_zone_technicals(st.session_state.saved_sz_ticker.strip().upper())
            def p_diff(x): return f"{(x/spot-1)*100:+.1f}%" if x else "—"
            
            badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
            if e8: badges.append(f'<span class="badge">EMA8: ${e8:,.2f} ({p_diff(e8)})</span>')
            if e21: badges.append(f'<span class="badge">EMA21: ${e21:,.2f} ({p_diff(e21)})</span>')
            if s200: badges.append(f'<span class="badge">SMA200: ${s200:,.2f} ({p_diff(s200)})</span>')
            st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)
            
            if st.session_state.saved_sz_view == "Price Zones":
                st.markdown(uo.generate_price_zones_html(final, spot, st.session_state.saved_sz_width_mode, st.session_state.saved_sz_fixed), unsafe_allow_html=True)
            else:
                st.markdown(uo.generate_expiry_buckets_html(final), unsafe_allow_html=True)
