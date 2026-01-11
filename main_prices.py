# main_prices.py
#This file contains the UI logic for the Price Divergences, RSI Scanner, Seasonality, and EMA Distance apps.
  
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import yfinance as yf
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils_prices as up

# --- 1. PRICE DIVERGENCES APP ---
def run_price_divergences_app(df_global): # df_global arg kept for signature consistency, unused here
    st.title("📉 Price Divergences")
    
    if 'saved_rsi_div_lookback' not in st.session_state: st.session_state.saved_rsi_div_lookback = 90
    if 'saved_rsi_div_strict' not in st.session_state: st.session_state.saved_rsi_div_strict = "Yes"
    if 'saved_rsi_div_days' not in st.session_state: st.session_state.saved_rsi_div_days = 25
    if 'saved_rsi_div_diff' not in st.session_state: st.session_state.saved_rsi_div_diff = 2.0
    
    def save(k, sk): st.session_state[sk] = st.session_state[k]
    
    datasets = up.get_parquet_config()
    opts = list(datasets.keys())
    
    t1, t2 = st.tabs(["📉 Active Signals", "📜 History"])
    
    with t1:
        sel_ds = st.pills("Dataset", opts, selection_mode="single", default=opts[0] if opts else None, key="div_ds")
        if sel_ds:
            df = up.load_parquet_and_clean(sel_ds)
            if df is not None:
                t_col = next((c for c in df.columns if c in ['TICKER','SYMBOL']), None)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.number_input("Days Since", 1, value=st.session_state.saved_rsi_div_days, key="d_days", on_change=save, args=("d_days","saved_rsi_div_days"))
                with c2: st.number_input("Min Diff", 0.5, value=st.session_state.saved_rsi_div_diff, key="d_diff", on_change=save, args=("d_diff","saved_rsi_div_diff"))
                with c3: st.number_input("Max Candle Gap", 30, value=st.session_state.saved_rsi_div_lookback, key="d_lb", on_change=save, args=("d_lb","saved_rsi_div_lookback"))
                with c4: st.selectbox("Strict", ["Yes","No"], index=0 if st.session_state.saved_rsi_div_strict=="Yes" else 1, key="d_str", on_change=save, args=("d_str","saved_rsi_div_strict"))
                
                strict = (st.session_state.saved_rsi_div_strict == "Yes")
                res = []
                
                if t_col:
                    prog = st.progress(0, "Scanning...")
                    grps = list(df.groupby(t_col))
                    for i, (tkr, g) in enumerate(grps):
                        dd, dw = up.prepare_data(g.copy())
                        if dd is not None: 
                            res.extend(up.find_divergences(dd, tkr, 'Daily', lookback_period=st.session_state.saved_rsi_div_lookback, strict_validation=strict, recent_days_filter=st.session_state.saved_rsi_div_days, rsi_diff_threshold=st.session_state.saved_rsi_div_diff))
                        if dw is not None:
                            res.extend(up.find_divergences(dw, tkr, 'Weekly', lookback_period=st.session_state.saved_rsi_div_lookback, strict_validation=strict, recent_days_filter=st.session_state.saved_rsi_div_days, rsi_diff_threshold=st.session_state.saved_rsi_div_diff))
                        if i%10==0: prog.progress((i+1)/len(grps))
                    prog.empty()
                    
                if res:
                    fin = pd.DataFrame(res)
                    fin = fin[fin["Is_Recent"]].sort_values("Signal_Date_ISO", ascending=False)
                    for tf in ['Daily','Weekly']:
                        for typ in ['Bullish','Bearish']:
                            st.subheader(f"{'🟢' if typ=='Bullish' else '🔴'} {tf} {typ}")
                            sub = fin[(fin['Type']==typ)&(fin['Timeframe']==tf)]
                            if not sub.empty:
                                st.dataframe(sub, use_container_width=True, hide_index=True, 
                                             column_order=["Ticker", "Tags", "Date_Display", "RSI_Display", "Price_Display", "Last_Close"],
                                             column_config={"Tags": st.column_config.ListColumn("Tags")})
                            else: st.info("None.")
                else: st.warning("No signals found.")

    with t2:
        st.caption("Detailed Single Ticker History")
        t_in = st.text_input("Ticker", "AMZN").upper().strip()
        if t_in:
            map_t = up.load_ticker_map()
            raw = up.get_ticker_technicals(t_in, map_t) or up.fetch_yahoo_data(t_in)
            if raw is not None:
                dd, dw = up.prepare_data(raw)
                res_h = []
                if dd is not None: res_h.extend(up.find_divergences(dd, t_in, 'Daily', recent_days_filter=99999))
                if dw is not None: res_h.extend(up.find_divergences(dw, t_in, 'Weekly', recent_days_filter=99999))
                
                if res_h:
                    st.dataframe(pd.DataFrame(res_h).sort_values("Signal_Date_ISO", ascending=False), use_container_width=True, hide_index=True)
                else: st.info("No history found.")

# --- 2. RSI SCANNER APP ---
def run_rsi_scanner_app(df_global):
    st.title("🤖 RSI Scanner")
    t1, t2 = st.tabs(["🤖 Context Backtester", "🔢 Percentiles"])
    
    with t1:
        c1, c2 = st.columns([1,3])
        with c1:
            tkr = st.text_input("Ticker", "NFLX", key="bt_t").upper()
            tol = st.number_input("RSI Tol", 0.5, 10.0, 2.0)
            use_hist = st.checkbox("Past Date?", False)
            ref_d = st.date_input("Ref Date") if use_hist else date.today()
        
        with c2:
            st.markdown("#### Context Filters")
            f200 = st.selectbox("vs 200 SMA", ["Any","Above","Below"])
            f50 = st.selectbox("vs 50 SMA", ["Any","Above","Below"])

        if tkr:
            df = up.fetch_yahoo_data(tkr)
            if df is not None:
                curr = df[df["DATE"].dt.date <= ref_d].copy()
                if not curr.empty:
                    last = curr.iloc[-1]
                    rsi_now = last['RSI']
                    
                    mask = (df['RSI'].between(rsi_now-tol, rsi_now+tol))
                    if f200 == "Above": mask &= (df['CLOSE'] > df['SMA200'])
                    elif f200 == "Below": mask &= (df['CLOSE'] < df['SMA200'])
                    if f50 == "Above": mask &= (df['CLOSE'] > df['SMA50'])
                    elif f50 == "Below": mask &= (df['CLOSE'] < df['SMA50'])
                    
                    matches = df[mask & (df.index < last.name)]
                    st.metric("Matches Found", len(matches), f"Ref RSI: {rsi_now:.1f}")
                    
                    if not matches.empty:
                        res = []
                        for d in [5, 10, 21, 63]:
                            rets = []
                            for idx in matches.index:
                                if idx+d < len(df):
                                    entry = df.loc[idx, 'CLOSE']
                                    exit_p = df.loc[idx+d, 'CLOSE']
                                    rets.append((exit_p-entry)/entry)
                            if rets:
                                res.append({"Days": d, "Count": len(rets), "Win Rate": np.mean(np.array(rets)>0)*100, "Avg Return": np.mean(rets)*100})
                        
                        st.dataframe(pd.DataFrame(res).style.format({"Win Rate":"{:.1f}%", "Avg Return":"{:.2f}%"}), hide_index=True)

    with t2:
        ds = up.get_parquet_config()
        sel = st.pills("Dataset", list(ds.keys()))
        if sel:
            df = up.load_parquet_and_clean(sel)
            if df is not None:
                t_col = next((c for c in df.columns if c in ['TICKER','SYMBOL']), None)
                if t_col:
                    low = st.number_input("Low %", 1, 40, 10)
                    hi = st.number_input("High %", 60, 99, 90)
                    
                    res = []
                    prog = st.progress(0, "Scanning...")
                    grps = list(df.groupby(t_col))
                    for i, (tkr, g) in enumerate(grps):
                        dd, _ = up.prepare_data(g.copy())
                        if dd is not None:
                            res.extend(up.find_rsi_percentile_signals(dd, tkr, low/100, hi/100))
                        if i%10==0: prog.progress((i+1)/len(grps))
                    prog.empty()
                    
                    if res:
                        st.dataframe(pd.DataFrame(res).sort_values("Date_Obj", ascending=False), hide_index=True, use_container_width=True)
                    else: st.info("No signals.")

# --- 3. SEASONALITY APP ---
def run_seasonality_app(df_global):
    st.title("📅 Seasonality")
    t1, t2 = st.tabs(["Single", "Scanner"])
    
    with t1:
        tkr = st.text_input("Ticker", "SPY", key="seas_t").upper()
        if tkr:
            m = up.load_ticker_map()
            df = up.fetch_history_optimized(tkr, m)
            if df is not None:
                df['DATE'] = pd.to_datetime(df['DATE'])
                df.set_index('DATE', inplace=True)
                m_ret = df['CLOSE'].resample('M').last().pct_change()*100
                seas = pd.DataFrame({'Pct': m_ret, 'Month': m_ret.index.month, 'Year': m_ret.index.year})
                
                stats = seas.groupby('Month')['Pct'].agg(['mean', lambda x: (x>0).mean()*100])
                stats.columns = ['Avg %', 'Win Rate']
                
                st.subheader("Monthly Stats")
                st.dataframe(stats.style.format("{:.1f}"), use_container_width=True)
                
                chart_data = stats.reset_index()
                base = alt.Chart(chart_data).encode(x='Month:O')
                bar = base.mark_bar().encode(y='Avg %', color=alt.condition(alt.datum['Avg %']>0, alt.value("#71d28a"), alt.value("#f29ca0")))
                st.altair_chart(bar, use_container_width=True)

    with t2:
        st.caption("Find stocks with high EV for next 21 days")
        if st.button("Run Scan"):
            m = up.load_ticker_map()
            tkrs = [k for k in m.keys() if not k.endswith('_PARQUET')]
            res = []
            
            prog = st.progress(0)
            with ThreadPoolExecutor(max_workers=10) as exe:
                futs = {exe.submit(up.fetch_history_optimized, t, m): t for t in tkrs[:50]} # Limit for demo speed
                done = 0
                for fut in as_completed(futs):
                    t = futs[fut]
                    df = fut.result()
                    if df is not None:
                        df['DATE'] = pd.to_datetime(df['DATE'])
                        curr_doy = date.today().timetuple().tm_yday
                        
                        # Filter historical dates around today's DOY
                        hist = df[df['DATE'].dt.dayofyear.between(curr_doy-3, curr_doy+3)]
                        if len(hist) > 5:
                            rets = []
                            for idx in hist.index:
                                # Check forward 21d
                                try:
                                    # Find matching date in full df
                                    loc_idx = df.index.get_loc(idx)
                                    if loc_idx + 21 < len(df):
                                        entry = df.iloc[loc_idx]['CLOSE']
                                        exit_p = df.iloc[loc_idx+21]['CLOSE']
                                        rets.append((exit_p-entry)/entry)
                                except: pass
                            
                            if rets:
                                res.append({"Ticker": t, "EV 21d": np.mean(rets)*100, "Win Rate": np.mean(np.array(rets)>0)*100})
                    done += 1
                    prog.progress(done/len(futs))
            prog.empty()
            
            if res:
                st.dataframe(pd.DataFrame(res).sort_values("EV 21d", ascending=False).head(50), hide_index=True)

# --- 4. EMA DISTANCE APP ---
def run_ema_distance_app(df_global):
    st.title("📏 EMA Distance")
    c1, c2 = st.columns(2)
    t = c1.text_input("Ticker", "QQQ").upper()
    y = c2.number_input("Years", 1, 20, 5)
    
    if t:
        df = up.fetch_yahoo_data(t)
        if df is not None:
            # Re-calc specific distances
            df['Dist50'] = ((df['CLOSE'] - df['SMA200']) / df['SMA200']) * 100
            curr = df['Dist50'].iloc[-1]
            
            st.metric("Current Dist to 200 SMA", f"{curr:.1f}%")
            
            st.subheader("Historical Distribution")
            chart = alt.Chart(df).mark_rect().encode(
                x=alt.X('Dist50', bin=alt.Bin(maxbins=50)),
                y='count()',
                color=alt.condition(alt.datum.Dist50 > 0, alt.value("#71d28a"), alt.value("#f29ca0"))
            )
            st.altair_chart(chart, use_container_width=True)
            
            st.subheader("Extreme Gap Signals")
            # Logic: If gap > 95th percentile, what happens next?
            p95 = df['Dist50'].quantile(0.95)
            mask = df['Dist50'] > p95
            
            hits = df[mask]
            st.write(f"Occurrences > {p95:.1f}% gap: {len(hits)}")
            if len(hits) > 0:
                rets = []
                for idx in hits.index:
                    if idx + 20 < len(df):
                        e = df.loc[idx, 'CLOSE']
                        x = df.loc[idx+20, 'CLOSE']
                        rets.append((x-e)/e)
                if rets:
                    st.metric("Avg Return (20d later)", f"{np.mean(rets)*100:.2f}%")
