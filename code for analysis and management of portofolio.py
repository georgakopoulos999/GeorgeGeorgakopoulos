import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Financial Analysis Pro", layout="wide")
st.title("🚀 Financial Analysis & Portfolio Management")

# --- Συναρτήσεις Γραφημάτων ---
def plot_colored_chart(data, ticker_name):
    if data.empty: return None
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data = data.dropna()
    if data.empty: return None

    first_price = float(data.iloc[0])
    last_price = float(data.iloc[-1])
    change = last_price - first_price
    pct_change = (change / first_price) * 100
    
    if last_price >= first_price:
        line_color = 'rgb(0, 100, 0)'      
        fill_color = 'rgba(0, 255, 0, 0.2)' 
        delta_color = "normal"
    else:
        line_color = 'rgb(150, 0, 0)'      
        fill_color = 'rgba(255, 0, 0, 0.2)' 
        delta_color = "inverse"

    st.metric(label=f"Τελευταία Τιμή {ticker_name}", 
              value=f"{last_price:.2f}", 
              delta=f"{change:.2f} ({pct_change:.2f}%)",
              delta_color=delta_color)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data.values, 
        fill='tozeroy',
        mode='lines',
        line=dict(color=line_color, width=3),
        fillcolor=fill_color,
        name=str(ticker_name)
    ))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=400, template="plotly_white", hovermode="x unified")
    return fig

# --- Συναρτήσεις Υπολογισμών ---
def calculate_all_betas(stock_ret, market_ret):
    results = {}
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    df.columns = ['Stock', 'Market']
    X1 = sm.add_constant(df['Market'])
    model1 = sm.OLS(df['Stock'], X1).fit()
    results['Market Model'] = (model1.params['Market'], model1.pvalues['Market'])
    
    df['Market_Lag'] = df['Market'].shift(1)
    df['Market_Lead'] = df['Market'].shift(-1)
    df_sw = df.dropna()
    X2 = sm.add_constant(df_sw[['Market', 'Market_Lag', 'Market_Lead']])
    model2 = sm.OLS(df_sw['Stock'], X2).fit()
    beta_sw = model2.params['Market'] + model2.params['Market_Lag'] + model2.params['Market_Lead']
    results['Scholes-Williams'] = (beta_sw, model2.f_pvalue)
    
    df['Market_Lag1'] = df['Market'].shift(1)
    df['Market_Lag2'] = df['Market'].shift(2)
    df_d = df.dropna()
    X3 = sm.add_constant(df_d[['Market', 'Market_Lag1', 'Market_Lag2']])
    model3 = sm.OLS(df_d['Stock'], X3).fit()
    beta_dimson = model3.params['Market'] + model3.params['Market_Lag1'] + model3.params['Market_Lag2']
    results['Dimson'] = (beta_dimson, model3.f_pvalue)
    return results

def bond_analysis(face_value, coupon_rate, years, ytm):
    if years <= 0: return 0, 0, 0
    coupons = [coupon_rate * face_value] * int(years)
    coupons[-1] += face_value
    times = list(range(1, int(years) + 1))
    pv_cf = [cf / (1 + ytm)**t for cf, t in zip(coupons, times)]
    price = sum(pv_cf)
    dur = sum([pv * t for pv, t in zip(pv_cf, times)]) / price
    conv = sum([pv * (t**2 + t) for pv, t in zip(pv_cf, times)]) / (price * (1 + ytm)**2)
    return dur, conv, price

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock View", "⚖️ Beta Analysis", "⛓️ Bond Immunization", "📉 Statman Diversification"])

# --- TAB 1: Stock View ---
with tab1:
    st.header("Επισκόπηση Μετοχής")
    col1, col2 = st.columns([2, 1])
    with col1:
        t1_view = st.text_input("Εισάγετε Ticker:", "AAPL", key="main_t").upper()
        st.caption("💡 Επιθέματα: **.AT** (Αθήνα), **.DE** (Γερμανία), **.L** (Λονδίνο).")
    with col2:
        freq_v = st.selectbox("Συχνότητα Γραφήματος:", ["Daily", "Weekly", "Monthly", "Annual"], key="freq_v")

    p_type = st.radio("Επιλογή Περιόδου:", ["Εύρος", "Max"], horizontal=True, key="p1")
    if p_type == "Εύρος":
        c3, c4 = st.columns(2)
        start_v = c3.date_input("Έναρξη:", datetime.now() - timedelta(days=365), key="s1")
        end_v = c4.date_input("Λήξη:", datetime.now(), key="e1")
    else: start_v, end_v = None, None

    if st.button("Προβολή Τιμών", type="primary"):
        t_obj = yf.Ticker(t1_view)
        raw_v = t_obj.history(period="max", auto_adjust=True) if p_type == "Max" else yf.download(t1_view, start=start_v, end=end_v, auto_adjust=True)
        if not raw_v.empty:
            data_v = raw_v['Close']
            if freq_v == "Weekly": data_v = data_v.resample('W').last()
            elif freq_v == "Monthly": data_v = data_v.resample('M').last()
            elif freq_v == "Annual": data_v = data_v.resample('Y').last()
            st.plotly_chart(plot_colored_chart(data_v, t1_view), use_container_width=True)

# --- TAB 2: Beta Analysis (ΜΕ ΕΠΙΛΟΓΗ ΠΕΡΙΟΔΟΥ) ---
with tab2:
    st.header("Υπολογισμός Beta")
    
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        t1_b = st.text_input("Μετοχή (Stock):", "AAPL", key="t1b").upper()
        t2_b = st.text_input("Δείκτης (Benchmark):", "^GSPC", key="t2b").upper()
    with col_b2:
        f_b = st.selectbox("Συχνότητα:", ["Daily", "Weekly", "Monthly", "Annual"], key="fb")
    
    # Προσθήκη επιλογής περιόδου για το Beta
    p_type_b = st.radio("Περίοδος Υπολογισμού:", ["Τελευταία 5 Έτη (Default)", "Συγκεκριμένο Εύρος", "Μέγιστο Ιστορικό (Max)"], horizontal=True, key="pb")
    
    if p_type_b == "Συγκεκριμένο Εύρος":
        cb3, cb4 = st.columns(2)
        start_b = cb3.date_input("Έναρξη:", datetime.now() - timedelta(days=1825), key="sb")
        end_b = cb4.date_input("Λήξη:", datetime.now(), key="eb")
    elif p_type_b == "Τελευταία 5 Έτη (Default)":
        start_b, end_b = datetime.now() - timedelta(days=1825), datetime.now()
    else:
        start_b, end_b = "1900-01-01", datetime.now()

    if st.button("Εκτέλεση Ανάλυσης Beta", type="primary"):
        with st.spinner("Υπολογισμός..."):
            raw_b = yf.download([t1_b, t2_b], start=start_b, end=end_b, auto_adjust=True)
            if not raw_b.empty and t1_b in raw_b['Close'].columns and t2_b in raw_b['Close'].columns:
                data_b = raw_b['Close']
                if f_b == "Weekly": data_b = data_b.resample('W').last()
                elif f_b == "Monthly": data_b = data_b.resample('M').last()
                elif f_b == "Annual": data_b = data_b.resample('Y').last()
                
                s_ret = data_b[t1_b].pct_change().dropna()
                m_ret = data_b[t2_b].pct_change().dropna()
                
                all_b = calculate_all_betas(s_ret, m_ret)
                cols = st.columns(3)
                for i, (m, v) in enumerate(all_b.items()):
                    with cols[i]:
                        st.subheader(m)
                        st.metric("Beta", f"{v[0]:.4f}")
                        st.write(f"P-Value: {v[1]:.4f}")
                st.info(f"💡 Ανάλυση από {data_b.index.date.min()} έως {data_b.index.date.max()}")
            else:
                st.error("Αδυναμία λήψης δεδομένων. Ελέγξτε τα Tickers και την περίοδο.")

# --- TAB 3: Bonds ---
with tab3:
    st.header("Ανοσοποίηση Ομολόγων")
    ca, cb = st.columns(2)
    f_val = ca.number_input("Ονομαστική Αξία (€):", value=1000.0)
    c_rate = ca.slider("Ετήσιο Κουπόνι (%):", 0.0, 0.2, 0.05)
    y_mat = cb.number_input("Έτη:", value=10)
    ytm_val = cb.slider("YTM (%):", 0.0, 0.2, 0.04)
    if st.button("Υπολογισμός"):
        d, c, p = bond_analysis(f_val, c_rate, y_mat, ytm_val)
        st.metric("Τρέχουσα Τιμή", f"{p:,.2f} €")
        st.metric("Duration", f"{d:.2f}")

# --- TAB 4: Statman ---
with tab4:
    st.header("Ανάλυση Statman")
    t_in = st.text_area("Tickers (κόμμα):", "AAPL, TSLA, MSFT, AMZN, GOOG")
    t_list = [x.strip().upper() for x in t_in.split(",")]
    if st.button("Ανάλυση"):
        d_s = yf.download(t_list, period="2y", auto_adjust=True)['Close']
        if isinstance(d_s.columns, pd.MultiIndex): d_s.columns = d_s.columns.get_level_values(1)
        rets = d_s.pct_change().dropna()
        risks = []
        for n in range(1, len(t_list) + 1):
            sub = rets.iloc[:, :n]
            weights = np.array([1/n] * n)
            v = np.dot(weights.T, np.dot(sub.cov() * 252, weights))
            risks.append(np.sqrt(v))
        st.plotly_chart(plot_colored_chart(pd.Series(risks, index=range(1, len(t_list) + 1)), "Risk"), use_container_width=True)