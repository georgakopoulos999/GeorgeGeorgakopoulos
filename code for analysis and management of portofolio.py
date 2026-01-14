import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests_cache

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Financial Analysis Pro", layout="wide")
st.title("🚀 Financial Analysis & Portfolio Management")

# --- Optimization: Session Caching ---
session = requests_cache.CachedSession('yfinance.cache', expire_after=3600)

# --- Λειτουργία Caching Δεδομένων ---
@st.cache_data(ttl=3600)  # Αποθήκευση δεδομένων για 1 ώρα
def fetch_data(ticker, start=None, end=None, period=None):
    try:
        if period:
            return yf.Ticker(ticker, session=session).history(period=period, auto_adjust=True)
        else:
            return yf.download(ticker, start=start, end=end, auto_adjust=True, session=session)
    except Exception:
        return pd.DataFrame()

# --- Συναρτήσεις Γραφημάτων ---
def plot_colored_chart(data, ticker_name):
    if data.empty: return None
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data = data.dropna()
    
    first_price, last_price = float(data.iloc[0]), float(data.iloc[-1])
    change, pct_change = last_price - first_price, ((last_price - first_price) / first_price) * 100
    
    color = 'rgb(0, 100, 0)' if last_price >= first_price else 'rgb(150, 0, 0)'
    fill = 'rgba(0, 255, 0, 0.2)' if last_price >= first_price else 'rgba(255, 0, 0, 0.2)'

    st.metric(label=f"Τιμή {ticker_name}", value=f"{last_price:.2f}", delta=f"{change:.2f} ({pct_change:.2f}%)")

    fig = go.Figure(go.Scatter(x=data.index, y=data.values, fill='tozeroy', mode='lines',
                               line=dict(color=color, width=3), fillcolor=fill))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=400, template="plotly_white")
    return fig

# --- Συναρτήσεις Υπολογισμών (Beta & Bonds) ---
def calculate_all_betas(stock_ret, market_ret):
    results = {}
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    df.columns = ['Stock', 'Market']
    # Market Model
    X1 = sm.add_constant(df['Market'])
    model1 = sm.OLS(df['Stock'], X1).fit()
    results['Market Model'] = (model1.params['Market'], model1.pvalues['Market'])
    # Scholes-Williams & Dimson (Συνοπτικά)
    df['Lag'] = df['Market'].shift(1); df['Lead'] = df['Market'].shift(-1)
    df_sw = df.dropna()
    model2 = sm.OLS(df_sw['Stock'], sm.add_constant(df_sw[['Market', 'Lag', 'Lead']])).fit()
    results['Scholes-Williams'] = (model2.params.sum() - model2.params['const'], model2.f_pvalue)
    return results

def bond_analysis(face_value, coupon_rate, years, ytm):
    times = list(range(1, int(years) + 1))
    pv_cf = [(coupon_rate * face_value) / (1 + ytm)**t for t in times[:-1]]
    pv_cf.append(((coupon_rate * face_value) + face_value) / (1 + ytm)**times[-1])
    price = sum(pv_cf)
    dur = sum([pv * t for pv, t in zip(pv_cf, times)]) / price
    return dur, price

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock View", "⚖️ Beta Analysis", "⛓️ Bond Immunization", "📉 Statman Diversification"])

with tab1:
    st.header("Επισκόπηση Μετοχής")
    t1_view = st.text_input("Ticker:", "AAPL").upper()
    freq_v = st.selectbox("Συχνότητα:", ["Daily", "Weekly", "Monthly"])
    p_type = st.radio("Περίοδος:", ["Εύρος", "Max"], horizontal=True)
    
    if p_type == "Εύρος":
        start_v = st.date_input("Έναρξη:", datetime.now() - timedelta(days=365))
        end_v = st.date_input("Λήξη:", datetime.now())
        if st.button("Προβολή"):
            data = fetch_data(t1_view, start=start_v, end=end_v)
            st.plotly_chart(plot_colored_chart(data['Close'], t1_view), use_container_width=True)
    else:
        if st.button("Προβολή Max"):
            data = fetch_data(t1_view, period="max")
            st.plotly_chart(plot_colored_chart(data['Close'], t1_view), use_container_width=True)

with tab2:
    st.header("Beta Analysis")
    t1_b = st.text_input("Μετοχή:", "AAPL", key="tb1").upper()
    t2_b = st.text_input("Δείκτης:", "^GSPC", key="tb2").upper()
    if st.button("Υπολογισμός Beta"):
        data_b = fetch_data([t1_b, t2_b], period="5y")['Close']
        s_ret = data_b[t1_b].pct_change().dropna()
        m_ret = data_b[t2_b].pct_change().dropna()
        res = calculate_all_betas(s_ret, m_ret)
        for m, v in res.items():
            st.metric(m, f"{v[0]:.4f}", f"p={v[1]:.3f}")

with tab3:
    st.header("Bond Analysis")
    fv = st.number_input("Face Value:", 1000.0); cr = st.slider("Coupon:", 0.0, 0.2, 0.05)
    yr = st.number_input("Years:", 10); yt = st.slider("YTM:", 0.0, 0.2, 0.04)
    if st.button("Υπολογισμός Ομολόγου"):
        d, p = bond_analysis(fv, cr, yr, yt)
        st.metric("Price", f"{p:.2f} €"); st.metric("Duration", f"{d:.2f}")

with tab4:
    st.header("Statman Diversification")
    t_in = st.text_area("Tickers:", "AAPL, TSLA, MSFT, AMZN, GOOG")
    if st.button("Ανάλυση"):
        t_list = [x.strip().upper() for x in t_in.split(",")]
        data_s = fetch_data(t_list, period="2y")['Close']
        rets = data_s.pct_change().dropna()
        risks = [np.sqrt(np.dot(np.array([1/n]*n).T, np.dot(rets.iloc[:, :n].cov()*252, np.array([1/n]*n)))) for n in range(1, len(t_list)+1)]
        st.plotly_chart(plot_colored_chart(pd.Series(risks), "Risk Reduction"), use_container_width=True)