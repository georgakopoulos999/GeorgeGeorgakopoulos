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
@st.cache_data(ttl=3600)
def fetch_data(ticker, start=None, end=None, period=None):
    try:
        if period:
            df = yf.Ticker(ticker, session=session).history(period=period, auto_adjust=True)
        else:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, session=session)
        return df
    except Exception:
        return pd.DataFrame()

# --- Συναρτήσεις Γραφημάτων ---
def plot_colored_chart(data, ticker_name):
    if data is None or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty):
        return None
    
    # Μετατροπή σε Series αν είναι DataFrame
    if isinstance(data, pd.DataFrame):
        if 'Close' in data.columns:
            data = data['Close']
        else:
            data = data.iloc[:, 0]
    
    data = data.dropna()
    if data.empty: return None

    first_price, last_price = float(data.iloc[0]), float(data.iloc[-1])
    change, pct_change = last_price - first_price, ((last_price - first_price) / first_price) * 100
    
    color = 'rgb(0, 100, 0)' if last_price >= first_price else 'rgb(150, 0, 0)'
    fill = 'rgba(0, 255, 0, 0.2)' if last_price >= first_price else 'rgba(255, 0, 0, 0.2)'

    st.metric(label=f"Τιμή {ticker_name}", value=f"{last_price:.2f}", delta=f"{change:.2f} ({pct_change:.2f}%)")

    fig = go.Figure(go.Scatter(x=data.index, y=data.values, fill='tozeroy', mode='lines',
                               line=dict(color=color, width=3), fillcolor=fill))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=400, template="plotly_white")
    return fig

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock View", "⚖️ Beta Analysis", "⛓️ Bond Immunization", "📉 Statman Diversification"])

# --- TAB 1 ---
with tab1:
    st.header("Επισκόπηση Μετοχής")
    t1_view = st.text_input("Ticker:", "AAPL").upper()
    freq_v = st.selectbox("Συχνότητα:", ["Daily", "Weekly", "Monthly"])
    p_type = st.radio("Περίοδος:", ["Εύρος", "Max"], horizontal=True)
    
    if st.button("Προβολή Τιμών", type="primary"):
        if p_type == "Εύρος":
            start_v = datetime.now() - timedelta(days=365)
            data = fetch_data(t1_view, start=start_v, end=datetime.now())
        else:
            data = fetch_data(t1_view, period="max")
            
        if not data.empty:
            # Δυναμική επιλογή της στήλης Close για αποφυγή KeyError
            plot_data = data['Close'] if 'Close' in data.columns else data
            
            if freq_v == "Weekly": plot_data = plot_data.resample('W').last()
            elif freq_v == "Monthly": plot_data = plot_data.resample('M').last()
            
            st.plotly_chart(plot_colored_chart(plot_data, t1_view), use_container_width=True)
        else:
            st.error("Δεν βρέθηκαν δεδομένα.")

# --- TAB 2 ---
with tab2:
    st.header("Beta Analysis")
    t1_b = st.text_input("Μετοχή:", "AAPL", key="tb1").upper()
    t2_b = st.text_input("Δείκτης:", "^GSPC", key="tb2").upper()
    if st.button("Υπολογισμός Beta"):
        data_b = fetch_data([t1_b, t2_b], period="5y")
        if not data_b.empty:
            # Χειρισμός MultiIndex στήλης Close
            prices = data_b['Close'] if 'Close' in data_b.columns else data_b
            s_ret = prices[t1_b].pct_change().dropna()
            m_ret = prices[t2_b].pct_change().dropna()
            
            # Απλός υπολογισμός Beta (Market Model)
            df_ret = pd.concat([s_ret, m_ret], axis=1).dropna()
            df_ret.columns = ['Stock', 'Market']
            X = sm.add_constant(df_ret['Market'])
            model = sm.OLS(df_ret['Stock'], X).fit()
            
            st.metric("Market Model Beta", f"{model.params['Market']:.4f}", f"p={model.pvalues['Market']:.3f}")

# --- TAB 3 ---
with tab3:
    st.header("Bond Analysis")
    fv = st.number_input("Face Value:", 1000.0)
    cr = st.slider("Coupon:", 0.0, 0.2, 0.05)
    yr = st.number_input("Years:", 10)
    yt = st.slider("YTM:", 0.0, 0.2, 0.04)
    if st.button("Υπολογισμός"):
        times = list(range(1, int(yr) + 1))
        pv_cf = [(cr * fv) / (1 + yt)**t for t in times[:-1]]
        pv_cf.append(((cr * fv) + fv) / (1 + yt)**times[-1])
        price = sum(pv_cf)
        dur = sum([pv * t for pv, t in zip(pv_cf, times)]) / price
        st.metric("Price", f"{price:.2f} €")
        st.metric("Duration", f"{dur:.2f}")

# --- TAB 4 ---
with tab4:
    st.header("Statman Diversification")
    t_in = st.text_area("Tickers:", "AAPL, TSLA, MSFT, AMZN, GOOG")
    if st.button("Ανάλυση"):
        t_list = [x.strip().upper() for x in t_in.split(",")]
        data_s = fetch_data(t_list, period="2y")
        if not data_s.empty:
            prices_s = data_s['Close'] if 'Close' in data_s.columns else data_s
            rets = prices_s.pct_change().dropna()
            risks = []
            for n in range(1, len(t_list)+1):
                subset = rets.iloc[:, :n]
                w = np.array([1/n]*n)
                v = np.dot(w.T, np.dot(subset.cov()*252, w))
                risks.append(np.sqrt(v))
            
            risk_series = pd.Series(risks, index=range(1, len(t_list)+1))
            st.plotly_chart(plot_colored_chart(risk_series, "Portfolio Risk"), use_container_width=True)