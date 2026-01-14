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

# --- Optimization: Session Caching για αποφυγή Rate Limits ---
# Αποθηκεύει τα αιτήματα για 1 ώρα (3600 δευτερόλεπτα)
session = requests_cache.CachedSession('yfinance.cache', expire_after=3600)
session.headers.update({'User-agent': 'my-streamlit-app/1.0'})

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
    
    color = 'rgb(0, 100, 0)' if last_price >= first_price else 'rgb(150, 0, 0)'
    fill = 'rgba(0, 255, 0, 0.2)' if last_price >= first_price else 'rgba(255, 0, 0, 0.2)'

    st.metric(label=f"Τελευταία Τιμή {ticker_name}", 
              value=f"{last_price:.2f}", 
              delta=f"{change:.2f} ({pct_change:.2f}%)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values, fill='tozeroy', mode='lines',
                             line=dict(color=color, width=3), fillcolor=fill, name=str(ticker_name)))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=400, template="plotly_white")
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

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock View", "⚖️ Beta Analysis", "⛓️ Bond Immunization", "📉 Statman Diversification"])

# --- TAB 1: Stock View ---
with tab1:
    st.header("Επισκόπηση Μετοχής")
    col1, col2 = st.columns([2, 1])
    with col1:
        t1_view = st.text_input("Ticker:", "AAPL", key="main_t").upper()
        st.caption("💡 Χρησιμοποιήστε επιθέματα: **.L** (Λονδίνο), **.AT** (Αθήνα), **.DE** (Γερμανία).")
    with col2:
        freq_v = st.selectbox("Συχνότητα:", ["Daily", "Weekly", "Monthly", "Annual"], key="freq_v")

    p_type = st.radio("Περίοδος:", ["Εύρος", "Max"], horizontal=True, key="p1")
    if p_type == "Εύρος":
        c3, c4 = st.columns(2)
        start_v = c3.date_input("Έναρξη:", datetime.now() - timedelta(days=365), key="s1")
        end_v = c4.date_input("Λήξη:", datetime.now(), key="e1")
    else: start_v, end_v = None, None

    if st.button("Προβολή Τιμών", type="primary"):
        try:
            t_obj = yf.Ticker(t1_view, session=session)
            if p_type == "Max":
                raw_v = t_obj.history(period="max", auto_adjust=True)
            else:
                raw_v = yf.download(t1_view, start=start_v, end=end_v, auto_adjust=True, session=session)
            
            if raw_v.empty:
                st.warning(f"Δεν βρέθηκαν δεδομένα για το Ticker: {t1_view}")
            else:
                data_v = raw_v['Close']
                if freq_v == "Weekly": data_v = data_v.resample('W').last()
                elif freq_v == "Monthly": data_v = data_v.resample('M').last()
                elif freq_v == "Annual": data_plot = data_v.resample('Y').last()
                st.plotly_chart(plot_colored_chart(data_v, t1_view), use_container_width=True)
        except Exception as e:
            st.error("⚠️ Rate Limit ή Σφάλμα Σύνδεσης. Παρακαλώ περιμένετε λίγα λεπτά.")

# (Οι υπόλοιπες λειτουργίες Tab 2, 3, 4 παραμένουν ως έχουν, αλλά χρησιμοποιούν το 'session')