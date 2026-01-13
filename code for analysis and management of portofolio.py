import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta

st.set_page_config(page_title="Financial Analysis Pro", layout="wide")
st.title("🚀 Financial Analysis & Portfolio Management")

def calculate_beta(stock_returns, benchmark_returns):
    df = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    df.columns = ['Stock', 'Benchmark']
    X = sm.add_constant(df['Benchmark'])
    model = sm.OLS(df['Stock'], X).fit()
    return model.params['Benchmark'], model.pvalues['Benchmark']

def bond_analysis(face_value, coupon_rate, years, ytm):
    coupons = [coupon_rate * face_value] * int(years)
    coupons[-1] += face_value
    times = list(range(1, int(years) + 1))
    pv_cf = [cf / (1 + ytm)**t for cf, t in zip(coupons, times)]
    price = sum(pv_cf)
    dur = sum([pv * t for pv, t in zip(pv_cf, times)]) / price
    conv = sum([pv * (t**2 + t) for pv, t in zip(pv_cf, times)]) / (price * (1 + ytm)**2)
    return dur, conv, price

tab1, tab2, tab3 = st.tabs(["📈 Stock Analysis", "⚖️ Beta Hedging", "⛓️ Bond Immunization"])

with tab1:
    st.header("Ανάλυση Μετοχής & Beta")
    freq_label = st.selectbox("Επιλέξτε Συχνότητα Δεδομένων:", ["Daily", "Weekly", "Monthly", "Annual"])
    freq_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo", "Annual": "1y"}
    c1, c2 = st.columns(2)
    t1 = c1.text_input("Κύριο Ticker (π.χ. AAPL):", "AAPL").upper()
    t2 = c2.text_input("Ticker Σύγκρισης (π.χ. ^GSPC):", "^GSPC").upper()
    col_s, col_e = st.columns(2)
    start = col_s.date_input("Έναρξη", datetime.now() - timedelta(days=730))
    end = col_e.date_input("Λήξη", datetime.now())
    if st.button("Εκτέλεση Ανάλυσης"):
        data = yf.download([t1, t2], start=start, end=end, interval=freq_map[freq_label], auto_adjust=False)['Adj Close']
        if not data.empty and t1 in data.columns and t2 in data.columns:
            st.subheader(f"Διάγραμμα Τιμών ({freq_label})")
            st.line_chart(data[t1])
            stock_ret = data[t1].pct_change().dropna()
            bench_ret = data[t2].pct_change().dropna()
            if not stock_ret.empty:
                beta, p_val = calculate_beta(stock_ret, bench_ret)
                st.session_state['current_beta'] = beta
                st.session_state['main_ticker'] = t1
                st.session_state['bench_ticker'] = t2
                res1, res2, res3 = st.columns(3)
                res1.metric(f"Beta (β)", f"{beta:.4f}")
                res2.metric("P-Value", f"{p_val:.4f}")
                res3.metric("Σημαντικότητα", "ΝΑΙ" if p_val < 0.05 else "ΟΧΙ")

with tab2:
    st.header("Στρατηγική Beta-Neutral")
    if 'current_beta' in st.session_state:
        amount = st.number_input("Ποσό επένδυσης (€):", min_value=0.0, value=10000.0)
        hedge = st.session_state['current_beta'] * amount
        st.success(f"Πρέπει να σορτάρετε {hedge:,.2f} € στο {st.session_state['bench_ticker']}.")
    else:
        st.warning("Τρέξτε την ανάλυση στο Tab 1.")

with tab3:
    st.header("Bond Duration & Convexity")
    col_a, col_b = st.columns(2)
    with col_a:
        face = st.number_input("Ονομαστική Αξία:", value=1000.0)
        coupon = st.slider("Ετήσιο Κουπόνι (0.05 = 5%):", 0.0, 0.20, 0.05, step=0.01)
    with col_b:
        years = st.number_input("Έτη μέχρι τη λήξη:", value=10, step=1)
        ytm = st.slider("Απόδοση YTM (0.04 = 4%):", 0.0, 0.20, 0.04, step=0.01)
    target_dur = st.number_input("Επιθυμητή Διάρκεια:", value=5.0)
    if st.button("Υπολογισμός Ανοσοποίησης"):
        dur, conv, price = bond_analysis(face, coupon, years, ytm)
        m1, m2, m3 = st.columns(3)
        m1.metric("Τιμή", f"{price:,.2f} €")
        m2.metric("Duration", f"{dur:.2f}")
        m3.metric("Convexity", f"{conv:.2f}")
        diff = dur - target_dur
        if abs(diff) < 0.1:
            st.success("✅ ΑΝΟΣΟΠΟΙΗΜΕΝΟ")
        else:
            st.warning(f"⚠️ Απόκλιση: {diff:.2f} έτη.")
