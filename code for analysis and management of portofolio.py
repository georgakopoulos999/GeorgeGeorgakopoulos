import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Financial Analysis Pro", layout="wide")
st.title("🚀 Financial Analysis & Portfolio Management")

# --- Συναρτήσεις Υπολογισμών Beta ---
def calculate_all_betas(stock_ret, market_ret):
    results = {}
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    df.columns = ['Stock', 'Market']
    
    # 1. Market Model (Simple OLS)
    X1 = sm.add_constant(df['Market'])
    model1 = sm.OLS(df['Stock'], X1).fit()
    results['Market Model'] = (model1.params['Market'], model1.pvalues['Market'])
    
    # 2. Scholes and Williams
    df['Market_Lag'] = df['Market'].shift(1)
    df['Market_Lead'] = df['Market'].shift(-1)
    df_sw = df.dropna()
    X2 = sm.add_constant(df_sw[['Market', 'Market_Lag', 'Market_Lead']])
    model2 = sm.OLS(df_sw['Stock'], X2).fit()
    beta_sw = model2.params['Market'] + model2.params['Market_Lag'] + model2.params['Market_Lead']
    results['Scholes-Williams'] = (beta_sw, model2.f_pvalue)
    
    # 3. Dimson (Aggregated Coefficients)
    df['Market_Lag1'] = df['Market'].shift(1)
    df['Market_Lag2'] = df['Market'].shift(2)
    df_d = df.dropna()
    X3 = sm.add_constant(df_d[['Market', 'Market_Lag1', 'Market_Lag2']])
    model3 = sm.OLS(df_d['Stock'], X3).fit()
    beta_dimson = model3.params['Market'] + model3.params['Market_Lag1'] + model3.params['Market_Lag2']
    results['Dimson'] = (beta_dimson, model3.f_pvalue)
    
    return results

def bond_analysis(face_value, coupon_rate, years, ytm):
    coupons = [coupon_rate * face_value] * int(years)
    coupons[-1] += face_value
    times = list(range(1, int(years) + 1))
    pv_cf = [cf / (1 + ytm)**t for cf, t in zip(coupons, times)]
    price = sum(pv_cf)
    dur = sum([pv * t for pv, t in zip(pv_cf, times)]) / price
    conv = sum([pv * (t**2 + t) for pv, t in zip(pv_cf, times)]) / (price * (1 + ytm)**2)
    return dur, conv, price

# --- Δημιουργία Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Stock View", "⚖️ Advanced Beta Analysis", "⛓️ Bond Immunization"])

# --- TAB 1: Stock View ---
with tab1:
    st.header("Επισκόπηση Μετοχής")
    t1_view = st.text_input("Ticker:", "AAPL", key="t1_v").upper()
    if st.button("Προβολή"):
        data_v = yf.download(t1_view, period="1y")
        st.line_chart(data_v['Adj Close'])

# --- TAB 2: Advanced Beta Analysis ---
with tab2:
    st.header("Υπολογισμός Beta (Market, Scholes-Williams, Dimson)")
    
    freq = st.selectbox("Συχνότητα:", ["Daily", "Weekly", "Monthly", "Annual"])
    c1, c2 = st.columns(2)
    t1 = c1.text_input("Κύρια Μετοχή:", "AAPL").upper()
    t2 = c2.text_input("Δείκτης Αναφοράς:", "^GSPC").upper()
    
    if st.button("Εκτέλεση Στατιστικής Ανάλυσης"):
        # Λήψη δεδομένων 5 ετών για αξιοπιστία
        raw = yf.download([t1, t2], start=(datetime.now() - timedelta(days=1825)), end=datetime.now())['Adj Close']
        
        if not raw.empty and t1 in raw.columns:
            # Resampling Logic
            if freq == "Weekly": data = raw.resample('W').last()
            elif freq == "Monthly": data = raw.resample('M').last()
            elif freq == "Annual": data = raw.resample('Y').last()
            else: data = raw
            
            stock_ret = data[t1].pct_change().dropna()
            market_ret = data[t2].pct_change().dropna()
            
            # Υπολογισμός και με τις 3 μεθόδους
            all_results = calculate_all_betas(stock_ret, market_ret)
            
            # Εμφάνιση Αποτελεσμάτων σε στήλες
            cols = st.columns(3)
            for i, (method, val) in enumerate(all_results.items()):
                with cols[i]:
                    st.subheader(method)
                    st.metric("Beta", f"{val[0]:.4f}")
                    st.write(f"P-Value: {val[1]:.4f}")
                    if val[1] < 0.05:
                        st.success("Στατιστικά Σημαντικό")
                    else:
                        st.warning("Μη Σημαντικό")

            # Εύρεση της καλύτερης μεθόδου
            best_method = min(all_results, key=lambda x: all_results[x][1])
            st.divider()
            st.info(f"💡 Η πιο αξιόπιστη μέθοδος για το συγκεκριμένο δείγμα είναι η **{best_method}** (χαμηλότερο P-Value).")

# --- TAB 3: Bond Immunization ---
with tab3:
    st.header("Ανοσοποίηση Ομολόγων")
    col_a, col_b = st.columns(2)
    with col_a:
        face = st.number_input("Ονομαστική Αξία:", value=1000.0)
        coupon = st.slider("Ετήσιο Κουπόνι:", 0.0, 0.20, 0.05, step=0.01)
    with col_b:
        years = st.number_input("Έτη:", value=10, step=1)
        ytm
