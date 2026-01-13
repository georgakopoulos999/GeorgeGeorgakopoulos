import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Financial Analysis Pro", layout="wide")
st.title("🚀 Financial Analysis & Portfolio Management")

# --- Συναρτήσεις Υπολογισμών ---
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

# --- Δημιουργία Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Stock Analysis", "⚖️ Beta Hedging", "⛓️ Bond Immunization"])

# --- TAB 1: Ανάλυση Μετοχής ---
with tab1:
    st.header("Ανάλυση Μετοχής & Beta")
    
    # Επιλογή Συχνότητας (Optimization: Προσθήκη Weekly & Annual)
    col_freq = st.columns(1)[0]
    freq_label = col_freq.selectbox("Επιλέξτε Συχνότητα Δεδομένων:", 
                                  ["Daily", "Weekly", "Monthly", "Annual"])
    
    # Χάρτης για το yfinance
    freq_map = {
        "Daily": "1d",
        "Weekly": "1wk",
        "Monthly": "1mo",
        "Annual": "1y" # Σημείωση: Το 1y δουλεύει καλύτερα ως resampling αν το yfinance έχει κενά
    }
    
    c1, c2 = st.columns(2)
    t1 = c1.text_input("Κύριο Ticker (π.χ. AAPL):", "AAPL").upper()
    t2 = c2.text_input("Ticker Σύγκρισης (π.χ. ^GSPC):", "^GSPC").upper()
    
    col_s, col_e = st.columns(2)
    start = col_s.date_input("Έναρξη", datetime.now() - timedelta(days=365*2))
    end = col_e.date_input("Λήξη", datetime.now())
    
    if st.button("Εκτέλεση Ανάλυσης"):
        with st.spinner('Λήψη δεδομένων...'):
            # Λήψη δεδομένων με τη σωστή συχνότητα
            data = yf.download([t1, t2], start=start, end=end, interval=freq_map[freq_label], auto_adjust=False)['Adj Close']
            
            if not data.empty and t1 in data.columns and t2 in data.columns:
                st.subheader(f"Διάγραμμα Τιμών ({freq_label})")
                st.line_chart(data[t1])
                
                # Υπολογισμός αποδόσεων
                stock_ret = data[t1].pct_change().dropna()
                bench_ret = data[t2].pct_change().dropna()
                
                if not stock_ret.empty:
                    beta, p_val = calculate_beta(stock_ret, bench_ret)
                    
                    # Αποθήκευση του beta στο session_state για να το βλέπει το Tab 2
                    st.session_state['current_beta'] = beta
                    st.session_state['main_ticker'] = t1
                    st.session_state['bench_ticker'] = t2
                    
                    res1, res2, res3 = st.columns(3)
                    res1.metric(f"Beta (β) - {freq_label}", f"{beta:.4f}")
                    res2.metric("P-Value", f"{p_val:.4f}")
                    res3.metric("Σημαντικότητα", "ΝΑΙ" if p_val < 0.05 else "ΟΧΙ")
                    
                    # Scatter Plot για οπτική επιβεβαίωση
                    st.subheader("Συσχέτιση Αποδόσεων")
                    scatter_df = pd.concat([stock_ret, bench_ret], axis=1)
                    st.scatter_chart(scatter_df)
                else:
                    st.error("Δεν υπάρχουν αρκετά δεδομένα για τον υπολογισμό των αποδόσεων.")
            else:
                st.error("Σφάλμα στη λήψη δεδομένων. Ελέγξτε τα Tickers.")

# --- TAB 2: Beta Neutrality ---
with tab2:
    st.header("Στρατηγική Beta-Neutral")
    if 'current_beta' in st.session_state:
        amount = st.number_input("Ποσό επένδυσης (€):", min_value=0.0, value=10000.0)
        beta_val = st.session_state['current_beta']
        t1_val = st.session_state['main_ticker']
        t2_val = st.session_state['bench_ticker']
        
        hedge = beta_val * amount
        st.write(f"Με βάση την **{freq_label}** ανάλυση:")
        st.success(f"Για να καλύψετε τη θέση σας στο **{t1_val}**, πρέπει να σορτάρετε **{hedge:,.2f} €** στον δείκτη **{t2_val}**.")
    else:
        st.warning("Παρακαλώ τρέξτε πρώτα την ανάλυση στο Tab 1 για να υπολογιστεί το Beta.")

# --- TAB 3: Ανοσοποίηση Ομολόγων ---
with tab3:
    st.header("Bond Duration & Convexity")
    col_a, col_b = st.columns(2)
    with col_a:
        face = st.number_input("Ονομαστική Αξία:", value=1000.0)
        coupon = st.slider("Ετήσιο Κουπόνι (0.05 = 5%):", 0.0, 0.20, 0.05, step=0.01)
    with col_b:
      years = st.number_input("Έτη μέχρι τη λήξη:", value=10, step=1)
        ytm = st.slider("Απόδοση YTM (0.04 = 4%):", 0.0, 0.20, 0.04, step=0.01)

