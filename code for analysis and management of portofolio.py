import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Portfolio Analysis Tool", layout="wide")
st.title("📊 Ανάλυση Μετοχών & Υπολογισμός Beta")

# --- Συναρτήσεις Υπολογισμού ---
def calculate_beta(stock_returns, market_returns, method):
    df = pd.concat([stock_returns, market_returns], axis=1).dropna()
    df.columns = ['Stock', 'Market']
    
    if method == "Market Model":
        X = sm.add_constant(df['Market'])
        model = sm.OLS(df['Stock'], X).fit()
        return model.params['Market'], model.pvalues['Market']

    elif method == "Scholes and Williams":
        df['Market_Lag'] = df['Market'].shift(1)
        df['Market_Lead'] = df['Market'].shift(-1)
        df = df.dropna()
        X = sm.add_constant(df[['Market', 'Market_Lag', 'Market_Lead']])
        model = sm.OLS(df['Stock'], X).fit()
        beta_sw = (model.params['Market'] + model.params['Market_Lag'] + model.params['Market_Lead'])
        return beta_sw, model.f_pvalue

    elif method == "Dimson":
        df['Market_Lag1'] = df['Market'].shift(1)
        df['Market_Lag2'] = df['Market'].shift(2)
        df = df.dropna()
        X = sm.add_constant(df[['Market', 'Market_Lag1', 'Market_Lag2']])
        model = sm.OLS(df['Stock'], X).fit()
        beta_dimson = model.params['Market'] + model.params['Market_Lag1'] + model.params['Market_Lag2']
        return beta_dimson, model.f_pvalue

# --- Sidebar για Εισαγωγή Δεδομένων ---
st.sidebar.header("Παράμετροι Ανάλυσης")
ticker = st.sidebar.text_input("Σύμβολο Μετοχής (π.χ. AAPL, TSLA)", "AAPL").upper()

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Έναρξη", datetime.now() - timedelta(days=365))
end_date = col2.date_input("Λήξη", datetime.now())

analysis_mode = st.sidebar.radio("Λειτουργία:", ["Ιστορικές Τιμές", "Υπολογισμός Beta (β)"])

# --- Κύριο Πρόγραμμα ---
if ticker:
    try:
        if analysis_mode == "Ιστορικές Τιμές":
            freq = st.selectbox("Συχνότητα:", ["Daily", "Monthly", "Annual"])
            freq_map = {"Daily": "1d", "Monthly": "1mo", "Annual": "1y"}
            
            with st.spinner('Λήψη δεδομένων...'):
                data = yf.download(ticker, start=start_date, end=end_date, interval=freq_map[freq], auto_adjust=False)
            
            if not data.empty:
                st.subheader(f"Δεδομένα για τη μετοχή {ticker}")
                st.line_chart(data['Adj Close'])
                st.write(data)
            else:
                st.error("Δεν βρέθηκαν δεδομένα. Ελέγξτε το σύμβολο και τις ημερομηνίες.")

        elif analysis_mode == "Υπολογισμός Beta (β)":
            method = st.selectbox("Μέθοδος Υπολογισμού:", ["Market Model", "Scholes and Williams", "Dimson"])
            
            with st.spinner('Υπολογισμός...'):
                all_data = yf.download([ticker, "^GSPC"], start=start_date, end=end_date, auto_adjust=False)['Adj Close']
                
                if ticker in all_data.columns and "^GSPC" in all_data.columns:
                    stock_ret = all_data[ticker].pct_change().dropna()
                    market_ret = all_data["^GSPC"].pct_change().dropna()
                    
                    beta, p_val = calculate_beta(stock_ret, market_ret, method)
                    
                    # Εμφάνιση Αποτελεσμάτων σε "Κάρτες"
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Beta (β)", f"{beta:.4f}")
                    c2.metric("P-Value", f"{p_val:.4f}")
                    significance = "Σημαντικό" if p_val < 0.05 else "Μη Σημαντικό"
                    c3.metric("Στατιστική Σημαντικότητα", significance)
                    
                    # Γράφημα Συσχέτισης
                    st.subheader("Διάγραμμα Συσχέτισης (Returns Analysis)")
                    chart_data = pd.concat([stock_ret, market_ret], axis=1)
                    st.scatter_chart(chart_data)
                else:
                    st.error("Αδυναμία λήψης δεδομένων για τον υπολογισμό του Beta.")

    except Exception as e:
        st.error(f"Παρουσιάστηκε σφάλμα: {e}")
else:
    st.info("Παρακαλώ εισάγετε ένα σύμβολο μετοχής στη sidebar για να ξεκινήσετε.")