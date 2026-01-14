import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Financial Analysis Pro", layout="wide")
st.title("🚀 Financial Analysis & Portfolio Management")

# --- Συναρτήσεις ---
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
    coupons = [coupon_rate * face_value] * int(years)
    coupons[-1] += face_value
    times = list(range(1, int(years) + 1))
    pv_cf = [cf / (1 + ytm)**t for cf, t in zip(coupons, times)]
    price = sum(pv_cf)
    dur = sum([pv * t for pv, t in zip(pv_cf, times)]) / price
    conv = sum([pv * (t**2 + t) for pv, t in zip(pv_cf, times)]) / (price * (1 + ytm)**2)
    return dur, conv, price

# --- Δημιουργία Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Stock View", "⚖️ Beta Analysis", "⛓️ Bond Immunization", "📉 Statman Diversification"])

# --- TAB 1: Stock View (Βελτιωμένο με Ημερομηνίες και Συχνότητα) ---
with tab1:
    st.header("Επισκόπηση Μετοχής")
    
    col1, col2 = st.columns(2)
    t1_view = col1.text_input("Ticker:", "AAPL", key="t1_v").upper()
    freq_v = col2.selectbox("Συχνότητα Γραφήματος:", ["Daily", "Weekly", "Monthly", "Annual"], index=0)
    
    col3, col4 = st.columns(2)
    start_v = col3.date_input("Ημερομηνία Έναρξης:", datetime.now() - timedelta(days=365))
    end_v = col4.date_input("Ημερομηνία Λήξης:", datetime.now())
    
    if st.button("Προβολή Τιμών"):
        # Κατεβάζουμε daily δεδομένα για να κάνουμε σωστό resampling
        raw_v = yf.download(t1_view, start=start_v, end=end_v)
        if not raw_v.empty:
            prices_v = raw_v['Close']
            
            # Resampling βάσει επιλογής χρήστη
            if freq_v == "Weekly": data_plot = prices_v.resample('W').last()
            elif freq_v == "Monthly": data_plot = prices_v.resample('M').last()
            elif freq_v == "Annual": data_plot = prices_v.resample('Y').last()
            else: data_plot = prices_v
            
            st.subheader(f"Διάγραμμα Τιμών ({freq_v}) - {t1_view}")
            st.line_chart(data_plot)
            st.write("Τελευταίες Τιμές:", data_plot.tail())

# --- TAB 2: Advanced Beta Analysis ---
with tab2:
    st.header("Υπολογισμός Beta")
    freq = st.selectbox("Συχνότητα Υπολογισμού Beta:", ["Daily", "Weekly", "Monthly", "Annual"])
    c1, c2 = st.columns(2)
    t1 = c1.text_input("Κύρια Μετοχή:", "AAPL").upper()
    t2 = c2.text_input("Δείκτης Αναφοράς:", "^GSPC").upper()
    if st.button("Εκτέλεση Στατιστικής Ανάλυσης"):
        raw = yf.download([t1, t2], start=(datetime.now() - timedelta(days=1825)), end=datetime.now())
        if not raw.empty:
            prices = raw['Close']
            if freq == "Weekly": data = prices.resample('W').last()
            elif freq == "Monthly": data = prices.resample('M').last()
            elif freq == "Annual": data = prices.resample('Y').last()
            else: data = prices
            
            if t1 in data.columns and t2 in data.columns:
                stock_ret = data[t1].pct_change().dropna()
                market_ret = data[t2].pct_change().dropna()
                all_results = calculate_all_betas(stock_ret, market_ret)
                cols = st.columns(3)
                for i, (method, val) in enumerate(all_results.items()):
                    with cols[i]:
                        st.subheader(method)
                        st.metric("Beta", f"{val[0]:.4f}")
                        st.write(f"P-Value: {val[1]:.4f}")
                best_method = min(all_results, key=lambda x: all_results[x][1])
                st.info(f"💡 Καλύτερη μέθοδος: {best_method}")

# --- TAB 3: Bond Immunization ---
with tab3:
    st.header("Ανοσοποίηση Ομολόγων")
    col_a, col_b = st.columns(2)
    face = col_a.number_input("Ονομαστική Αξία:", value=1000.0)
    coupon = col_a.slider("Ετήσιο Κουπόνι:", 0.0, 0.20, 0.05)
    years = col_b.number_input("Έτη:", value=10)
    ytm = col_b.slider("Απόδοση YTM:", 0.0, 0.20, 0.04)
    if st.button("Υπολογισμός Ομολόγου"):
        dur, conv, price = bond_analysis(face, coupon, years, ytm)
        st.metric("Τιμή", f"{price:,.2f} €")
        st.metric("Duration", f"{dur:.2f}")

# --- TAB 4: Statman Diversification ---
with tab4:
    st.header("Ανάλυση Κινδύνου κατά Statman")
    tickers_input = st.text_area("Εισάγετε Tickers χωρισμένα με κόμμα:", "AAPL, TSLA, MSFT, GOOG, AMZN")
    ticker_list = [t.strip().upper() for t in tickers_input.split(",")]
    
    if st.button("Ανάλυση Διαφοροποίησης"):
        data_port = yf.download(ticker_list, period="2y")['Close']
        if not data_port.empty:
            returns = data_port.pct_change().dropna()
            risk_levels = []
            for i in range(1, len(ticker_list) + 1):
                subset = returns.iloc[:, :i]
                weights = np.array([1/i] * i)
                port_variance = np.dot(weights.T, np.dot(subset.cov() * 252, weights))
                port_std = np.sqrt(port_variance)
                risk_levels.append(port_std)
            
            df_statman = pd.DataFrame({"Αριθμός Μετοχών": range(1, len(ticker_list) + 1), "Κίνδυνος": risk_levels})
            st.line_chart(df_statman.set_index("Αριθμός Μετοχών"))
            reduction = (risk_levels[0] - risk_levels[-1]) / risk_levels[0] * 100
            st.success(f"Μείωση κινδύνου κατά {reduction:.2f}%")
