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

# --- TAB 1: Stock View ---
with tab1:
    st.header("Επισκόπηση Μετοχής")
    col1, col2 = st.columns([2, 1])
    with col1:
        t1_view = st.text_input("Εισάγετε Ticker:", "AAPL").upper()
        st.caption("💡 Χρησιμοποιήστε επιθέματα: **.AT** (Αθήνα), **.DE** (Γερμανία), **.L** (Λονδίνο).")
    with col2:
        freq_v = st.selectbox("Συχνότητα Γραφήματος:", ["Daily", "Weekly", "Monthly", "Annual"])

    period_type = st.radio("Επιλογή Περιόδου:", ["Συγκεκριμένο Εύρος", "Όλο το Ιστορικό (Max)"], horizontal=True)
    
    if period_type == "Συγκεκριμένο Εύρος":
        c3, c4 = st.columns(2)
        start_v = c3.date_input("Έναρξη:", datetime.now() - timedelta(days=365))
        end_v = c4.date_input("Λήξη:", datetime.now())
    else:
        start_v, end_v = None, None

    if st.button("Προβολή Τιμών", type="primary"):
        ticker_obj = yf.Ticker(t1_view)
        # Χρήση auto_adjust=True για να έχουμε πάντα σωστή στήλη 'Close'
        raw_v = ticker_obj.history(period="max", auto_adjust=True) if period_type == "Όλο το Ιστορικό (Max)" else yf.download(t1_view, start=start_v, end=end_v, auto_adjust=True)

        if raw_v.empty:
            st.error("❌ Δεν βρέθηκαν δεδομένα.")
        else:
            # Διόρθωση γραφήματος: Επιλογή μόνο της στήλης Close
            data_v = raw_v['Close']
            if freq_v == "Weekly": data_plot = data_v.resample('W').last()
            elif freq_v == "Monthly": data_plot = data_v.resample('M').last()
            elif freq_v == "Annual": data_plot = data_v.resample('Y').last()
            else: data_plot = data_v
            
            st.area_chart(data_plot)

# --- TAB 2: Beta Analysis ---
with tab2:
    st.header("Υπολογισμός Beta")
    freq_b = st.selectbox("Συχνότητα Δεδομένων:", ["Daily", "Weekly", "Monthly", "Annual"])
    cb1, cb2 = st.columns(2)
    t1_b = cb1.text_input("Μετοχή:", "AAPL", key="t1b").upper()
    t2_b = cb2.text_input("Benchmark:", "^GSPC", key="t2b").upper()
    
    if st.button("Ανάλυση Beta"):
        raw_b = yf.download([t1_b, t2_b], start=(datetime.now() - timedelta(days=1825)), end=datetime.now(), auto_adjust=True)
        if not raw_b.empty:
            # Flatten MultiIndex αν υπάρχει
            data_b = raw_b['Close']
            if freq_b == "Weekly": data_b = data_b.resample('W').last()
            elif freq_b == "Monthly": data_b = data_b.resample('M').last()
            elif freq_b == "Annual": data_b = data_b.resample('Y').last()
            
            s_ret = data_b[t1_b].pct_change().dropna()
            m_ret = data_b[t2_b].pct_change().dropna()
            betas = calculate_all_betas(s_ret, m_ret)
            
            cols = st.columns(3)
            for i, (m, v) in enumerate(betas.items()):
                cols[i].metric(m, f"{v[0]:.4f}", f"p={v[1]:.3f}")

# --- TAB 4: Statman ---
with tab4:
    st.header("Ανάλυση Διαφοροποίησης Statman")
    t_input = st.text_area("Λίστα Tickers (κόμμα):", "AAPL, TSLA, MSFT, AMZN, GOOG")
    t_list = [x.strip().upper() for x in t_input.split(",")]
    if st.button("Υπολογισμός"):
        # Εδώ διορθώνεται το γράφημα του Statman
        d_p = yf.download(t_list, period="2y", auto_adjust=True)['Close']
        rets = d_p.pct_change().dropna()
        r_levels = []
        for n in range(1, len(t_list) + 1):
            sub = rets.iloc[:, :n]
            w = np.array([1/n]*n)
            v = np.dot(w.T, np.dot(sub.cov() * 252, w))
            r_levels.append(np.sqrt(v))
        
        # Καθαρό γράφημα χωρίς μπερδεμένα labels
        res_df = pd.DataFrame({"Risk": r_levels}, index=range(1, len(t_list)+1))
        st.line_chart(res_df)