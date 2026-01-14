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
        # Μικρό μήνυμα υποβοήθησης ακριβώς κάτω από το input
        st.caption("💡 Χρησιμοποιήστε επιθέματα για διεθνή χρηματιστήρια: **.AT** (Αθήνα), **.DE** (Γερμανία), **.L** (Λονδίνο), **.PA** (Παρίσι).")
    
    with col2:
        freq_v = st.selectbox("Συχνότητα Γραφήματος:", ["Daily", "Weekly", "Monthly", "Annual"])

    period_type = st.radio("Επιλογή Περιόδου:", ["Συγκεκριμένο Εύρος", "Όλο το Ιστορικό (Max)"], horizontal=True)
    
    if period_type == "Συγκεκριμένο Εύρος":
        c3, c4 = st.columns(2)
        start_v = c3.date_input("Ημερομηνία Έναρξης:", datetime.now() - timedelta(days=365))
        end_v = c4.date_input("Ημερομηνία Λήξης:", datetime.now())
    else:
        start_v, end_v = None, None

    if st.button("Προβολή Τιμών", type="primary"):
        ticker_obj = yf.Ticker(t1_view)
        
        if period_type == "Όλο το Ιστορικό (Max)":
            raw_v = ticker_obj.history(period="max")
        else:
            raw_v = yf.download(t1_view, start=start_v, end=end_v)

        if raw_v.empty:
            try:
                info = ticker_obj.info
                first_date_epoch = info.get('firstTradeDateEpochUtc')
                if first_date_epoch:
                    first_date = datetime.fromtimestamp(first_date_epoch).date()
                    st.error(f"❌ Δεν υπάρχουν δεδομένα για την επιλεγμένη περίοδο.")
                    st.info(f"📅 Η μετοχή **{t1_view}** ξεκίνησε τη διαπραγμάτευση στις: **{first_date}**")
                else:
                    st.error("Το Ticker δεν βρέθηκε. Βεβαιωθείτε ότι είναι σωστό.")
            except:
                st.error("Σφάλμα σύνδεσης. Ελέγξτε το Ticker.")
        else:
            prices_v = raw_v['Close']
            if freq_v == "Weekly": data_plot = prices_v.resample('W').last()
            elif freq_v == "Monthly": data_plot = prices_v.resample('M').last()
            elif freq_v == "Annual": data_plot = prices_v.resample('Y').last()
            else: data_plot = prices_v
            
            st.subheader(f"Διάγραμμα {freq_v} Τιμών - {t1_view}")
            st.area_chart(data_plot) # Area chart για πιο όμορφο αποτέλεσμα
            st.success(f"Δεδομένα από {data_plot.index.date.min()} έως {data_plot.index.date.max()}")

# --- TAB 2: Beta Analysis (Resampling & Multiple Methods) ---
with tab2:
    st.header("Υπολογισμός Beta")
    freq_b = st.selectbox("Συχνότητα Δεδομένων για Beta:", ["Daily", "Weekly", "Monthly", "Annual"])
    
    c_b1, c_b2 = st.columns(2)
    t1_b = c_b1.text_input("Κύρια Μετοχή:", "AAPL", key="t1b").upper()
    t2_b = c_b2.text_input("Δείκτης (Benchmark):", "^GSPC", key="t2b").upper()
    
    if st.button("Ανάλυση Beta"):
        # Λήψη δεδομένων 5 ετών
        raw_b = yf.download([t1_b, t2_b], start=(datetime.now() - timedelta(days=1825)), end=datetime.now())['Close']
        if not raw_b.empty:
            if freq_b == "Weekly": data_b = raw_b.resample('W').last()
            elif freq_b == "Monthly": data_b = raw_b.resample('M').last()
            elif freq_b == "Annual": data_b = raw_b.resample('Y').last()
            else: data_b = raw_b
            
            s_ret = data_b[t1_b].pct_change().dropna()
            m_ret = data_b[t2_b].pct_change().dropna()
            betas = calculate_all_betas(s_ret, m_ret)
            
            cols_b = st.columns(3)
            for i, (m, v) in enumerate(betas.items()):
                with cols_b[i]:
                    st.metric(m, f"{v[0]:.4f}", f"p={v[1]:.3f}", delta_color="inverse")
            best = min(betas, key=lambda x: betas[x][1])
            st.info(f"Η μέθοδος **{best}** είναι η πιο αξιόπιστη.")

# --- TAB 3: Bond Immunization ---
with tab3:
    st.header("Ανοσοποίηση Ομολόγων")
    ca, cb = st.columns(2)
    f_val = ca.number_input("Ονομαστική Αξία:", value=1000.0)
    c_rate = ca.slider("Κουπόνι:", 0.0, 0.20, 0.05)
    y_mat = cb.number_input("Έτη:", value=10)
    ytm_val = cb.slider("YTM:", 0.0, 0.20, 0.04)
    t_dur = st.number_input("Στόχος Duration:", value=5.0)
    
    if st.button("Υπολογισμός"):
        d, c, p = bond_analysis(f_val, c_rate, y_mat, ytm_val)
        st.metric("Τιμή", f"{p:,.2f} €")
        st.metric("Duration", f"{d:.2f}")
        if abs(d - t_dur) < 0.1: st.success("ΑΝΟΣΟΠΟΙΗΜΕΝΟ")
        else: st.warning(f"Απόκλιση: {d-t_dur:.2f}")

# --- TAB 4: Statman ---
with tab4:
    st.header("Ανάλυση Διαφοροποίησης")
    t_input = st.text_area("Λίστα Tickers (χωρισμένα με κόμμα):", "AAPL, TSLA, MSFT, AMZN, GOOG")
    t_list = [x.strip().upper() for x in t_input.split(",")]
    if st.button("Υπολογισμός Στατιστικών"):
        d_p = yf.download(t_list, period="2y")['Close']
        rets = d_p.pct_change().dropna()
        r_levels = []
        for n in range(1, len(t_list) + 1):
            sub = rets.iloc[:, :n]
            w = np.array([1/n]*n)
            v = np.dot(w.T, np.dot(sub.cov() * 252, w))
            r_levels.append(np.sqrt(v))
        st.line_chart(pd.DataFrame({"Κίνδυνος": r_levels}, index=range(1, len(t_list)+1)))
