import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Ρυθμίσεις Σελίδας ---
st.set_page_config(page_title="Financial Analysis Pro", layout="wide")
st.title("🚀 Financial Analysis & Portfolio Management")

# --- Συναρτήσεις Γραφημάτων ---
def plot_colored_chart(data, ticker_name):
    if data.empty: return None
    
    # Μετατροπή σε Series αν είναι DataFrame για αποφυγή ValueError
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    
    # Καθαρισμός NaN τιμών
    data = data.dropna()
    if data.empty: return None

    first_price = float(data.iloc[0])
    last_price = float(data.iloc[-1])
    change = last_price - first_price
    pct_change = (change / first_price) * 100
    
    # Χρωματική παλέτα βάσει απόδοσης
    if last_price >= first_price:
        line_color = 'rgb(0, 100, 0)'      # Σκούρο Πράσινο
        fill_color = 'rgba(0, 255, 0, 0.2)' # Ανοιχτό Πράσινο
        delta_color = "normal"
    else:
        line_color = 'rgb(150, 0, 0)'      # Σκούρο Κόκκινο
        fill_color = 'rgba(255, 0, 0, 0.2)' # Ανοιχτό Κόκκινο
        delta_color = "inverse"

    # Εμφάνιση Μετρικών πάνω από το γράφημα
    st.metric(label=f"Τελευταία Τιμή {ticker_name}", 
              value=f"{last_price:.2f}", 
              delta=f"{change:.2f} ({pct_change:.2f}%)",
              delta_color=delta_color)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data.values, 
        fill='tozeroy',
        mode='lines',
        line=dict(color=line_color, width=3),
        fillcolor=fill_color,
        name=str(ticker_name)
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=400,
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    return fig

# --- Συναρτήσεις Υπολογισμών ---
def calculate_all_betas(stock_ret, market_ret):
    results = {}
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    df.columns = ['Stock', 'Market']
    
    # 1. Market Model
    X1 = sm.add_constant(df['Market'])
    model1 = sm.OLS(df['Stock'], X1).fit()
    results['Market Model'] = (model1.params['Market'], model1.pvalues['Market'])
    
    # 2. Scholes-Williams
    df['Market_Lag'] = df['Market'].shift(1)
    df['Market_Lead'] = df['Market'].shift(-1)
    df_sw = df.dropna()
    X2 = sm.add_constant(df_sw[['Market', 'Market_Lag', 'Market_Lead']])
    model2 = sm.OLS(df_sw['Stock'], X2).fit()
    beta_sw = model2.params['Market'] + model2.params['Market_Lag'] + model2.params['Market_Lead']
    results['Scholes-Williams'] = (beta_sw, model2.f_pvalue)
    
    # 3. Dimson
    df['Market_Lag1'] = df['Market'].shift(1)
    df['Market_Lag2'] = df['Market'].shift(2)
    df_d = df.dropna()
    X3 = sm.add_constant(df_d[['Market', 'Market_Lag1', 'Market_Lag2']])
    model3 = sm.OLS(df_d['Stock'], X3).fit()
    beta_dimson = model3.params['Market'] + model3.params['Market_Lag1'] + model3.params['Market_Lag2']
    results['Dimson'] = (beta_dimson, model3.f_pvalue)
    return results

def bond_analysis(face_value, coupon_rate, years, ytm):
    if years <= 0: return 0, 0, 0
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
        t1_view = st.text_input("Εισάγετε Ticker:", "AAPL", key="main_t").upper()
        st.caption("💡 Επιθέματα: **.AT** (Αθήνα), **.DE** (Γερμανία), **.L** (Λονδίνο), **.PA** (Παρίσι).")
    with col2:
        freq_v = st.selectbox("Συχνότητα Γραφήματος:", ["Daily", "Weekly", "Monthly", "Annual"], key="freq_v")

    p_type = st.radio("Επιλογή Περιόδου:", ["Συγκεκριμένο Εύρος", "Όλο το Ιστορικό (Max)"], horizontal=True)
    
    if p_type == "Συγκεκριμένο Εύρος":
        c3, c4 = st.columns(2)
        start_v = c3.date_input("Ημερομηνία Έναρξης:", datetime.now() - timedelta(days=365))
        end_v = c4.date_input("Ημερομηνία Λήξης:", datetime.now())
    else:
        start_v, end_v = None, None

    if st.button("Προβολή Τιμών", type="primary"):
        t_obj = yf.Ticker(t1_view)
        
        if p_type == "Όλο το Ιστορικό (Max)":
            raw_v = t_obj.history(period="max", auto_adjust=True)
        else:
            raw_v = yf.download(t1_view, start=start_v, end=end_v, auto_adjust=True)

        if raw_v.empty:
            try:
                info = t_obj.info
                first_trade = info.get('firstTradeDateEpochUtc')
                if first_trade:
                    ipo_date = datetime.fromtimestamp(first_trade).date()
                    st.error(f"❌ Δεν υπάρχουν δεδομένα για αυτή την περίοδο.")
                    st.info(f"📅 Η μετοχή {t1_view} ξεκίνησε τη διαπραγμάτευση στις: **{ipo_date}**")
                else: st.error("Το Ticker δεν βρέθηκε.")
            except: st.error("Σφάλμα σύνδεσης με το API.")
        else:
            data_v = raw_v['Close']
            # Resampling
            if freq_v == "Weekly": data_plot = data_v.resample('W').last()
            elif freq_v == "Monthly": data_plot = data_v.resample('M').last()
            elif freq_v == "Annual": data_plot = data_v.resample('Y').last()
            else: data_plot = data_v
            
            st.plotly_chart(plot_colored_chart(data_plot, t1_view), use_container_width=True)

# --- TAB 2: Beta Analysis ---
with tab2:
    st.header("Υπολογισμός Beta (Στατιστική Ανάλυση)")
    f_b = st.selectbox("Συχνότητα Δεδομένων:", ["Daily", "Weekly", "Monthly", "Annual"], key="fb")
    cb1, cb2 = st.columns(2)
    t1_b = cb1.text_input("Μετοχή (Stock):", "AAPL", key="t1b").upper()
    t2_b = cb2.text_input("Δείκτης (Benchmark):", "^GSPC", key="t2b").upper()
    
    if st.button("Εκτέλεση Ανάλυσης"):
        raw_b = yf.download([t1_b, t2_b], start=(datetime.now() - timedelta(days=1825)), end=datetime.now(), auto_adjust=True)
        if not raw_b.empty:
            data_b = raw_b['Close']
            if f_b == "Weekly": data_b = data_b.resample('W').last()
            elif f_b == "Monthly": data_b = data_b.resample('M').last()
            elif f_b == "Annual": data_b = data_b.resample('Y').last()
            
            if t1_b in data_b.columns and t2_b in data_b.columns:
                s_ret = data_b[t1_b].pct_change().dropna()
                m_ret = data_b[t2_b].pct_change().dropna()
                all_b = calculate_all_betas(s_ret, m_ret)
                
                cols = st.columns(3)
                for i, (m, v) in enumerate(all_b.items()):
                    with cols[i]:
                        st.subheader(m)
                        st.metric("Beta", f"{v[0]:.4f}")
                        st.write(f"P-Value: {v[1]:.4f}")
                best = min(all_b, key=lambda x: all_b[x][1])
                st.info(f"💡 Η μέθοδος **{best}** έχει το χαμηλότερο P-Value (υψηλότερη αξιοπιστία).")

# --- TAB 3: Bond Immunization ---
with tab3:
    st.header("Ανοσοποίηση Ομολόγων")
    ca, cb = st.columns(2)
    f_val = ca.number_input("Ονομαστική Αξία (€):", value=1000.0)
    c_rate = ca.slider("Ετήσιο Κουπόνι (%):", 0.0, 0.2, 0.05)
    y_mat = cb.number_input("Έτη μέχρι τη λήξη:", value=10)
    ytm_val = cb.slider("Απόδοση στη λήξη (YTM %):", 0.0, 0.2, 0.04)
    target_d = st.number_input("Επιθυμητή Διάρκεια (Target Duration):", value=5.0)
    
    if st.button("Υπολογισμός Ομολόγου"):
        d, c, p = bond_analysis(f_val, c_rate, y_mat, ytm_val)
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Τρέχουσα Τιμή", f"{p:,.2f} €")
        col_res2.metric("Duration", f"{d:.2f}")
        if abs(d - target_d) < 0.1: st.success("✅ Το ομόλογο είναι Ανοσοποιημένο.")
        else: st.warning(f"Απόκλιση από τον στόχο: {d - target_d:.2f}")

# --- TAB 4: Statman Diversification ---
with tab4:
    st.header("Ανάλυση Διαφοροποίησης κατά Statman")
    t_in = st.text_area("Εισάγετε Tickers (χωρισμένα με κόμμα):", "AAPL, TSLA, MSFT, AMZN, GOOG, META, NVDA, NFLX")
    t_list = [x.strip().upper() for x in t_in.split(",")]
    
    if st.button("Υπολογισμός Μείωσης Κινδύνου"):
        with st.spinner("Λήψη δεδομένων..."):
            d_s = yf.download(t_list, period="2y", auto_adjust=True)['Close']
            if not d_s.empty:
                # Flatten MultiIndex αν χρειάζεται
                if isinstance(d_s.columns, pd.MultiIndex):
                    d_s.columns = d_s.columns.get_level_values(1)
                
                rets = d_s.pct_change().dropna()
                risks = []
                # Υπολογισμός κινδύνου προσθέτοντας μία-μία μετοχή
                for n in range(1, len(t_list) + 1):
                    sub = rets.iloc[:, :n]
                    weights = np.array([1/n] * n)
                    # Ετήσιος κίνδυνος (Portfolio Variance)
                    v = np.dot(weights.T, np.dot(sub.cov() * 252, weights))
                    risks.append(np.sqrt(v))
                
                # Οπτικοποίηση
                risk_series = pd.Series(risks, index=range(1, len(t_list) + 1))
                st.plotly_chart(plot_colored_chart(risk_series, "Κίνδυνος Χαρτοφυλακίου"), use_container_width=True)
                
                red = ((risks[0] - risks[-1]) / risks[0]) * 100
                st.success(f"⚖️ Προσθέτοντας {len(t_list)} μετοχές, μειώσατε τον κίνδυνο κατά **{red:.2f}%**.")