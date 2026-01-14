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
    first_price = data.iloc[0]
    last_price = data.iloc[-1]
    
    # Χρωματική παλέτα βάσει απόδοσης
    if last_price >= first_price:
        line_color = 'rgb(0, 100, 0)'      # Σκούρο Πράσινο
        fill_color = 'rgba(0, 255, 0, 0.3)' # Ανοιχτό Πράσινο
    else:
        line_color = 'rgb(150, 0, 0)'      # Σκούρο Κόκκινο
        fill_color = 'rgba(255, 0, 0, 0.3)' # Ανοιχτό Κόκκινο

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data.values, 
        fill='tozeroy',
        mode='lines',
        line=dict(color=line_color, width=3),
        fillcolor=fill_color,
        name=ticker_name
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# --- Συναρτήσεις Υπολογισμών ---
def calculate_all_betas(stock_ret, market_ret):
    results = {}
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    df.columns = ['Stock', 'Market']
    
    # Market Model
    X1 = sm.add_constant(df['Market'])
    model1 = sm.OLS(df['Stock'], X1).fit()
    results['Market Model'] = (model1.params['Market'], model1.pvalues['Market'])
    
    # Scholes-Williams
    df['Market_Lag'] = df['Market'].shift(1)
    df['Market_Lead'] = df['Market'].shift(-1)
    df_sw = df.dropna()
    X2 = sm.add_constant(df_sw[['Market', 'Market_Lag', 'Market_Lead']])
    model2 = sm.OLS(df_sw['Stock'], X2).fit()
    beta_sw = model2.params['Market'] + model2.params['Market_Lag'] + model2.params['Market_Lead']
    results['Scholes-Williams'] = (beta_sw, model2.f_pvalue)
    
    # Dimson
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
    t1_view = col1.text_input("Ticker (π.χ. AAPL):", "AAPL", key="main_t").upper()
    col1.caption("💡 Για διεθνή χρηματιστήρια: **.AT** (Αθήνα), **.DE** (Γερμανία), **.L** (Λονδίνο).")
    freq_v = col2.selectbox("Συχνότητα:", ["Daily", "Weekly", "Monthly", "Annual"], key="freq_v")

    p_type = st.radio("Περίοδος:", ["Εύρος", "Όλο το Ιστορικό (Max)"], horizontal=True)
    if p_type == "Εύρος":
        c3, c4 = st.columns(2)
        start_v = c3.date_input("Έναρξη:", datetime.now() - timedelta(days=365))
        end_v = c4.date_input("Λήξη:", datetime.now())
    else: start_v, end_v = None, None

    if st.button("Προβολή Τιμών", type="primary"):
        t_obj = yf.Ticker(t1_view)
        raw = t_obj.history(period="max", auto_adjust=True) if p_type == "Όλο το Ιστορικό (Max)" else yf.download(t1_view, start=start_v, end=end_v, auto_adjust=True)
        
        if raw.empty:
            try:
                info = t_obj.info
                ipo = datetime.fromtimestamp(info.get('firstTradeDateEpochUtc')).date()
                st.error(f"❌ Δεν υπάρχουν δεδομένα. Η μετοχή ξεκίνησε στις: {ipo}")
            except: st.error("Ticker μη έγκυρο.")
        else:
            data = raw['Close']
            if freq_v == "Weekly": data = data.resample('W').last()
            elif freq_v == "Monthly": data = data.resample('M').last()
            elif freq_v == "Annual": data = data.resample('Y').last()
            st.plotly_chart(plot_colored_chart(data, t1_view), use_container_width=True)

# --- TAB 2: Beta Analysis ---
with tab2:
    st.header("Προηγμένη Ανάλυση Beta")
    f_b = st.selectbox("Συχνότητα Υπολογισμού:", ["Daily", "Weekly", "Monthly", "Annual"], key="fb")
    cb1, cb2 = st.columns(2)
    t1_b = cb1.text_input("Μετοχή:", "AAPL", key="t1b").upper()
    t2_b = cb2.text_input("Δείκτης:", "^GSPC", key="t2b").upper()
    
    if st.button("Υπολογισμός Beta"):
        raw_b = yf.download([t1_b, t2_b], start=(datetime.now()-timedelta(days=1825)), end=datetime.now(), auto_adjust=True)['Close']
        if not raw_b.empty:
            if f_b == "Weekly": data_b = raw_b.resample('W').last()
            elif f_b == "Monthly": data_b = raw_b.resample('M').last()
            elif f_b == "Annual": data_b = raw_b.resample('Y').last()
            else: data_b = raw_b
            
            s_ret, m_ret = data_b[t1_b].pct_change().dropna(), data_b[t2_b].pct_change().dropna()
            all_b = calculate_all_betas(s_ret, m_ret)
            cols = st.columns(3)
            for i, (m, v) in enumerate(all_b.items()):
                cols[i].metric(m, f"{v[0]:.4f}", f"p={v[1]:.3f}")
            best = min(all_b, key=lambda x: all_b[x][1])
            st.info(f"💡 Η μέθοδος **{best}** είναι η στατιστικά επικρατέστερη.")

# --- TAB 3: Bonds ---
with tab3:
    st.header("Ανοσοποίηση Ομολόγων")
    ca, cb = st.columns(2)
    fv, cr = ca.number_input("Face Value:", 1000.0), ca.slider("Coupon:", 0.0, 0.2, 0.05)
    yr, yt = cb.number_input("Years:", 10), cb.slider("YTM:", 0.0, 0.2, 0.04)
    if st.button("Ανάλυση Ομολόγου"):
        d, c, p = bond_analysis(fv, cr, yr, yt)
        st.metric("Price", f"{p:,.2f} €")
        st.metric("Duration", f"{d:.2f}")

# --- TAB 4: Statman ---
with tab4:
    st.header("Διαφοροποίηση κατά Statman")
    t_in = st.text_area("Λίστα (κόμμα):", "AAPL, TSLA, MSFT, AMZN, GOOG")
    t_l = [x.strip().upper() for x in t_in.split(",")]
    if st.button("Ανάλυση Κινδύνου"):
        d_s = yf.download(t_l, period="2y", auto_adjust=True)['Close']
        rets = d_s.pct_change().dropna()
        risks = []
        for n in range(1, len(t_l)+1):
            sub = rets.iloc[:, :n]
            w = np.array([1/n]*n)
            v = np.dot(w.T, np.dot(sub.cov()*252, w))
            risks.append(np.sqrt(v))
        st.plotly_chart(plot_colored_chart(pd.Series(risks, index=range(1, len(t_l)+1)), "Portfolio Risk"), use_container_width=True)
        st.success(f"Μείωση κινδύνου: {((risks[0]-risks[-1])/risks[0])*100:.2f}%")