import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime

def validate_date(date_text):
    try:
        datetime.strptime(date_text, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_ticker(ticker_symbol):
    ticker_info = yf.Ticker(ticker_symbol)
    history = ticker_info.history(period="1d")
    return not history.empty

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

def main():
    while True:
        ticker = input("Εισάγετε το σύμβολο της μετοχής (π.χ. AAPL): ").upper()
        if validate_ticker(ticker): break
        print(f"Σφάλμα: Το σύμβολο '{ticker}' δεν είναι έγκυρο.")

    while True:
        start = input("Εισάγετε ημερομηνία έναρξης (YYYY-MM-DD): ")
        if validate_date(start): break
        print("Λάθος μορφή ημερομηνίας.")
        
    while True:
        end = input("Εισάγετε ημερομηνία λήξης (YYYY-MM-DD): ")
        if validate_date(end) and end > start: break
        print("Η ημερομηνία λήξης πρέπει να είναι μεταγενέστερη.")

    print("\n1: Ιστορικές Τιμές\n2: Υπολογισμό Beta (β)")
    while True:
        mode = input("Επιλογή (1/2): ")
        if mode in ["1", "2"]: break

    if mode == "1":
        print("\nΣυχνότητα: 1: Daily, 2: Monthly, 3: Annual")
        while True:
            f_choice = input("Επιλογή: ")
            if f_choice in ["1", "2", "3"]: break
            
        freq_map = {"1": "1d", "2": "1mo", "3": "1y"}
        # Χρησιμοποιούμε auto_adjust=False για να βλέπουμε σίγουρα το Adj Close αν υπάρχει
        data = yf.download(ticker, start=start, end=end, interval=freq_map[f_choice], auto_adjust=False)
        print(data)

    elif mode == "2":
        print("\n1: Market Model, 2: Scholes-Williams, 3: Dimson")
        while True:
            m_choice = input("Επιλογή (1/2/3): ")
            if m_choice in ["1", "2", "3"]: break
        
        methods_map = {"1": "Market Model", "2": "Scholes and Williams", "3": "Dimson"}
        
        # Λήψη δεδομένων με ρητή επιλογή στήλης για να αποφύγουμε το KeyError
        # Κατεβάζουμε και τα δύο μαζί για να διαχειριστεί το yfinance το formatting σωστά
        all_data = yf.download([ticker, "^GSPC"], start=start, end=end, auto_adjust=False)['Adj Close']
        
        if ticker in all_data.columns and "^GSPC" in all_data.columns:
            stock_ret = all_data[ticker].pct_change().dropna()
            market_ret = all_data["^GSPC"].pct_change().dropna()
            
            beta, p_val = calculate_beta(stock_ret, market_ret, methods_map[m_choice])
            
            print(f"\n>>> Αποτέλεσμα {methods_map[m_choice]}:")
            print(f"Beta (β): {beta:.4f}")
            print(f"P-Value: {p_val:.4f}")
            print("Αξιολόγηση: " + ("Στατιστικά Σημαντικό" if p_val < 0.05 else "Μη Στατιστικά Σημαντικό"))
        else:
            print("Σφάλμα κατά τη λήψη των στηλών Adj Close.")

if __name__ == "__main__":
    main()

