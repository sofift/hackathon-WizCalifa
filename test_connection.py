"""
test_connection.py
Verifica che le credenziali Alpaca Paper Trading funzionino.
Lancia con:  python test_connection.py
"""

import os
import sys
from dotenv import load_dotenv

# Carica le variabili dal file .env nella stessa cartella
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Controllo che le chiavi esistano
if not API_KEY or not SECRET_KEY:
    print("ERRORE: chiavi non trovate nel file .env")
    print("Assicurati che il file .env contenga:")
    print("  ALPACA_API_KEY=...")
    print("  ALPACA_SECRET_KEY=...")
    sys.exit(1)

try:
    from alpaca.trading.client import TradingClient

    # paper=True -> usa l'ambiente di simulazione (non soldi reali)
    client = TradingClient(API_KEY, SECRET_KEY, paper=True)

    account = client.get_account()

    print("=" * 45)
    print("  CONNESSIONE ALPACA OK")
    print("=" * 45)
    print(f"  Account status : {account.status}")
    print(f"  Cash           : {account.cash} USD")
    print(f"  Buying power    : {account.buying_power} USD")
    print(f"  Portfolio value: {account.portfolio_value} USD")
    print(f"  Currency       : {account.currency}")
    print("=" * 45)
    print("  Tutto funziona. Sei pronta per l'hackathon.")

except Exception as e:
    print("ERRORE durante la connessione ad Alpaca:")
    print(f"  {type(e).__name__}: {e}")
    print()
    print("Possibili cause:")
    print("  - chiavi API errate o copiate male")
    print("  - hai usato le chiavi LIVE invece di quelle PAPER")
    print("  - alpaca-py non installato (uv pip install alpaca-py)")
    sys.exit(1)