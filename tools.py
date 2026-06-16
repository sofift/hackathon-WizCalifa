"""
Tool functions deterministiche — nessuna invenzione di dati.
Tutto ciò che l'LLM usa proviene da queste chiamate reali.
"""

import os
import requests
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest


# ---------------------------------------------------------------------------
# Client Alpaca (inizializzati una volta sola al momento dell'import)
# ---------------------------------------------------------------------------

ALPACA_API_KEY      = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY   = os.environ["ALPACA_SECRET_KEY"]
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

trading_client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=True,         # Paper Trading — nessun soldo reale
)

stock_data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)

crypto_data_client = CryptoHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)


def _is_crypto(ticker: str) -> bool:
    """Restituisce True se il ticker è un asset crypto (es. BTC/USD, ETH/USD)."""
    return "/" in ticker


# ---------------------------------------------------------------------------
# Tool 1: get_price
# ---------------------------------------------------------------------------

def get_price(ticker: str) -> dict:
    """
    Restituisce il prezzo ask più recente per il ticker.
    Supporta sia azioni (es. AAPL) che crypto (es. BTC/USD).
    Ritorna {"price": float, "ticker": str} oppure {"error": str}.
    """
    try:
        if _is_crypto(ticker):
            req = CryptoLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = crypto_data_client.get_crypto_latest_quote(req)
            ask_price = quote[ticker].ask_price
        else:
            req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = stock_data_client.get_stock_latest_quote(req)
            ask_price = quote[ticker].ask_price

        if ask_price is None or ask_price == 0:
            return {"error": f"Prezzo non disponibile per {ticker}"}
        return {"ticker": ticker, "price": float(ask_price)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 2: search_news
# ---------------------------------------------------------------------------

def _fetch_alpaca_news(ticker: str) -> list[str]:
    """Ritorna titoli da Alpaca News Feed (ultimi 2 giorni, max 5)."""
    # Per crypto, Alpaca News usa il simbolo senza slash (es. BTCUSD)
    news_symbol = ticker.replace("/", "") if _is_crypto(ticker) else ticker
    url = "https://data.alpaca.markets/v1beta1/news"
    params = {
        "symbols": news_symbol,
        "limit": 5,
        "start": (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    headers = {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("news", [])
    return [f"[Alpaca] {item['headline']}" for item in items]


def _fetch_twelve_data_news(ticker: str) -> list[str]:
    """Ritorna titoli da Twelve Data News (max 5)."""
    if not TWELVE_DATA_API_KEY:
        return []
    url = "https://api.twelvedata.com/news"
    params = {
        "symbol":   ticker,
        "apikey":   TWELVE_DATA_API_KEY,
        "outputsize": 5,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Twelve Data restituisce {"data": [{"title": ...}, ...]}
    items = data.get("data", [])
    return [f"[TwelveData] {item['title']}" for item in items if "title" in item]


def search_news(ticker: str) -> dict:
    """
    Recupera le ultime notizie per il ticker combinando:
    - Alpaca News Feed (ultimi 2 giorni)
    - Twelve Data News
    Ritorna {"headlines": [str, ...]} oppure {"error": str}.
    """
    headlines: list[str] = []
    errors: list[str] = []

    try:
        headlines += _fetch_alpaca_news(ticker)
    except Exception as e:
        errors.append(f"Alpaca: {e}")

    try:
        headlines += _fetch_twelve_data_news(ticker)
    except Exception as e:
        errors.append(f"TwelveData: {e}")

    if not headlines:
        if errors:
            return {"error": "; ".join(errors)}
        return {"headlines": ["Nessuna notizia recente trovata."]}

    return {"headlines": headlines}


# ---------------------------------------------------------------------------
# Tool 3: place_order
# ---------------------------------------------------------------------------

def place_order(ticker: str, side: str, quantity: float) -> dict:
    """
    Invia un ordine di mercato su Alpaca Paper Trading.
    Supporta sia azioni (qty intera) che crypto (qty frazionaria).
    side: "buy" | "sell"
    Ritorna {"order_id": str, "status": str} oppure {"error": str}.
    """
    try:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        # Crypto usa GTC (mercato sempre aperto); azioni usano DAY
        tif = TimeInForce.GTC if _is_crypto(ticker) else TimeInForce.DAY
        req = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=order_side,
            time_in_force=tif,
        )
        order = trading_client.submit_order(req)
        return {"order_id": str(order.id), "status": str(order.status)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 4: get_portfolio (utile per risk management Level 3)
# ---------------------------------------------------------------------------

def get_portfolio() -> dict:
    """
    Restituisce il saldo del conto e le posizioni aperte.
    """
    try:
        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        pos_list = [
            {"ticker": p.symbol, "qty": float(p.qty), "market_value": float(p.market_value)}
            for p in positions
        ]
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "positions": pos_list,
        }
    except Exception as e:
        return {"error": str(e)}