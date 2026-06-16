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
from alpaca.data.requests import StockLatestQuoteRequest


# ---------------------------------------------------------------------------
# Client Alpaca (inizializzati una volta sola al momento dell'import)
# ---------------------------------------------------------------------------

ALPACA_API_KEY    = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

trading_client = TradingClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=True,         # Paper Trading — nessun soldo reale
)

data_client = StockHistoricalDataClient(
    api_key=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
)


# ---------------------------------------------------------------------------
# Tool 1: get_price
# ---------------------------------------------------------------------------

def get_price(ticker: str) -> dict:
    """
    Restituisce il prezzo ask più recente per il ticker.
    Ritorna {"price": float, "ticker": str} oppure {"error": str}.
    """
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quote = data_client.get_stock_latest_quote(req)
        ask_price = quote[ticker].ask_price
        if ask_price is None or ask_price == 0:
            return {"error": f"Prezzo non disponibile per {ticker}"}
        return {"ticker": ticker, "price": float(ask_price)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 2: search_news
# ---------------------------------------------------------------------------

def search_news(ticker: str) -> dict:
    """
    Recupera le ultime notizie per il ticker tramite Alpaca News Feed.
    Ritorna {"headlines": [str, ...]} oppure {"error": str}.
    """
    try:
        # Alpaca News API v1beta1
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": ticker,
            "limit": 5,
            "start": (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        news_items = resp.json().get("news", [])
        if not news_items:
            return {"headlines": ["Nessuna notizia recente trovata."]}
        headlines = [f"• {item['headline']}" for item in news_items]
        return {"headlines": headlines}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 3: place_order
# ---------------------------------------------------------------------------

def place_order(ticker: str, side: str, quantity: int) -> dict:
    """
    Invia un ordine di mercato su Alpaca Paper Trading.
    side: "buy" | "sell"
    Ritorna {"order_id": str, "status": str} oppure {"error": str}.
    """
    try:
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY,
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