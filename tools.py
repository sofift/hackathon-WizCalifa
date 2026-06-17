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

from dotenv import load_dotenv
load_dotenv()

ALPACA_API_KEY      = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY   = os.environ.get("ALPACA_SECRET_KEY", "")
FINNHUB_API_KEY     = os.environ.get("FINNHUB_API_KEY", "")
POLYGON_API_KEY     = os.environ.get("POLYGON_API_KEY", "")

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
    """Restituisce True se il ticker è un asset crypto (es. BTC/USD, ETH/USD o BTCUSD)."""
    return "/" in ticker or ticker.endswith("USD")


# ---------------------------------------------------------------------------
# Tool 1: get_price
# ---------------------------------------------------------------------------

def get_price(ticker: str) -> dict:
    """
    Restituisce il prezzo più recente per il ticker.
    Supporta sia azioni/ETF (es. SPY, AAPL) che crypto (es. BTC/USD).
    Ritorna {"price": float, "ticker": str} oppure {"error": str}.
    """
    try:
        if _is_crypto(ticker):
            req = CryptoLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = crypto_data_client.get_crypto_latest_quote(req)
            price = quote[ticker].ask_price
        else:
            # Per gli ETF/Azioni usiamo l'ultimo scambio (Trade) invece della quotazione (Quote ask),
            # in questo modo il prezzo è sempre disponibile anche a mercati chiusi o nel weekend!
            from alpaca.data.requests import StockLatestTradeRequest
            req = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trade = stock_data_client.get_stock_latest_trade(req)
            price = trade[ticker].price

        if price is None or price == 0:
            return {"error": f"Prezzo non disponibile per {ticker}"}
        return {"ticker": ticker, "price": float(price)}
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


def _fetch_finnhub_news(ticker: str) -> list[str]:
    """Ritorna titoli da Finnhub News (max 5, ultimi 3 giorni)."""
    if not FINNHUB_API_KEY:
        return []
    # Finnhub non supporta crypto nel company-news endpoint, o se lo fa usa un formato diverso
    if _is_crypto(ticker):
        return []
        
    url = "https://finnhub.io/api/v1/company-news"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    params = {
        "symbol": ticker,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY,
    }
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                return []
            return [f"[Finnhub] {item['headline']}" for item in data[:5] if "headline" in item]
        except Exception as e:
            if attempt == 1:
                print(f"  [fetch_news] Errore Finnhub per {ticker} dopo 2 tentativi: {e}")
            else:
                import time
                time.sleep(1)
    return []


def _fetch_polygon_news(ticker: str) -> list[str]:
    """Ritorna titoli da Polygon.io News (max 5)."""
    if not POLYGON_API_KEY:
        return []
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": 5,
        "apiKey": POLYGON_API_KEY,
    }
    response = requests.get(url, params=params, timeout=10)
    if response.status_code == 429:
        return [] # Ignora graziosamente il rate limit (5 calls/min sul piano free)
    response.raise_for_status()
    data = response.json()
    items = data.get("results", [])
    return [f"[Polygon] {item['title']}" for item in items if "title" in item]


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
        headlines += _fetch_finnhub_news(ticker)
    except Exception as e:
        errors.append(f"Finnhub: {e}")

    try:
        headlines += _fetch_polygon_news(ticker)
    except Exception as e:
        errors.append(f"Polygon: {e}")

    if not headlines:
        if errors:
            return {"error": "; ".join(errors)}
        return {"headlines": ["Nessuna notizia recente trovata."]}

    return {"headlines": headlines}


# ---------------------------------------------------------------------------
# Tool 3: place_order
# ---------------------------------------------------------------------------

def place_order(ticker: str, side: str, quantity: float, user_chat_id: int | None = None) -> dict:
    """
    Invia un ordine di mercato su Alpaca Paper Trading.
    Supporta sia azioni (qty intera) che crypto (qty frazionaria).
    side: "buy" | "sell"
    user_chat_id: se fornito, usa le credenziali Alpaca dell'utente (multi-tenant).
    Ritorna {"order_id": str, "status": str} oppure {"error": str}.
    """
    try:
        # Scegli il client appropriato (per-utente o default)
        if user_chat_id is not None:
            from user_clients import get_trading_client
            client = get_trading_client(user_chat_id)
        else:
            client = trading_client

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        # Crypto usa GTC (mercato sempre aperto); azioni usano DAY
        tif = TimeInForce.GTC if _is_crypto(ticker) else TimeInForce.DAY
        req = MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=order_side,
            time_in_force=tif,
        )
        order = client.submit_order(req)
        return {"order_id": str(order.id), "status": str(order.status)}
    except Exception as e:
        err_str = str(e)
        clean_err = err_str
        
        import json
        start_idx = err_str.find("{")
        if start_idx != -1:
            try:
                err_json = json.loads(err_str[start_idx:])
                if "message" in err_json:
                    clean_err = err_json["message"]
            except Exception:
                pass

        if "potential wash trade detected" in err_str or "opposite side" in err_str:
            try:
                if start_idx != -1:
                    err_json = json.loads(err_str[start_idx:])
                    if "existing_order_id" in err_json:
                        order_id_to_cancel = err_json["existing_order_id"]
                        client.cancel_order_by_id(order_id_to_cancel)
                        # Ritenta l'ordine dopo aver cancellato quello opposto
                        order = client.submit_order(req)
                        return {"order_id": str(order.id), "status": str(order.status) + " (ordine opposto forzatamente cancellato)"}
            except Exception as retry_err:
                return {"error": f"Wash trade detectato. Impossibile annullare l'ordine pendente: {retry_err}"}
            return {"error": "Wash trade detectato (ordine opposto pendente e non cancellabile)."}
            
        if "insufficient qty available" in err_str and "held_for_orders" in err_str:
            return {"error": "Ordine già in coda (mercato chiuso o pending)."}
            
        return {"error": clean_err}


# ---------------------------------------------------------------------------
# Tool 4: get_portfolio (utile per risk management Level 3)
# ---------------------------------------------------------------------------

def get_portfolio(user_chat_id: int | None = None) -> dict:
    """
    Restituisce il saldo del conto e le posizioni aperte.
    user_chat_id: se fornito, usa le credenziali Alpaca dell'utente (multi-tenant).
    """
    try:
        # Scegli il client appropriato (per-utente o default)
        if user_chat_id is not None:
            from user_clients import get_trading_client
            client = get_trading_client(user_chat_id)
        else:
            client = trading_client

        account = client.get_account()
        positions = client.get_all_positions()
        pos_list = [
            {
                "ticker": p.symbol, 
                "qty": float(p.qty), 
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "profit_pct": float(p.unrealized_plpc) * 100  # Convertiamo in percentuale (es. 0.015 -> 1.5%)
            }
            for p in positions
        ]
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "positions": pos_list,
        }
    except Exception as e:
        return {"error": str(e)}





# ---------------------------------------------------------------------------
# Tool 5: get_position_qty
# ---------------------------------------------------------------------------

def get_position_qty(portfolio: dict, ticker: str) -> float:
    """
    Restituisce la quantità detenuta per un dato ticker nel portfolio snapshot.
    Ritorna 0.0 se il ticker non è in portafoglio.
    """
    for pos in portfolio.get("positions", []):
        if pos["ticker"].upper() == ticker.upper():
            return float(pos["qty"])
    return 0.0


# ---------------------------------------------------------------------------
# Tool 6: get_market_news (feed generale di mercato — nessun ticker specifico)
# ---------------------------------------------------------------------------

def get_market_news(limit: int = 50) -> dict:
    """
    Recupera le notizie generali di mercato (senza filtrare per ticker).
    Ritorna {"articles": [...], "symbol_counts": {ticker: int}} oppure {"error": str}.
    """
    try:
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "limit": limit,
            "start": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        news_items = resp.json().get("news", [])

        articles = []
        symbol_counts: dict[str, int] = {}

        for item in news_items:
            symbols = item.get("symbols", [])
            if not symbols:
                continue
            article = {
                "headline":   item.get("headline", ""),
                "created_at": item.get("created_at", ""),
                "symbols":    [s.upper() for s in symbols],
            }
            articles.append(article)
            for sym in symbols:
                sym = sym.upper()
                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

        return {"articles": articles, "symbol_counts": symbol_counts}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 7: get_open_orders — ordini inviati ma NON ancora eseguiti
# ---------------------------------------------------------------------------
# Distingue le posizioni reali (filled) dagli ordini "sospesi/pending":
# tipicamente ordini inviati a mercato chiuso, o in coda, o trattenuti.
# Usato dal report Telegram per dire "ordine in attesa perche' mercato chiuso".
# ---------------------------------------------------------------------------

_OPEN_ORDER_STATUSES = {
    "new", "accepted", "pending_new", "accepted_for_bidding",
    "held", "partially_filled", "pending_replace", "replaced",
    "calculated", "stopped", "suspended",
}


def get_open_orders(user_chat_id: int | None = None) -> dict:
    """
    Restituisce gli ordini attualmente aperti (inviati ma non ancora eseguiti).
    user_chat_id: se fornito, usa le credenziali Alpaca dell'utente (multi-tenant).
    Ritorna {"orders": [ {...}, ... ]} oppure {"error": str}.

    Ogni ordine: ticker, side, qty, filled_qty, status, created_at,
                 market_closed (bool euristico: ordine azionario DAY ancora aperto).
    """
    try:
        if user_chat_id is not None:
            from user_clients import get_trading_client
            client = get_trading_client(user_chat_id)
        else:
            client = trading_client
            
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50)
        orders = client.get_orders(filter=req)

        out = []
        for o in orders:
            status = str(getattr(o, "status", "")).lower().split(".")[-1]
            if status not in _OPEN_ORDER_STATUSES:
                continue
            symbol = getattr(o, "symbol", "?")
            is_crypto_sym = _is_crypto(symbol)
            filled = float(getattr(o, "filled_qty", 0) or 0)
            # Euristica: ordine azionario non eseguito e non parziale => mercato chiuso
            market_closed = (
                not is_crypto_sym
                and status in ("new", "accepted", "pending_new", "accepted_for_bidding", "held")
                and filled == 0
            )
            out.append({
                "ticker":        symbol,
                "side":          str(getattr(o, "side", "")).lower().split(".")[-1],
                "qty":           float(getattr(o, "qty", 0) or 0),
                "filled_qty":    filled,
                "status":        status,
                "created_at":    str(getattr(o, "created_at", "")),
                "market_closed": market_closed,
            })
        return {"orders": out}
    except Exception as e:
        return {"error": str(e)}