"""
user_clients.py
Gestione client Alpaca per-utente (multi-tenant).

Ogni membro del gruppo Telegram è mappato alle proprie credenziali Alpaca
tramite variabili d'ambiente nel .env:
    ALPACA_KEY_{chat_id}    = chiave API Alpaca
    ALPACA_SECRET_{chat_id} = chiave segreta Alpaca

Se le credenziali per un utente non sono configurate, si usa il fallback
alle credenziali DEFAULT (ALPACA_API_KEY / ALPACA_SECRET_KEY).
"""

import os
from alpaca.trading.client import TradingClient


# Cache dei client per evitare di ricreare ad ogni chiamata
_client_cache: dict[int, TradingClient] = {}


def get_trading_client(chat_id: int | None = None) -> TradingClient:
    """
    Restituisce il TradingClient Alpaca appropriato:
    - Se chat_id è fornito, usa le credenziali per-utente (ALPACA_KEY_{chat_id})
    - Se chat_id è None, usa il client DEFAULT (per l'agent loop autonomo)

    Le credenziali vengono cercate nell'ordine:
    1. ALPACA_KEY_{chat_id} / ALPACA_SECRET_{chat_id}  (per-utente)
    2. ALPACA_API_KEY / ALPACA_SECRET_KEY               (fallback default)
    """
    if chat_id is None:
        # Usa il client default (importato da tools.py per evitare duplicazione)
        from tools import trading_client
        return trading_client

    if chat_id in _client_cache:
        return _client_cache[chat_id]

    # Cerca credenziali per-utente
    api_key = os.environ.get(f"ALPACA_KEY_{chat_id}")
    secret_key = os.environ.get(f"ALPACA_SECRET_{chat_id}")

    if api_key and secret_key:
        print(f"[user_clients] ✅ Client Alpaca creato per chat_id={chat_id} (credenziali personali)")
    else:
        # Fallback alle credenziali default
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        print(f"[user_clients] ⚠️  chat_id={chat_id}: credenziali personali non trovate, uso DEFAULT")

    client = TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
    )
    _client_cache[chat_id] = client
    return client


def clear_cache() -> None:
    """Svuota la cache dei client (utile per test o reload credenziali)."""
    _client_cache.clear()
