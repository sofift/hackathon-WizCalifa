from typing import TypedDict, Optional, List


class AgentState(TypedDict):
    """
    Stato condiviso tra tutti i nodi del grafo LangGraph.
    Ogni campo viene aggiornato progressivamente durante il ciclo.
    """
    # Input del ciclo e campi per il mini-ciclo di scelta autonoma
    ticker: Optional[str]
    candidate_ticker: Optional[str]
    candidate_news: Optional[str]
    search_attempts: int

    # Blacklist accumulata per evitare duplicati nello stesso ciclo globale
    blacklist_tickers: List[str]

    # Ticker già analizzati nella sessione corrente (per evitare ri-analisi inutili)
    # NON include i ticker in portafoglio — quelli possono sempre essere ri-valutati per incremento
    session_analyzed: List[str]

    # Snapshot live del portafoglio Alpaca (aggiornato da main.py prima di ogni ciclo)
    portfolio_snapshot: dict

    # Output di fetch_market_data
    price: Optional[float]
    price_error: Optional[str]

    # Output di fetch_news
    news_summary: Optional[str]
    news_error: Optional[str]

    # Output di reason (LLM) — strategia News Sentiment + Price Confirmation
    overall_sentiment: Optional[str]   # "RIALZISTA (ALTA)" | "RIALZISTA (BASSA)" | "RIBASSISTA" | "NEUTRO"
    decision: Optional[str]            # "BUY" | "SELL" | "HOLD"
    quantity: Optional[float]
    rationale: Optional[str]           # rationale arricchito con sentiment analysis

    # Output di execute_order
    order_id: Optional[str]
    order_error: Optional[str]

    # Flag di controllo del loop
    cycle_count: int
    max_cycles: int

    # Memoria: lista dei ticker già valutati o posseduti da non riproporre
    blacklist_tickers: list[str]