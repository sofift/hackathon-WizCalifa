from typing import TypedDict, Optional


class AgentState(TypedDict):
    """
    Stato condiviso tra tutti i nodi del grafo LangGraph.
    Ogni campo viene aggiornato progressivamente durante il ciclo.
    """
    # Input del ciclo
    ticker: str

    # Output di fetch_market_data
    price: Optional[float]
    price_error: Optional[str]

    # Output di fetch_news
    news_summary: Optional[str]
    news_error: Optional[str]

    # Output di reason (LLM)
    decision: Optional[str]       # "BUY" | "SELL" | "HOLD"
    quantity: Optional[int]       # numero di azioni (0 se HOLD)
    rationale: Optional[str]      # spiegazione in linguaggio naturale

    # Output di execute_order
    order_id: Optional[str]
    order_error: Optional[str]

    # Flag di controllo del loop
    cycle_count: int
    max_cycles: int