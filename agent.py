"""
Grafo LangGraph del Trading Agent.
Ciclo: fetch_market_data → fetch_news → reason → execute_order → journal → [loop o stop]
"""

import json
from langgraph.graph import StateGraph, END

from state import AgentState
from tools import get_price, search_news, place_order, get_portfolio
from journal import log_decision, print_journal

from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# Nodo 1: fetch_market_data
# ---------------------------------------------------------------------------

def fetch_market_data(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    print(f"\n[fetch_market_data] Recupero prezzo per {ticker}...")
    result = get_price(ticker)
    if "error" in result:
        print(f"  ⚠️  Errore: {result['error']}")
        return {**state, "price": None, "price_error": result["error"]}
    print(f"  ✅ Prezzo: {result['price']}")
    return {**state, "price": result["price"], "price_error": None}


# ---------------------------------------------------------------------------
# Nodo 2: fetch_news
# ---------------------------------------------------------------------------

def fetch_news(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    print(f"\n[fetch_news] Recupero notizie per {ticker}...")
    result = search_news(ticker)
    if "error" in result:
        print(f"  ⚠️  Errore news: {result['error']}")
        return {**state, "news_summary": "Notizie non disponibili.", "news_error": result["error"]}
    headlines = result["headlines"]
    summary = "\n".join(headlines)
    print(f"  ✅ {len(headlines)} notizie trovate.")
    return {**state, "news_summary": summary, "news_error": None}


# ---------------------------------------------------------------------------
# Nodo 3: reason (LLM)
# ---------------------------------------------------------------------------

REASON_PROMPT = """Sei un agente di trading autonomo. Devi analizzare i dati di mercato
e prendere UNA decisione: BUY, SELL, o HOLD.

Regola fondamentale: NON inventare dati. Usa SOLO le informazioni qui sotto.
Se i dati sono insufficienti o mancanti, decidi HOLD.

--- DATI DI MERCATO ---
Ticker: {ticker}
Prezzo attuale: {price}
Errore prezzo: {price_error}

--- NOTIZIE RECENTI ---
{news_summary}
Errore notizie: {news_error}

--- PORTFOLIO ---
{portfolio}

--- ISTRUZIONI ---
Rispondi SOLO con un JSON valido, nessun testo aggiuntivo, nessun markdown:
{{
  "decision": "BUY" | "SELL" | "HOLD",
  "quantity": <intero, 0 se HOLD>,
  "rationale": "<spiegazione in 2-4 frasi, citando i dati usati>"
}}

Regole per quantity:
- BUY: massimo il 10% del cash disponibile diviso per il prezzo (risk management base)
- SELL: vendi al massimo le quote già in portafoglio
- HOLD: quantity = 0
- Se price è None, decidi sempre HOLD
"""

def reason(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    print(f"\n[reason] LLM sta ragionando su {ticker}...")

    # Recupera portfolio (info deterministica, non dal modello)
    portfolio_data = get_portfolio()
    portfolio_str = json.dumps(portfolio_data, indent=2)

    prompt = REASON_PROMPT.format(
        ticker=ticker,
        price=state.get("price"),
        price_error=state.get("price_error") or "nessuno",
        news_summary=state.get("news_summary") or "nessuna",
        news_error=state.get("news_error") or "nessuno",
        portfolio=portfolio_str,
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Pulizia del JSON (rimuove eventuali backtick markdown)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        decision = parsed.get("decision", "HOLD").upper()
        quantity = int(parsed.get("quantity", 0))
        rationale = parsed.get("rationale", "Nessuna spiegazione fornita.")
    except Exception as e:
        print(f"  ⚠️  Errore parsing JSON: {e} — raw: {raw[:200]}")
        decision = "HOLD"
        quantity = 0
        rationale = f"Errore nel parsing della risposta LLM: {e}"

    print(f"  🤖 Decisione: {decision} x{quantity}")
    print(f"  📝 Rationale: {rationale}")
    return {**state, "decision": decision, "quantity": quantity, "rationale": rationale}


# ---------------------------------------------------------------------------
# Nodo 4: execute_order
# ---------------------------------------------------------------------------

def execute_order(state: AgentState) -> AgentState:
    decision = state.get("decision", "HOLD")
    ticker   = state["ticker"]
    quantity = state.get("quantity", 0)

    if decision == "HOLD" or quantity == 0:
        print(f"\n[execute_order] Decisione HOLD — nessun ordine inviato.")
        return {**state, "order_id": None, "order_error": None}

    print(f"\n[execute_order] Invio ordine {decision} {quantity}x {ticker}...")
    result = place_order(ticker, decision.lower(), quantity)
    if "error" in result:
        print(f"  ⚠️  Errore ordine: {result['error']}")
        return {**state, "order_id": None, "order_error": result["error"]}

    print(f"  ✅ Ordine eseguito: {result['order_id']} — status: {result['status']}")
    return {**state, "order_id": result["order_id"], "order_error": None}


# ---------------------------------------------------------------------------
# Nodo 5: write_journal
# ---------------------------------------------------------------------------

def write_journal(state: AgentState) -> AgentState:
    print(f"\n[write_journal] Registrazione nel journal...")

    outcome = "ok"
    if state.get("price_error"):
        outcome = f"price_error: {state['price_error']}"
    elif state.get("order_error"):
        outcome = f"order_error: {state['order_error']}"

    log_decision(
        ticker=state["ticker"],
        price=state.get("price"),
        decision=state.get("decision", "HOLD"),
        quantity=state.get("quantity", 0),
        rationale=state.get("rationale"),
        order_id=state.get("order_id"),
        outcome=outcome,
    )
    print_journal()

    # Incrementa il contatore cicli
    new_count = state["cycle_count"] + 1
    return {**state, "cycle_count": new_count}


# ---------------------------------------------------------------------------
# Edge condizionale: continua il loop o termina
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    if state["cycle_count"] >= state["max_cycles"]:
        print(f"\n[loop] Raggiunti {state['max_cycles']} cicli — stop.")
        return "end"
    print(f"\n[loop] Ciclo {state['cycle_count']}/{state['max_cycles']} — continua...")
    return "continue"


# ---------------------------------------------------------------------------
# Costruzione del grafo
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_market_data", fetch_market_data)
    graph.add_node("fetch_news",        fetch_news)
    graph.add_node("reason",            reason)
    graph.add_node("execute_order",     execute_order)
    graph.add_node("write_journal",     write_journal)

    # Sequenza principale
    graph.set_entry_point("fetch_market_data")
    graph.add_edge("fetch_market_data", "fetch_news")
    graph.add_edge("fetch_news",        "reason")
    graph.add_edge("reason",            "execute_order")
    graph.add_edge("execute_order",     "write_journal")

    # Loop condizionale
    graph.add_conditional_edges(
        "write_journal",
        should_continue,
        {
            "continue": "fetch_market_data",
            "end":      END,
        }
    )

    return graph.compile()