"""
Grafo LangGraph del Trading Agent.
Strategia: News Sentiment + Price Confirmation
Ciclo: fetch_market_data → fetch_news → reason → execute_order → journal → [loop o stop]
"""

import os
import json
from langgraph.graph import StateGraph, END

from state import AgentState
from tools import get_price, search_news, place_order, get_portfolio
from journal import log_decision, print_journal

from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
)

# ---------------------------------------------------------------------------
# Nodi Iniziali: Scelta Autonoma e Ponderata (Mini-Ciclo)
# ---------------------------------------------------------------------------

PROPOSE_PROMPT = """Sei un agente di trading esplorativo. Il tuo compito è suggerire un singolo ticker (azione USA, ETF, o Crypto come BTC/USD, ETH/USD, SPY, QQQ, TSLA, AAPL, NVDA, ecc.) che potrebbe essere interessante analizzare oggi.
Scegli un asset diverso se possibile.
Rispondi SOLO con il simbolo del ticker, nient'altro."""

def propose_candidate(state: AgentState) -> AgentState:
    attempts = state.get("search_attempts", 0) + 1
    print(f"\n[propose_candidate] L'agente cerca un asset interessante (Tentativo {attempts}/3)...")
    response = llm.invoke(PROPOSE_PROMPT)
    scelta = response.content.strip().upper()
    print(f"  🤔 Ticker proposto: {scelta}")
    return {**state, "candidate_ticker": scelta, "search_attempts": attempts}

def fetch_candidate_news(state: AgentState) -> AgentState:
    ticker = state.get("candidate_ticker")
    print(f"[fetch_candidate_news] Controllo le notizie per {ticker}...")
    result = search_news(ticker)
    if "error" in result:
        return {**state, "candidate_news": "Errore nel recupero notizie."}
    summary = "\n".join(result["headlines"])
    return {**state, "candidate_news": summary}

EVALUATE_PROMPT = """Valuta le seguenti notizie recenti per il ticker {ticker}.
Notizie:
{news}

Ci sono spunti chiari (positivi o negativi) che giustifichino un'operazione di trading?
Se le notizie sono interessanti e mostrano un trend (positivo o negativo), rispondi esattamente con "APPROVATO".
Se le notizie sono piatte, di routine, irrilevanti o assenti, rispondi esattamente con "RIFIUTATO".
Non aggiungere altre parole.
"""

def evaluate_candidate(state: AgentState) -> AgentState:
    ticker = state.get("candidate_ticker")
    news = state.get("candidate_news", "")
    print(f"[evaluate_candidate] L'agente valuta se le news di {ticker} sono interessanti...")
    
    prompt = EVALUATE_PROMPT.format(ticker=ticker, news=news)
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    
    if "APPROVATO" in decision:
        print(f"  ✅ {ticker} APPROVATO! Diventa l'asset ufficiale del ciclo.")
        return {**state, "ticker": ticker}
    else:
        print(f"  ❌ {ticker} RIFIUTATO (notizie non interessanti).")
        if state.get("search_attempts", 0) >= 3:
            print(f"  ⚠️ Raggiunto limite di 3 tentativi. Forzo {ticker} come asset ufficiale per evitare loop infiniti.")
            return {**state, "ticker": ticker}
        return {**state, "ticker": None}

def route_candidate(state: AgentState) -> str:
    # Se il ticker è stato approvato (quindi valorizzato), procediamo con i dati di mercato
    if state.get("ticker"):
        return "fetch_market_data"
    # Altrimenti cerchiamo un altro candidato
    return "propose_candidate"

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
# Nodo 3: reason (LLM) — strategia News Sentiment + Price Confirmation
# ---------------------------------------------------------------------------

REASON_PROMPT = """Sei un agente di trading autonomo che opera su un mercato simulato.
Il tuo processo decisionale segue due fasi obbligatorie e sequenziali:
FASE 1 → Analisi del sentiment delle notizie (classificazione deterministica)
FASE 2 → Conferma del prezzo + Decisione finale

═══════════════════════════════════════
FASE 1 — ANALISI DEL SENTIMENT
═══════════════════════════════════════
Classifica OGNI titolo di notizia qui sotto come POSITIVO, NEGATIVO o NEUTRO.
Regole di classificazione:
- POSITIVO: lancio di prodotti, utili sopra le attese, partnership, upgrade degli analisti, buyback
- NEGATIVO: cause legali, utili sotto le attese, licenziamenti, downgrade, ritiri di prodotto, problemi regolatori
- NEUTRO: annunci di routine, reiterazioni degli analisti, aggiornamenti minori
- Se non ci sono notizie disponibili → classifica come NEUTRO

Titoli delle notizie per {ticker}:
{news_summary}

Dopo aver classificato ogni titolo, calcola il SENTIMENT COMPLESSIVO:
- RIALZISTA → la maggioranza è POSITIVA (più positivi che negativi)
- RIBASSISTA → la maggioranza è NEGATIVA (più negativi che positivi)
- NEUTRO     → in parità oppure tutti neutri

═══════════════════════════════════════
FASE 2 — CONFERMA DEL PREZZO
═══════════════════════════════════════
Prezzo attuale: {price}
Errore prezzo: {price_error}
Stato del portafoglio:
{portfolio}

Applica queste regole di conferma nell'ordine indicato:
- Se sentiment RIALZISTA E prezzo disponibile E cash > 0 → considera BUY
- Se sentiment RIBASSISTA E hai già azioni di {ticker} in portafoglio → considera SELL
- Se sentiment NEUTRO → HOLD (non agire sull'incertezza)
- Se il prezzo è None o non disponibile → HOLD in ogni caso (non agire mai alla cieca)
- Se il cash è insufficiente per acquistare almeno 1 azione → HOLD

Regola di rischio: non allocare mai più del 10% del cash disponibile in un singolo ordine.
Calcolo della quantità in base al tipo di asset:
- Se {ticker} contiene "/" → è un asset CRYPTO: usa round(cash * 0.10 / prezzo, 6) — quantità DECIMALE (es. 0.0015).
- Altrimenti → è un'AZIONE: usa floor(cash * 0.10 / prezzo) — quantità INTERA (es. 3).
In ogni caso il valore minimo è la quantità minima acquistabile (per crypto può essere 0.0001).
Per SELL: vendi la quantità già detenuta in portafoglio per {ticker}, oppure HOLD se la posizione è 0.

═══════════════════════════════════════
REGOLE FONDAMENTALI
═══════════════════════════════════════
- NON inventare mai prezzi, notizie o dati di portafoglio. Usa SOLO le informazioni fornite sopra.
- Se i dati sono mancanti o contraddittori → vai su HOLD per default.
- La motivazione deve citare esplicitamente: la classificazione del sentiment, il prezzo usato e quale regola ha scatenato la decisione.

═══════════════════════════════════════
FORMATO DI OUTPUT
═══════════════════════════════════════
Rispondi SOLO con JSON valido. Niente markdown, niente backtick, niente testo aggiuntivo.

{{
  "analisi_sentiment": [
    {{"titolo": "<testo esatto del titolo>", "classificazione": "POSITIVO|NEGATIVO|NEUTRO", "motivazione": "<una frase che spiega il perché>"}}
  ],
  "sentiment_complessivo": "RIALZISTA|RIBASSISTA|NEUTRO",
  "conferma_prezzo": "<una frase: il prezzo conferma o contraddice il sentiment?>",
  "decisione": "BUY|SELL|HOLD",
  "quantita": <numero, 0 se HOLD — decimale per crypto, intero per azioni>,
  "motivazione_finale": "<2-4 frasi che citano punteggio sentiment, prezzo, contesto portafoglio e quale regola ha scattato>"
}}
"""


def _clean_json(raw: str) -> str:
    """Rimuove eventuali backtick markdown attorno al JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] contiene il blocco interno (es. "json\n{...}")
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def reason(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    print(f"\n[reason] LLM sta ragionando su {ticker} (strategia: News Sentiment + Price Confirmation)...")

    # Recupera portfolio deterministicamente — mai dall'LLM
    portfolio_data = get_portfolio()
    portfolio_str = json.dumps(portfolio_data, indent=2)

    prompt = REASON_PROMPT.format(
        ticker=ticker,
        price=state.get("price"),
        price_error=state.get("price_error") or "none",
        news_summary=state.get("news_summary") or "Nessun titolo di notizia disponibile.",
        portfolio=portfolio_str,
    )

    response = llm.invoke(prompt)
    raw = _clean_json(response.content)

    try:
        parsed = json.loads(raw)

        # Campi principali
        decision         = parsed.get("decisione", "HOLD").upper()
        quantity         = round(float(parsed.get("quantita", 0)), 6)
        rationale        = parsed.get("motivazione_finale", "Nessuna motivazione fornita.")

        # Campi della strategia sentiment
        overall_sentiment  = parsed.get("sentiment_complessivo", "NEUTRO")
        price_confirmation = parsed.get("conferma_prezzo", "")
        sentiment_analysis = parsed.get("analisi_sentiment", [])

        # Stampa dettagliata in console per debug e demo
        print(f"\n  📰 Analisi sentiment:")
        for item in sentiment_analysis:
            icon = "✅" if item.get("classificazione") == "POSITIVO" else \
                   "❌" if item.get("classificazione") == "NEGATIVO" else "➖"
            print(f"     {icon} [{item.get('classificazione')}] {item.get('titolo', '')[:80]}")
            print(f"        → {item.get('motivazione', '')}")

        print(f"\n  📊 Sentiment complessivo : {overall_sentiment}")
        print(f"  💰 Conferma prezzo       : {price_confirmation}")
        print(f"  🤖 Decisione             : {decision} x{quantity}")
        print(f"  📝 Motivazione finale    : {rationale}")

        # Costruisce una stringa rationale arricchita per il journal
        sentiment_lines = "\n".join(
            f"  [{i.get('classificazione')}] {i.get('titolo', '')[:60]} — {i.get('motivazione', '')}"
            for i in sentiment_analysis
        )
        full_rationale = (
            f"SENTIMENT: {overall_sentiment}\n"
            f"{sentiment_lines}\n"
            f"CONFERMA PREZZO: {price_confirmation}\n"
            f"DECISIONE: {rationale}"
        )

    except Exception as e:
        print(f"  ⚠️  Errore parsing JSON: {e}")
        print(f"  Risposta grezza (primi 300 char): {raw[:300]}")
        decision          = "HOLD"
        quantity          = 0
        overall_sentiment = "NEUTRO"
        full_rationale    = f"Errore nel parsing JSON — HOLD per default. Errore: {e}"

    return {
        **state,
        "decision":          decision,
        "quantity":          quantity,
        "rationale":         full_rationale,
        "overall_sentiment": overall_sentiment,
    }


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

    # Determina outcome
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

    new_count = state["cycle_count"] + 1
    # Resettiamo i parametri di ricerca per il ciclo successivo
    return {
        **state, 
        "cycle_count": new_count, 
        "search_attempts": 0, 
        "ticker": None, 
        "candidate_ticker": None
    }


# ---------------------------------------------------------------------------
# Edge condizionale: continua il loop o termina
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> str:
    if state["cycle_count"] >= state["max_cycles"]:
        print(f"\n[loop] Raggiunti {state['max_cycles']} cicli — stop.")
        return "end"
    print(f"\n[loop] Ciclo {state['cycle_count']}/{state['max_cycles']} — continua tra poco...")
    return "continue"


# ---------------------------------------------------------------------------
# Costruzione del grafo
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodi di ricerca autonoma
    graph.add_node("propose_candidate",     propose_candidate)
    graph.add_node("fetch_candidate_news",  fetch_candidate_news)
    graph.add_node("evaluate_candidate",    evaluate_candidate)

    # Nodi standard
    graph.add_node("fetch_market_data", fetch_market_data)
    graph.add_node("fetch_news",        fetch_news)
    graph.add_node("reason",            reason)
    graph.add_node("execute_order",     execute_order)
    graph.add_node("write_journal",     write_journal)

    # Entry point è la ricerca
    graph.set_entry_point("propose_candidate")
    
    # Ciclo di ricerca
    graph.add_edge("propose_candidate", "fetch_candidate_news")
    graph.add_edge("fetch_candidate_news", "evaluate_candidate")
    graph.add_conditional_edges("evaluate_candidate", route_candidate, {
        "fetch_market_data": "fetch_market_data",
        "propose_candidate": "propose_candidate"
    })

    # Sequenza principale (post-ricerca)
    graph.add_edge("fetch_market_data", "fetch_news")
    graph.add_edge("fetch_news",        "reason")
    graph.add_edge("reason",            "execute_order")
    graph.add_edge("execute_order",     "write_journal")

    # Loop condizionale finale: riparte dalla ricerca autonoma o si ferma
    graph.add_conditional_edges(
        "write_journal",
        should_continue,
        {
            "continue": "propose_candidate",
            "end":      END,
        }
    )

    return graph.compile()