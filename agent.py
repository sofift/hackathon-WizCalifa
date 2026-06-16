"""
Grafo LangGraph del Trading Agent.
Strategia: News Sentiment + Price Confirmation
Ciclo: propose_candidate → [evaluate] → fetch_market_data → fetch_news → reason → execute_order → journal → [loop o stop]
"""

import os
import math
import json
from datetime import time as dt_time
from langgraph.graph import StateGraph, END

from state import AgentState
from tools import get_price, search_news, place_order, get_portfolio
from journal import log_decision, print_journal

from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# LLM — due modelli: uno veloce per lo scouting, uno potente per il reasoning
# ---------------------------------------------------------------------------

# Usato per propose_candidate ed evaluate_candidate (compiti semplici e veloci)
llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
)

# Usato per reason (compito complesso: sentiment + risk management + JSON strutturato)
llm_reason = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
)

# ---------------------------------------------------------------------------
# Helper: verifica se il mercato azionario USA è aperto
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    """
    Ritorna True se il mercato NYSE/NASDAQ è attualmente aperto
    (lunedì-venerdì, 09:30–16:00 America/New_York).
    Per le crypto il mercato è sempre aperto (ritorna sempre True se invocata con is_crypto=True).
    """
    try:
        import datetime
        import pytz
        ny = pytz.timezone("America/New_York")
        now = datetime.datetime.now(ny)
        if now.weekday() >= 5:  # 5=Sabato, 6=Domenica
            return False
        return dt_time(9, 30) <= now.time() <= dt_time(16, 0)
    except Exception:
        return True  # In caso di errore, non blocchiamo l'agente


# ---------------------------------------------------------------------------
# Nodi Iniziali: Scelta Autonoma e Ponderata (Mini-Ciclo)
# ---------------------------------------------------------------------------

PROPOSE_PROMPT = """Sei un Senior Quantitative Broker di Wall Street. Il tuo unico obiettivo è massimizzare il profitto (Alpha) gestendo il rischio.
Il tuo compito è suggerire un singolo ticker (azione USA, ETF, o Crypto come BTC/USD, ETH/USD, SPY, QQQ, TSLA, AAPL, NVDA, ecc.) che presenta OGGI un potenziale squilibrio o un momentum direzionale.
Considera il contesto macroeconomico: i tassi d'interesse favoriscono i finanziari? L'AI spinge i tech? Tensioni geopolitiche favoriscono l'oro o il petrolio?
Scegli un asset diverso dai soliti se intravvedi un'opportunità, oppure punta sui big se c'è un vero catalizzatore.

─── PORTAFOGLIO ATTUALE ───
Cash disponibile: ${cash:,.2f}
Posizioni aperte: {posizioni}

NOTA IMPORTANTE: Puoi suggerire un ticker GIÀ IN PORTAFOGLIO se hai ragione di credere che vi sia un catalizzatore forte che giustifichi un incremento della posizione (es. nuovi prodotti, utili sopra le attese, breakout tecnico). Non farlo per rumore di fondo.

Ticker già analizzati IN QUESTA SESSIONE (non riproporre perché già valutati oggi): {session_analyzed}

Rispondi SOLO con il simbolo del ticker, nient'altro."""

def propose_candidate(state: AgentState) -> AgentState:
    attempts = state.get("search_attempts", 0) + 1
    print(f"\n[propose_candidate] L'agente cerca un asset interessante (Tentativo {attempts}/3)...")

    # session_analyzed = ticker già analizzati oggi (no duplicati); NON include i ticker in portafoglio
    session_analyzed = state.get("session_analyzed", [])
    session_analyzed_str = ", ".join(session_analyzed) if session_analyzed else "nessuno"

    # Estrai info portafoglio dallo snapshot live
    snap = state.get("portfolio_snapshot", {})
    cash = snap.get("cash", 0)
    positions = snap.get("positions", [])
    if positions:
        posizioni_str = ", ".join(
            f"{p['ticker']} (qty={p['qty']}, valore=${p['market_value']:,.2f})"
            for p in positions
        )
    else:
        posizioni_str = "nessuna"

    prompt = PROPOSE_PROMPT.format(
        cash=cash,
        posizioni=posizioni_str,
        session_analyzed=session_analyzed_str,
    )
    response = llm_fast.invoke(prompt)
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

EVALUATE_PROMPT = """Sei un Senior Quantitative Broker. Valuta le seguenti notizie recenti per il ticker {ticker}.
Notizie:
{news}

{position_context}

Cerca SOLO VERI CATALIZZATORI direzionali (Earnings sorprendenti, sviluppi normativi, M&A, squilibri domanda/offerta, forti macro-trend).
Se le notizie contengono un vero catalizzatore che giustifica un'operazione per massimizzare il profitto, rispondi esattamente con "APPROVATO".
Se le notizie sono rumore di fondo, generiche (es. "nuovo sito web", "aggiornamenti di routine"), irrilevanti o assenti, rispondi esattamente con "RIFIUTATO" per non sprecare capitale.
Non aggiungere altre parole.
"""

def evaluate_candidate(state: AgentState) -> AgentState:
    ticker = state.get("candidate_ticker")
    news = state.get("candidate_news", "")
    print(f"[evaluate_candidate] L'agente valuta se le news di {ticker} sono interessanti...")

    # Verifica se il ticker è già in portafoglio e aggiunge contesto specifico
    snap = state.get("portfolio_snapshot", {})
    portfolio_tickers = {p["ticker"]: p for p in snap.get("positions", [])}
    if ticker in portfolio_tickers:
        pos = portfolio_tickers[ticker]
        position_context = (
            f"NOTA: Hai già {pos['qty']} unità di {ticker} in portafoglio "
            f"(valore attuale: ${pos['market_value']:,.2f}). "
            f"Rispondi APPROVATO SOLO se il catalizzatore è abbastanza forte da giustificare un INCREMENTO della posizione."
        )
    else:
        position_context = f"Non hai {ticker} in portafoglio."

    prompt = EVALUATE_PROMPT.format(ticker=ticker, news=news, position_context=position_context)
    response = llm_fast.invoke(prompt)
    decision = response.content.strip().upper()

    # Aggiorna session_analyzed
    session_analyzed = list(state.get("session_analyzed", []))
    if ticker not in session_analyzed:
        session_analyzed.append(ticker)

    if "APPROVATO" in decision:
        already_owned = ticker in portfolio_tickers
        if already_owned:
            print(f"  ✅ {ticker} APPROVATO per INCREMENTO posizione esistente!")
        else:
            print(f"  ✅ {ticker} APPROVATO! Diventa l'asset ufficiale del ciclo.")
        return {**state, "ticker": ticker, "session_analyzed": session_analyzed}
    else:
        print(f"  ❌ {ticker} RIFIUTATO (notizie non interessanti).")
        if state.get("search_attempts", 0) >= 3:
            print(f"  ⚠️ Raggiunto limite di 3 tentativi. Forzo {ticker} come asset ufficiale per evitare loop infiniti.")
            return {**state, "ticker": ticker, "session_analyzed": session_analyzed}
        return {**state, "ticker": None, "session_analyzed": session_analyzed}

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

REASON_PROMPT = """Sei un Senior Quantitative Broker. Il tuo obiettivo è massimizzare l'Alpha del portafoglio applicando ferree regole di Risk Management.
Il tuo processo decisionale segue due fasi sequenziali:
FASE 1 → Analisi del Sentiment e identificazione del Catalizzatore
FASE 2 → Conferma del prezzo + Risk Management (Position Sizing Dinamico e Cut Losses)

═══════════════════════════════════════
FASE 1 — ANALISI DEL SENTIMENT E CATALIZZATORI
═══════════════════════════════════════
Classifica OGNI titolo di notizia come POSITIVO, NEGATIVO o NEUTRO filtrando il rumore.
- POSITIVO: Sorprese positive agli utili, upgrade di massa degli analisti, M&A, approvazioni normative.
- NEGATIVO: Cause legali gravi, miss sugli utili, downgrade drastici, problemi regolatori.
- NEUTRO: Rumore di fondo, annunci di routine.
- Se non ci sono notizie → NEUTRO.

Titoli delle notizie per {ticker}:
{news_summary}

Calcola il SENTIMENT COMPLESSIVO e il livello di CONFIDENZA (ALTA o BASSA):
- RIALZISTA (ALTA Confidenza) → presenza di catalizzatori positivi reali e inequivocabili.
- RIALZISTA (BASSA Confidenza) → notizie moderatamente positive.
- RIBASSISTA → notizie negative rilevanti.
- NEUTRO → solo rumore di fondo o parità.

═══════════════════════════════════════
FASE 2 — RISK MANAGEMENT E POSITION SIZING
═══════════════════════════════════════
Prezzo attuale: {price}
Errore prezzo: {price_error}
Stato del portafoglio:
{portfolio}

Applica le seguenti regole di trading per massimizzare i profitti e tagliare le perdite:
1. CUT LOSSES (Taglia le perdite): Se il sentiment è RIBASSISTA e hai già azioni di {ticker} in portafoglio, devi VENDERE (SELL) immediatamente tutta la posizione per proteggere il capitale.
2. POSITION SIZING DINAMICO (BUY):
   - Se sentiment RIALZISTA (ALTA Confidenza) E prezzo disponibile → Alloca il 15% del cash disponibile (opportunità forte).
- Se sentiment RIALZISTA (BASSA Confidenza) E prezzo disponibile → Alloca solo il 5% del cash disponibile (esposizione prudente).
3. HOLD DISCIPLINATO: Se sentiment NEUTRO o prezzo non disponibile o cash insufficiente → HOLD (il capitale è protetto non facendo nulla).
4. PRESA DI PROFITTO (Take Profit): Se il sentiment è NEUTRO/RIBASSISTA ma hai posizioni in largo profitto (valuta tu dal portfolio), valuta di vendere.

Non devi calcolare la quantità esatta da comprare/vendere. Devi solo indicare l'allocazione percentuale desiderata:
- Se BUY: indica 0.15 (per 15%) o 0.05 (per 5%).
- Se SELL: indica 1.0 (vendi tutto).
- Se HOLD: indica 0.0.

═══════════════════════════════════════
REGOLE FONDAMENTALI
═══════════════════════════════════════
- NON inventare prezzi, notizie o dati di portafoglio.
- Se i dati mancano → HOLD.
- La motivazione finale deve citare la confidenza, il calcolo del position sizing (es. 5% o 15%) o se si sta applicando il "Cut Losses".

═══════════════════════════════════════
FORMATO DI OUTPUT
═══════════════════════════════════════
Rispondi SOLO con JSON valido. Niente markdown, niente testo aggiuntivo.

{{
  "analisi_sentiment": [
    {{"titolo": "<titolo>", "classificazione": "POSITIVO|NEGATIVO|NEUTRO", "motivazione": "<spiegazione broker>"}}
  ],
  "sentiment_complessivo": "RIALZISTA (ALTA)|RIALZISTA (BASSA)|RIBASSISTA|NEUTRO",
  "conferma_prezzo": "<valutazione prezzo>",
  "decisione": "BUY|SELL|HOLD",
  "allocazione": <numero da 0.0 a 1.0>,
  "motivazione_finale": "<3-4 frasi da Senior Broker che spiegano la gestione del rischio e la size scelta>"
}}
"""


import re

def _clean_json(raw: str) -> str:
    """Estrae il JSON dal testo, ignorando preamboli e formattazione markdown."""
    raw = raw.strip()
    
    # Prova a estrarre il blocco delimitato da ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # Se non ci sono backtick, prova a estrarre tutto quello che sta tra la prima { e l'ultima }
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        return raw[start:end+1].strip()
        
    return raw


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

    response = llm_reason.invoke(prompt)
    raw = _clean_json(response.content)

    try:
        parsed = json.loads(raw)

        # Campi principali
        decision         = parsed.get("decisione", "HOLD").upper()
        allocazione      = float(parsed.get("allocazione", 0.0))
        rationale        = parsed.get("motivazione_finale", "Nessuna motivazione fornita.")

        # Calcolo quantità esatta in Python (deterministico)
        quantity = 0
        if decision == "BUY" and state.get("price"):
            cash = float(portfolio_data.get("cash", 0))
            invest_amount = cash * allocazione
            if "/" in ticker:
                quantity = round(invest_amount / state["price"], 6)
            else:
                quantity = math.floor(invest_amount / state["price"])
        elif decision == "SELL":
            # Cerca la posizione corrente per vendere la quantità esatta
            for pos in portfolio_data.get("positions", []):
                if pos["ticker"] == ticker:
                    quantity = float(pos["qty"])
                    break

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
    ticker   = state.get("ticker")
    quantity = state.get("quantity", 0)
    is_crypto = "/" in (ticker or "")

    # --- Filtro SELL x0: nessuna posizione aperta, tratta come HOLD ---
    if decision == "SELL" and quantity == 0:
        print(f"\n[execute_order] SELL su {ticker} ma posizione = 0 — nessun ordine (trattato come HOLD).")
        return {**state, "decision": "HOLD", "order_id": None, "order_error": None}

    if decision == "HOLD" or quantity == 0:
        print(f"\n[execute_order] Decisione HOLD — nessun ordine inviato.")
        return {**state, "order_id": None, "order_error": None}

    # --- Check mercato aperto per azioni (le crypto sono sempre aperte) ---
    if not is_crypto and not is_market_open():
        print(f"\n[execute_order] ⏸  Mercato chiuso per {ticker} — ordine sospeso (riprenderà nella prossima sessione).")
        return {**state, "order_id": None, "order_error": "mercato_chiuso"}

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
        "candidate_ticker": None,
        "candidate_news": None
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