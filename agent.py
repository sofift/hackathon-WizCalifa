"""
Grafo LangGraph del Trading Agent.
Strategia: News Sentiment + Price Confirmation
Ciclo: evaluate_positions → propose_candidate → [evaluate] → fetch_market_data → fetch_news → reason → execute_order → journal → [loop o stop]
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
    model="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=os.environ.get("GROQ_API_KEY"),
)

import time

def safe_invoke(llm, prompt, max_retries=5):
    """Esegue la chiamata all'LLM con un meccanismo di backoff per gestire i rate limit (429)."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = 2 + (attempt * 2)  # 2s, 4s, 6s...
                    print(f"  ⏳ Rate limit Groq! Attendo {sleep_time}s... (Tentativo {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
            raise e

import time

def safe_invoke(llm, prompt, max_retries=5):
    """Esegue la chiamata all'LLM con un meccanismo di backoff per gestire i rate limit (429)."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str:
                if attempt < max_retries - 1:
                    sleep_time = 2 + (attempt * 2)  # 2s, 4s, 6s...
                    print(f"  ⏳ Rate limit Groq! Attendo {sleep_time}s... (Tentativo {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
            raise e

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
# Nodo 0: evaluate_positions — Dedicated Sell Step (Strategia 1)
# Rivaluta TUTTE le posizioni aperte prima di cercare nuovi acquisti.
# ---------------------------------------------------------------------------

SELL_EVAL_PROMPT = """Sei un Senior Broker in modalità SCALPING a 5 minuti.
Hai una posizione aperta su {ticker}: {qty} unità.
Prezzo medio di carico: ${avg_entry:.2f}
Valore attuale: ${market_value:.2f}
Profitto/Perdita: {profit_pct:+.2f}%

Notizie recenti per {ticker}:
{news}

REGOLA DI SCALPING AGGIORNATA:
1. TAKE PROFIT MECCANICO: Se il profitto (Profitto/Perdita) è >= +1.00%, DEVI rispondere SELL per incassare immediatamente.
2. CONDIZIONE DI HOLD: Se il profitto è tra -1.00% e +1.00%, rispondi HOLD (dai tempo all'asset di muoversi), a meno che il sentiment non sia esplicitamente RIBASSISTA.
3. CUT LOSSES: Se sei in perdita netta (< -1.00%) E il sentiment NON è rialzista, rispondi SELL per tagliare le perdite. Se il sentiment è RIBASSISTA a prescindere dal profitto, rispondi SELL.

FORMATO DI RISPOSTA OBBLIGATORIO:
Devi rispondere ESATTAMENTE con questo formato:
DECISIONE|Motivazione breve (max 1 riga)
Esempio: SELL|Profitto superiore all'1%, take profit eseguito. Oppure HOLD|Sentiment ancora rialzista, mantengo.

Rispondi SOLO con la riga richiesta, nient'altro."""


def evaluate_positions(state: AgentState) -> AgentState:
    """Rivaluta ogni posizione aperta e vende se il sentiment non è più favorevole."""
    snap = state.get("portfolio_snapshot", {})
    positions = snap.get("positions", [])

    if not positions:
        return state  # Nessuna posizione aperta, vai direttamente a comprare

    print(f"\n[evaluate_positions] Valuto {len(positions)} posizione/i aperta/e per eventuale SELL...")
    sold_any = False

    for pos in positions:
        ticker = pos["ticker"]
        qty    = pos["qty"]
        mval   = pos.get("market_value", 0)
        avg    = pos.get("avg_entry_price", 0)
        profit = pos.get("profit_pct", 0)

        # Recupera notizie aggiornate per questo asset
        news_result = search_news(ticker)
        if "error" in news_result or not news_result.get("headlines"):
            news_text = "Nessuna notizia disponibile."
        else:
            news_text = "\n".join(news_result["headlines"])

        prompt = SELL_EVAL_PROMPT.format(
            ticker=ticker,
            qty=qty,
            avg_entry=avg,
            market_value=mval,
            profit_pct=profit,
            news=news_text,
        )
        response = safe_invoke(llm_fast, prompt)
        raw_output  = response.content.strip()
        
        # Parsing "DECISIONE|Motivazione"
        parts = raw_output.split("|", 1)
        verdict = parts[0].strip().upper()
        motivazione = parts[1].strip() if len(parts) > 1 else f"Nessuna motivazione. Raw: {raw_output}"

        if "SELL" in verdict:
            print(f"  📌 {ticker}: SELL segnalato — eseguo ordine di vendita (qty={qty})...")
            print(f"     Motivo: {motivazione}")
            result = place_order(ticker, "sell", float(qty))
            if "error" in result:
                print(f"  ⚠️  Errore vendita {ticker}: {result['error']}")
            else:
                print(f"  ✅ {ticker} venduto — order_id: {result['order_id']}")
                log_decision(
                    ticker=ticker,
                    price=mval / float(qty) if float(qty) > 0 else None,
                    decision="SELL",
                    quantity=float(qty),
                    rationale=f"[Scalping Eval] {motivazione}",
                    order_id=result.get("order_id"),
                    outcome="ok",
                )
                sold_any = True
        else:
            print(f"  ➕ {ticker}: HOLD — {motivazione}")

    if sold_any:
        print_journal()

    return state


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
    response = safe_invoke(llm_fast, prompt)
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
    response = safe_invoke(llm_fast, prompt)
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
            # Aggiunge il ticker forzato a session_analyzed per non riproporlo nel ciclo successivo
            if ticker not in session_analyzed:
                session_analyzed.append(ticker)
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

REASON_PROMPT = """Sei un Senior Quantitative Broker che opera in modalità SCALPING (timeframe 5 minuti).
Il tuo obiettivo è massimizzare l'Alpha del portafoglio con operazioni rapide: compra quando c'è momentum, vendi per prendere profitto o tagliare le perdite.

═══════════════════════════════════════
FASE 1 — ANALISI DEL SENTIMENT E CATALIZZATORI
═══════════════════════════════════════
Classifica OGNI titolo di notizia come POSITIVO, NEGATIVO o NEUTRO filtrando il rumore.
MANTIENI SEMPRE il prefisso della fonte originale (es. [Polygon], [Alpaca], [TwelveData]) nel titolo.
- POSITIVO: Sorprese positive agli utili, upgrade di massa degli analisti, M&A, approvazioni normative.
- NEGATIVO: Cause legali gravi, miss sugli utili, downgrade drastici, problemi regolatori.
- NEUTRO: Rumore di fondo, annunci di routine.
- Se non ci sono notizie → NEUTRO.

Titoli delle notizie per {ticker}:
{news_summary}

Calcola il SENTIMENT COMPLESSIVO e il livello di CONFIDENZA (ALTA o BASSA):
- RIALZISTA (ALTA Confidenza) → catalizzatori positivi reali e inequivocabili.
- RIALZISTA (BASSA Confidenza) → notizie moderatamente positive.
- RIBASSISTA → notizie negative rilevanti.
- NEUTRO → solo rumore di fondo o parità.

═══════════════════════════════════════
FASE 2 — DECISIONE OPERATIVA (SCALPING 5 MIN)
═══════════════════════════════════════
Prezzo attuale: {price}
Errore prezzo: {price_error}
Stato del portafoglio:
{portfolio}

REGOLA FONDAMENTALE: NON FARE MAI HOLD SU UN ASSET CHE NON POSSIEDI.
Se non possiedi {ticker} e il sentiment è neutro/debole, devi comunque scegliere:
- Se c'è anche un minimo segnale positivo → BUY con allocazione prudente (0.05).
- Se il sentiment è completamente piatto o negativo → BUY con allocazione minima (0.02) per avere esposizione, oppure HOLD solo se il prezzo non è disponibile.

REGOLE DI TRADING PER SCALPING A 5 MINUTI:

1. BUY — DIVERSIFICAZIONE SETTORIALE E SIZING DINAMICO:
   - Analizza il settore di {ticker} (es. AI, Tech, Moda, Energia) e confrontalo con il portafoglio.
   - Se possiedi già molti asset in quel settore → RIDUCI l'allocazione.
   - Se il portafoglio è scarso in quel settore → AUMENTA l'allocazione.
   - RIALZISTA (ALTA Confidenza): Alloca tra il 10% e il 25% del cash (da 0.10 a 0.25) in base alla necessità di diversificare.
   - RIALZISTA (BASSA Confidenza): Alloca tra il 3% e l'8% del cash (da 0.03 a 0.08).
   - NEUTRO e NON possiedi l'asset: Alloca tra l'1% e il 2% del cash (da 0.01 a 0.02) per esposizione minima.

2. SELL — GESTIONE AVANZATA DELLE NEWS (Se possiedi {ticker}):
   Nello scalping la vendita deve essere chirurgica. Analizza le news per applicare queste strategie:
   - "Sell the News" (Take Profit): Se le news annunciano o confermano un evento positivo che il mercato stava già aspettando (es. "lancio del nuovo prodotto", "trimestrale in linea con le attese"), il momentum sta per esaurirsi. Sentiment: NEUTRO/DEBOLE → SELL tutta la posizione per incassare.
   - "Panic Selling" Controllato (Cut Losses): Se le news riportano crisi gravi aziendali (frodi, indagini SEC, dimissioni a sorpresa, downgrade massicci degli analisti). Sentiment: RIBASSISTA ESTREMO → SELL IMMEDIATAMENTE.
   - "Sector Rotation" (Minaccia Competitiva): Se le news parlano di un enorme successo di un competitor diretto, il momentum per {ticker} si indebolisce. Sentiment: NEUTRO/RIBASSISTA → SELL.
   - "Sell on Silence" (Decadimento): Se non ci sono notizie o leggi solo "rumore di fondo" irrilevante. Sentiment: NEUTRO. Il momentum direzionale è svanito → SELL per liberare capitale.
   - In sintesi: Se possiedi l'asset e il sentiment NON è "RIALZISTA (ALTA Confidenza)", è quasi sempre meglio eseguire SELL per riallocare le risorse.

3. HOLD — SOLO se:
   - Il prezzo non è disponibile (errore).
   - Cash insufficiente per almeno 1 unità.
   - Possiedi già l'asset E il sentiment è RIALZISTA (mantieni la posizione aperta per cavalcare il trend).

Non devi calcolare la quantità esatta. Indica solo l'allocazione in formato decimale rispettando i range:
- Se BUY: indica il decimale scelto (es. 0.15, 0.05, 0.02). DEVE essere compatibile con il livello di confidenza scelto.
- Se SELL: indica 1.0 (vendi tutta la posizione).
- Se HOLD: indica 0.0.

═══════════════════════════════════════
REGOLE FONDAMENTALI
═══════════════════════════════════════
- NON inventare prezzi, notizie o dati di portafoglio.
- Se i dati mancano → HOLD.
- La motivazione finale deve citare la confidenza, l'allocazione scelta e la strategia (scalping / take profit / cut losses).

═══════════════════════════════════════
FORMATO DI OUTPUT
═══════════════════════════════════════
Rispondi SOLO con JSON valido. Niente markdown, niente testo aggiuntivo.

{{
  "analisi_sentiment": [
    {{"titolo": "[Fonte] <titolo>", "classificazione": "POSITIVO|NEGATIVO|NEUTRO", "motivazione": "<spiegazione broker>"}}
  ],
  "sentiment_complessivo": "RIALZISTA (ALTA)|RIALZISTA (BASSA)|RIBASSISTA|NEUTRO",
  "conferma_prezzo": "<valutazione prezzo>",
  "decisione": "BUY|SELL|HOLD",
  "allocazione": <numero da 0.0 a 1.0>,
  "motivazione_finale": "<3-4 frasi da Senior Broker in modalità scalping>"
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

    response = safe_invoke(llm_reason, prompt)
    raw = _clean_json(response.content)

    try:
        parsed = json.loads(raw)

        # Campi principali
        decision         = parsed.get("decisione", "HOLD").upper()
        allocazione      = float(parsed.get("allocazione", 0.0))
        rationale        = parsed.get("motivazione_finale", "Nessuna motivazione fornita.")

        # Verifica se possediamo effettivamente il ticker
        owned_tickers = {pos["ticker"] for pos in portfolio_data.get("positions", [])}
        owns_ticker = ticker in owned_tickers
        
        # Se la decisione è SELL ma non possediamo l'asset, la quantità resterà a 0
        # e verrà skippato senza lanciare un BUY forzato.

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
    
    # Aggiungiamo il ticker appena processato a session_analyzed per non riproporlo al ciclo successivo
    session_analyzed = list(state.get("session_analyzed", []))
    if state.get("ticker") and state["ticker"] not in session_analyzed:
        session_analyzed.append(state["ticker"])
    
    # Resettiamo i parametri di ricerca per il ciclo successivo
    # NOTA: Non resettiamo "ticker" o "decision" qui, altrimenti main.py 
    # non può leggerli per inviare la notifica su Telegram a fine ciclo.
    return {
        **state, 
        "cycle_count": new_count, 
        "search_attempts": 0, 
        "ticker": None, 
        "candidate_ticker": None,
        "candidate_news": None,
        "session_analyzed": session_analyzed,
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
# Nodo Forzato: handle_sector_command (gestisce /compra settore e /vendi settore)
# ---------------------------------------------------------------------------

SECTOR_BUY_PROMPT = """Sei un Senior Quantitative Broker.
Ti è stato richiesto di trovare i migliori ticker per il seguente settore: "{sector}"

Devi identificare ESATTAMENTE i Top 3 o Top 4 ticker azionari americani a MAGGIORE CAPITALIZZAZIONE per questo settore.
Regole TASSATIVE:
- Solo simboli validi negoziabili sulle borse USA (es. AAPL, NVDA, LMT).
- Niente ticker oscuri, delistati o non americani.
- Restituisci ESATTAMENTE una lista in formato JSON di stringhe. Nessun altro testo.

Esempio di output valido:
["LMT", "RTX", "GD"]
"""

SECTOR_SELL_PROMPT = """Sei un Senior Quantitative Broker.
Ti è stata fornita la lista dei ticker attualmente in portafoglio: {owned_tickers}
Devi identificare quali di questi ticker appartengono al seguente settore: "{sector}"

Regole TASSATIVE:
- Restituisci ESATTAMENTE una lista in formato JSON di stringhe.
- Includi SOLO i ticker della lista fornita che operano prevalentemente nel settore "{sector}".
- Se nessuno appartiene a quel settore, restituisci [].
- Nessun altro testo.

Esempio di output valido:
["AAPL", "MSFT"]
"""

def handle_sector_command(state: AgentState) -> AgentState:
    action = state.get("forced_sector_action")
    sector_name = state.get("forced_sector_name")
    chat_id = state.get("forced_chat_id")
    
    print(f"\n[handle_sector_command] Esecuzione comando settore: {action} {sector_name}")
    
    try:
        from command_bus import rep_queue
    except ImportError:
        rep_queue = None

    def _notify(msg: str):
        if rep_queue and chat_id:
            rep_queue.put({"chat_id": chat_id, "text": msg})

    snap = state.get("portfolio_snapshot", {})
    
    if action == "BUY":
        prompt = SECTOR_BUY_PROMPT.format(sector=sector_name)
        response = safe_invoke(llm_fast, prompt)
        raw = _clean_json(response.content)
        try:
            tickers = json.loads(raw)
            if not isinstance(tickers, list):
                raise ValueError("Non è una lista")
        except Exception as e:
            _notify(f"❌ Errore parsing risposta LLM per settore {sector_name}: {e}")
            return {**state, "cycle_count": state.get("max_cycles", 1)}
            
        tickers = [str(t).upper().strip() for t in tickers][:4]
        if not tickers:
            _notify(f"⚠️ Nessun ticker valido trovato per il settore {sector_name}.")
            return {**state, "cycle_count": state.get("max_cycles", 1)}
            
        cash = float(snap.get("cash", 0))
        budget = cash * 0.20
        budget_per_ticker = budget / len(tickers)
        
        lines = [f"📊 *Acquisto Settore: {sector_name}* (Budget: ${budget:,.2f})"]
        for t in tickers:
            p_res = get_price(t)
            if "error" in p_res:
                lines.append(f"❌ `{t}`: Errore prezzo ({p_res['error']})")
                continue
                
            price = p_res["price"]
            if "/" in t:
                qty = round(budget_per_ticker / price, 6)
            else:
                qty = math.floor(budget_per_ticker / price)
                
            if qty <= 0:
                lines.append(f"❌ `{t}`: Cash insufficiente.")
                continue
                
            res = place_order(t, "buy", qty)
            if "error" in res:
                lines.append(f"❌ `{t}`: Errore acquisto ({res['error']})")
            else:
                log_decision(t, price, "BUY", qty, f"Acquisto settore: {sector_name}", res.get("order_id"), "ok")
                lines.append(f"✅ `{t}` acquistato (x{qty})")
                
        _notify("\n".join(lines))
        print_journal()

    elif action == "SELL":
        owned = [p["ticker"] for p in snap.get("positions", [])]
        if not owned:
            _notify("⚠️ Nessuna posizione in portafoglio da vendere.")
            return {**state, "cycle_count": state.get("max_cycles", 1)}
            
        prompt = SECTOR_SELL_PROMPT.format(sector=sector_name, owned_tickers=json.dumps(owned))
        response = safe_invoke(llm_fast, prompt)
        raw = _clean_json(response.content)
        
        try:
            target_tickers = json.loads(raw)
            if not isinstance(target_tickers, list):
                raise ValueError("Non è una lista")
        except Exception as e:
            _notify(f"❌ Errore parsing risposta LLM per settore {sector_name}: {e}")
            return {**state, "cycle_count": state.get("max_cycles", 1)}
            
        target_tickers = [str(t).upper().strip() for t in target_tickers]
        to_sell = [t for t in target_tickers if t in owned]
        
        if not to_sell:
            _notify(f"⚠️ Nessun ticker del settore *{sector_name}* in portafoglio.")
            return {**state, "cycle_count": state.get("max_cycles", 1)}
            
        lines = [f"💼 *Vendita Settore: {sector_name}*"]
        for t in to_sell:
            qty = 0
            for p in snap.get("positions", []):
                if p["ticker"].upper() == t:
                    qty = float(p["qty"])
                    break
            
            res = place_order(t, "sell", qty)
            if "error" in res:
                lines.append(f"❌ `{t}`: Errore vendita ({res['error']})")
            else:
                log_decision(t, None, "SELL", qty, f"Vendita settore: {sector_name}", res.get("order_id"), "ok")
                lines.append(f"✅ `{t}` venduto (x{qty})")
                
        _notify("\n".join(lines))
        print_journal()

    # Avendo eseguito il comando di settore, terminiamo il ciclo per ora
    return {**state, "cycle_count": state.get("max_cycles", 1)}


def route_initial(state: AgentState) -> str:
    if state.get("forced_sector_action"):
        return "handle_sector_command"
    return "evaluate_positions"


# ---------------------------------------------------------------------------
# Costruzione del grafo
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodo dedicato alla valutazione/vendita delle posizioni esistenti
    graph.add_node("evaluate_positions", evaluate_positions)
    
    # Nodo forzato per i comandi di settore da Telegram
    graph.add_node("handle_sector_command", handle_sector_command)

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

    # Entry point condizionale: devia subito ai comandi di settore se presenti
    graph.set_conditional_entry_point(
        route_initial,
        {
            "handle_sector_command": "handle_sector_command",
            "evaluate_positions": "evaluate_positions"
        }
    )
    
    # Se esegue un comando di settore forzato, il ciclo termina immediatamente
    graph.add_edge("handle_sector_command", END)

    graph.add_edge("evaluate_positions", "propose_candidate")

    # Ciclo di ricerca nuovi asset
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

    # Loop condizionale finale: torna a evaluate_positions o termina
    graph.add_conditional_edges(
        "write_journal",
        should_continue,
        {
            "continue": "evaluate_positions",
            "end":      END,
        }
    )

    return graph.compile()