"""
Entry point del Trading Agent.
Modalità: loop temporale — il bot gira per RUN_DURATION_SEC secondi,
lanciando un nuovo ciclo ogni CYCLE_INTERVAL_SEC secondi.
"""

import os
import time
import datetime
from dotenv import load_dotenv

load_dotenv()

from journal import init_journal
from agent import build_graph
from state import AgentState

# ---------------------------------------------------------------------------
# Configurazione temporale
# ---------------------------------------------------------------------------

RUN_DURATION_SEC  = 120   # Durata totale dell'esecuzione (secondi)
CYCLE_INTERVAL_SEC = 30   # Pausa tra un ciclo e il successivo (secondi)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    end_time   = start_time + RUN_DURATION_SEC

    print("=" * 55)
    print("  TRADING AGENT — Agentic AI Hackathon (Time-Based Loop)")
    print(f"  Durata totale : {RUN_DURATION_SEC}s  |  Intervallo ciclo: {CYCLE_INTERVAL_SEC}s")
    print(f"  Avvio         : {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"  Termine previsto: {datetime.datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    print("=" * 55)

    # Inizializza il journal SQLite
    init_journal()

    # Costruisce il grafo LangGraph (una sola volta, riusato ad ogni ciclo)
    graph = build_graph()

    cycle_num     = 0
    # La blacklist persiste tra tutti i cicli della stessa sessione
    session_blacklist: list[str] = []

    while time.time() < end_time:
        cycle_num += 1
        elapsed = time.time() - start_time
        remaining = end_time - time.time()

        print(f"\n{'=' * 55}")
        print(f"🚀 CICLO #{cycle_num}  |  Trascorso: {elapsed:.0f}s  |  Rimanente: {remaining:.0f}s")
        print(f"{'=' * 55}")

        # Ogni ciclo usa max_cycles=1: il grafo esegue uno scouting + una trade decision, poi esce
        cycle_state: AgentState = {
            "ticker":            None,
            "candidate_ticker":  None,
            "candidate_news":    None,
            "search_attempts":   0,
            "price":             None,
            "price_error":       None,
            "news_summary":      None,
            "news_error":        None,
            "overall_sentiment": None,
            "decision":          None,
            "quantity":          None,
            "rationale":         None,
            "order_id":          None,
            "order_error":       None,
            "cycle_count":       0,
            "max_cycles":        1,           # ← Il grafo fa 1 ciclo e restituisce il controllo
            "blacklist_tickers": session_blacklist,  # ← Passa la blacklist accumulata
        }

        final_state = graph.invoke(cycle_state)

        # Aggiorna la blacklist di sessione con eventuali nuovi ticker analizzati
        session_blacklist = final_state.get("blacklist_tickers", session_blacklist)

        # Controlla se c'è ancora tempo per un altro ciclo
        remaining_after = end_time - time.time()
        if remaining_after <= 0:
            print(f"\n⏰ Tempo scaduto dopo il ciclo #{cycle_num}.")
            break

        if remaining_after < CYCLE_INTERVAL_SEC:
            print(f"\n⏰ Tempo rimanente ({remaining_after:.0f}s) insufficiente per un nuovo ciclo completo. Stop.")
            break

        print(f"\n⏳ Prossimo ciclo tra {CYCLE_INTERVAL_SEC}s... (Ctrl+C per interrompere)")
        time.sleep(CYCLE_INTERVAL_SEC)

    total = time.time() - start_time
    print(f"\n✅ Sessione terminata — {cycle_num} cicli eseguiti in {total:.0f}s.")
    print(f"   Blacklist finale: {session_blacklist}")


if __name__ == "__main__":
    main()