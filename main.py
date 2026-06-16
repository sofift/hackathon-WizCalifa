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
from tools import get_portfolio   # lettura diretta live del portafoglio Alpaca

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

    # -----------------------------------------------------------------------
    # Snapshot iniziale del portafoglio Alpaca — solo per display.
    # I ticker in portafoglio NON vanno in blacklist: l'agente può sempre
    # rivalutarli per un eventuale incremento di posizione.
    # -----------------------------------------------------------------------
    print("\n[main] Lettura portafoglio Alpaca...")
    initial_portfolio = get_portfolio()
    if "error" in initial_portfolio:
        print(f"  ⚠️  Impossibile leggere il portafoglio: {initial_portfolio['error']}")
    else:
        cash  = initial_portfolio.get("cash", 0)
        pval  = initial_portfolio.get("portfolio_value", 0)
        positions = initial_portfolio.get("positions", [])
        print(f"  💰 Cash disponibile  : ${cash:,.2f}")
        print(f"  📊 Valore portafoglio: ${pval:,.2f}")
        if positions:
            print(f"  📌 Posizioni aperte  :")
            for p in positions:
                print(f"     • {p['ticker']:10s}  qty={p['qty']}  valore=${p['market_value']:,.2f}")
        else:
            print("  📭 Nessuna posizione aperta.")

    # session_analyzed: ticker già analizzati nella sessione (non ripetere).
    # I ticker in portafoglio sono esclusi da questa lista: possono essere ri-proposti.
    session_analyzed: list[str] = []
    cycle_num = 0

    while time.time() < end_time:
        cycle_num += 1
        elapsed = time.time() - start_time
        remaining = end_time - time.time()

        print(f"\n{'=' * 55}")
        print(f"🚀 CICLO #{cycle_num}  |  Trascorso: {elapsed:.0f}s  |  Rimanente: {remaining:.0f}s")
        print(f"{'=' * 55}")

        # Aggiornamento live del portafoglio prima di ogni ciclo
        live_portfolio = get_portfolio()
        if "error" not in live_portfolio:
            live_positions = live_portfolio.get("positions", [])
            live_cash = live_portfolio.get("cash", 0)
            live_pval = live_portfolio.get("portfolio_value", 0)
            owned = [p['ticker'] for p in live_positions]
            owned_str = ", ".join(owned) if owned else "nessuna"
            print(f"  💼 Portfolio live → Cash: ${live_cash:,.2f}  |  Valore: ${live_pval:,.2f}  |  Posizioni: {owned_str}")
            portfolio_snapshot = {
                "cash": live_cash,
                "portfolio_value": live_pval,
                "positions": live_positions,
            }
        else:
            print(f"  ⚠️  Aggiornamento portafoglio fallito: {live_portfolio['error']}")
            portfolio_snapshot = {}

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
            "max_cycles":        1,
            "blacklist_tickers": [],                    # non più usata per i ticker di portafoglio
            "session_analyzed":  list(session_analyzed), # ticker già analizzati questa sessione
            "portfolio_snapshot": portfolio_snapshot,
        }

        final_state = graph.invoke(cycle_state)

        # Propaga session_analyzed al ciclo successivo
        session_analyzed = final_state.get("session_analyzed", session_analyzed)

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
    print(f"   Ticker analizzati questa sessione: {session_analyzed}")


if __name__ == "__main__":
    main()