"""
Entry point del Trading Agent.
Avvia il grafo LangGraph per N cicli sul ticker scelto.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

from journal import init_journal
from agent import build_graph
from state import AgentState

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

TICKER     = "BTC/USD"   # Ticker da monitorare (crypto: mercato 24/7)
MAX_CYCLES = 3        # Quanti cicli eseguire (aumenta per il demo L3)
WAIT_SEC   = 30       # Secondi di attesa tra un ciclo e l'altro (per il loop autonomo)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  TRADING AGENT — Agentic AI Hackathon")
    print(f"  Ticker: {TICKER} | Cicli: {MAX_CYCLES}")
    print("=" * 50)

    # Inizializza il journal SQLite
    init_journal()

    # Costruisce il grafo LangGraph
    graph = build_graph()

    # Stato iniziale
    initial_state: AgentState = {
        "ticker":       TICKER,
        "price":        None,
        "price_error":  None,
        "news_summary": None,
        "news_error":   None,
        "overall_sentiment": None,
        "decision":     None,
        "quantity":     None,
        "rationale":    None,
        "order_id":     None,
        "order_error":  None,
        "cycle_count":  0,
        "max_cycles":   MAX_CYCLES,
    }

    # Avvia il grafo
    # Nota: il loop interno è gestito da LangGraph tramite l'edge condizionale.
    # Se vuoi aggiungere una pausa reale tra i cicli (per il demo autonomo L3),
    # puoi invece fare un loop esterno con MAX_CYCLES=1 e time.sleep(WAIT_SEC).
    final_state = graph.invoke(initial_state)

    print("\n✅ Agent terminato.")
    print(f"   Cicli completati: {final_state['cycle_count']}")
    print(f"   Ultima decisione: {final_state['decision']} x{final_state['quantity']}")


if __name__ == "__main__":
    main()