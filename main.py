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

from tools import get_portfolio

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

MAX_CYCLES = 3      # Quanti cicli globali eseguire
WAIT_SEC   = 30      # Secondi di attesa (gestito internamente da dov'è richiesto o da LangGraph, ma qui non ci serve più il doppio loop)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  TRADING AGENT — Agentic AI Hackathon (Scelta Autonoma)")
    print(f"  Cicli massimi: {MAX_CYCLES}")
    print("=" * 50)

    # Inizializza il journal SQLite
    init_journal()

    # Costruisce il grafo LangGraph
    graph = build_graph()

    # Legge il portafoglio per inizializzare la memoria
    print("\n[main] Sincronizzazione memoria con il portafoglio Alpaca...")
    port = get_portfolio()
    initial_blacklist = [pos["ticker"] for pos in port.get("positions", [])]
    if initial_blacklist:
        print(f"  📌 Asset già in portafoglio (Blacklist iniziale): {', '.join(initial_blacklist)}")

    # Stato iniziale per l'agente esplorativo
    initial_state: AgentState = {
        "ticker":       None,
        "candidate_ticker": None,
        "candidate_news": None,
        "search_attempts": 0,
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
        "blacklist_tickers": initial_blacklist,
    }
    
    # Avvia il grafo e lo lascia lavorare in autonomia finché non raggiunge max_cycles
    final_state = graph.invoke(initial_state)

    print("\n✅ Multi-Agent terminato su tutto il portafoglio.")


if __name__ == "__main__":
    main()