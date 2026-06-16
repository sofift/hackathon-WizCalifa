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

TICKERS    = ["SPY", "QQQ", "GLD", "BTC/USD"]   # Lista di asset per diversificare il portafoglio
MAX_CYCLES = 3       # Quanti cicli globali eseguire
WAIT_SEC   = 30      # Secondi di attesa tra un ciclo globale e l'altro

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  TRADING AGENT — Agentic AI Hackathon (Multi-Asset)")
    print(f"  Tickers: {', '.join(TICKERS)} | Cicli: {MAX_CYCLES}")
    print("=" * 50)

    # Inizializza il journal SQLite
    init_journal()

    # Costruisce il grafo LangGraph
    graph = build_graph()

    for cycle in range(1, MAX_CYCLES + 1):
        print(f"\n" + "=" * 50)
        print(f"🚀 CICLO GLOBALE {cycle}/{MAX_CYCLES}")
        print("=" * 50)
        
        for ticker in TICKERS:
            # Stato iniziale per la singola passata su questo ticker
            initial_state: AgentState = {
                "ticker":       ticker,
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
                "cycle_count":  1,  # Forza l'uscita dal LangGraph dopo 1 passata
                "max_cycles":   1,
            }
            
            # Avvia il grafo per 1 ciclo su questo specifico ticker
            final_state = graph.invoke(initial_state)
            
            print(f"\n✅ {ticker} terminato: {final_state['decision']} x{final_state['quantity']}")
            time.sleep(2)  # Pausa tra un ticker e l'altro
            
        if cycle < MAX_CYCLES:
            print(f"\nAttesa di {WAIT_SEC} secondi prima del prossimo ciclo globale...")
            time.sleep(WAIT_SEC)

    print("\n✅ Multi-Agent terminato su tutto il portafoglio.")


if __name__ == "__main__":
    main()