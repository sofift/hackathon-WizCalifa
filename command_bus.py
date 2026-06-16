"""
command_bus.py
Singleton di modulo condiviso tra il loop dell'agente (Thread-1) e il bot Telegram (Thread-2).
Importato da entrambi — non istanziare direttamente, usare le variabili qui esposte.

Schema messaggi cmd_queue  (Bot → Agent):
  {"action": "sell"|"sell_all"|"report"|"stop"|"status", "ticker": str|None, "chat_id": int}

Schema messaggi rep_queue  (Agent → Bot):
  {"chat_id": int, "text": str}
"""

import queue
import threading

# ── Code di comunicazione ──────────────────────────────────────────────────
# Bot → Agent: comandi in entrata
cmd_queue: queue.Queue = queue.Queue()

# Agent → Bot: risposte da inviare all'utente
rep_queue: queue.Queue = queue.Queue()

# ── Segnale di stop ────────────────────────────────────────────────────────
# Impostato dal bot via /stop; il loop dell'agente lo controlla ogni ciclo.
stop_flag: threading.Event = threading.Event()

# ── Stato condiviso dell'agente (letto dal bot per /status) ───────────────
# Accedere sempre tramite status_lock per thread-safety.
agent_status: dict = {
    "running":       False,
    "cycle_count":   0,
    "last_decision": None,   # "BUY" | "SELL" | "HOLD"
    "last_ticker":   None,
}
status_lock: threading.Lock = threading.Lock()
