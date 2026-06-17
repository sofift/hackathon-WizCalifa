"""
command_bus.py
Singleton di modulo condiviso tra il loop dell'agente (Thread-1) e il bot Telegram (Thread-2).
Importato da entrambi — non istanziare direttamente, usare le variabili qui esposte.

Schema messaggi cmd_queue  (Bot -> Agent):
  {"action": "buy_ticker"|"sell_ticker"|"buy_sector"|"sell_sector"|"sell_all"|"report"|"stop",
   "target": str|None, "chat_id": int}

Schema messaggi rep_queue  (Agent -> Bot):
  {"chat_id": int, "text": str}                      -> messaggio semplice
  {"chat_id": int, "text": str, "confirm_id": str}   -> messaggio con bottoni SI/NO (richiesta conferma)

Schema messaggi confirm_queue  (Bot -> Agent):
  {"confirm_id": str, "answer": "yes"|"no"}           -> risposta dell'utente al tap del bottone
"""

import queue
import threading

# -- Code di comunicazione ---------------------------------------------------
# Bot -> Agent: comandi in entrata
cmd_queue: queue.Queue = queue.Queue()

# Agent -> Bot: risposte da inviare all'utente
rep_queue: queue.Queue = queue.Queue()

# Bot -> Agent: risposte dell'utente alle richieste di conferma (tap dei bottoni inline)
confirm_queue: queue.Queue = queue.Queue()

# -- Segnale di stop ---------------------------------------------------------
# Impostato dal bot via /stop; il loop dell'agente lo controlla ogni ciclo.
stop_flag: threading.Event = threading.Event()

# -- Stato condiviso dell'agente (letto dal bot per /status) -----------------
# Accedere sempre tramite status_lock per thread-safety.
agent_status: dict = {
    "running":       False,
    "cycle_count":   0,
    "last_decision": None,   # "BUY" | "SELL" | "HOLD"
    "last_ticker":   None,
}
status_lock: threading.Lock = threading.Lock()

# -- Conferme pendenti -------------------------------------------------------
# Tiene traccia delle richieste di conferma ancora in attesa di risposta.
# Chiave = confirm_id (str), valore = dict con metadati (ticker, motivo, timestamp).
# Letto/scritto sia dall'agente (crea la richiesta) sia dal bot (la chiude al tap).
# Accedere sempre tramite confirm_lock.
pending_confirmations: dict = {}
confirm_lock: threading.Lock = threading.Lock()
