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
# Bot -> Agent: comandi in entrata (indicizzate per chat_id)
_cmd_queues: dict[int, queue.Queue] = {}
_cmd_queues_lock = threading.Lock()

# Agent -> Bot: risposte da inviare all'utente (unica coda condivisa per il bot)
rep_queue: queue.Queue = queue.Queue()

# Bot -> Agent: risposte dell'utente alle richieste di conferma (indicizzate per chat_id)
_confirm_queues: dict[int, queue.Queue] = {}
_confirm_queues_lock = threading.Lock()

# -- Segnale di stop globale -------------------------------------------------
stop_flag: threading.Event = threading.Event()

# -- Stato condiviso dell'agente ---------------------------------------------
_agent_statuses: dict[int, dict] = {}
status_lock: threading.Lock = threading.Lock()

# -- Conferme pendenti -------------------------------------------------------
_pending_confirmations: dict[int, dict] = {}
confirm_lock: threading.Lock = threading.Lock()


def get_cmd_queue(chat_id: int) -> queue.Queue:
    with _cmd_queues_lock:
        if chat_id not in _cmd_queues:
            _cmd_queues[chat_id] = queue.Queue()
        return _cmd_queues[chat_id]

def get_confirm_queue(chat_id: int) -> queue.Queue:
    with _confirm_queues_lock:
        if chat_id not in _confirm_queues:
            _confirm_queues[chat_id] = queue.Queue()
        return _confirm_queues[chat_id]

def get_agent_status(chat_id: int) -> dict:
    with status_lock:
        if chat_id not in _agent_statuses:
            _agent_statuses[chat_id] = {
                "running":       False,
                "cycle_count":   0,
                "last_decision": None,
                "last_ticker":   None,
            }
        return _agent_statuses[chat_id]

def get_pending_confirmations(chat_id: int) -> dict:
    with confirm_lock:
        if chat_id not in _pending_confirmations:
            _pending_confirmations[chat_id] = {}
        return _pending_confirmations[chat_id]
