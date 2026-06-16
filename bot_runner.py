"""
bot_runner.py
Entry point che avvia in parallelo il loop dell'agente e il bot Telegram.

Avvio:
    python bot_runner.py

Thread-1 (daemon): Agent loop — gira in background, esegue cicli di trading.
Thread-2 (main):   Bot Telegram — asyncio event loop, gestisce i comandi.

I due thread comunicano tramite command_bus.py (code thread-safe + stop_flag).
"""

import sys
import asyncio
import threading
from dotenv import load_dotenv

# Fix encoding emoji su Windows (cp1252 non supporta unicode)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()  # carica .env PRIMA di importare i moduli (che leggono os.environ)

from command_bus import stop_flag
from main import run_agent_loop
from telegram_bot import start_bot


def _run_agent_in_thread() -> None:
    """
    Wrapper per eseguire il loop dell'agente in un thread daemon.
    Se l'agente termina o crasha, imposta stop_flag per fermare anche il bot.
    """
    try:
        print("[bot_runner] 🤖 Agent loop partito nel thread background.")
        run_agent_loop()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[bot_runner] ❌ Agente terminato con errore: {e}")
    finally:
        print("[bot_runner] 🏁 Agent loop terminato — imposto stop_flag.")
        stop_flag.set()


if __name__ == "__main__":
    print("=" * 55)
    print("  WizCalifa — Bot Runner")
    print("  Avvio: Agent Loop + Bot Telegram")
    print("=" * 55)

    # Thread-1: Agent loop (daemon — termina quando termina il main thread)
    agent_thread = threading.Thread(
        target=_run_agent_in_thread,
        daemon=True,
        name="AgentLoop",
    )
    agent_thread.start()

    # Thread-2 (main): Bot Telegram (asyncio event loop — bloccante)
    try:
        print("[bot_runner] 📱 Avvio bot Telegram nel thread principale...")
        asyncio.run(start_bot())
    except KeyboardInterrupt:
        print("\n[bot_runner] Ctrl+C ricevuto — shutdown.")
        stop_flag.set()
    except ValueError as e:
        # Token o chat_id mancanti nel .env
        print(f"\n[bot_runner] ❌ Configurazione bot non valida:\n  {e}")
        stop_flag.set()

    # Aspetta che l'agente finisca il ciclo corrente (max 30s)
    agent_thread.join(timeout=30)
    print("[bot_runner] ✅ Shutdown completato.")
