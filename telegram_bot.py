"""
telegram_bot.py
Bot Telegram asincrono per il controllo del Trading Agent.

Sicurezza: risponde SOLO al TELEGRAM_CHAT_ID caricato dal file .env.
Qualsiasi altro utente riceve un rifiuto secco senza log dell'azione.

Comandi:
  /start      — messaggio di benvenuto + lista comandi
  /status     — stato attuale dell'agente (attivo/fermo, cicli, ultima decisione)
  /report     — portfolio live + ultime 5 decisioni dal journal
  /compra ticker AAPL  — forza acquisto del ticker
  /compra settore armi — acquisto top 3/4 ticker del settore specificato
  /vendi ticker TSLA   — forza SELL su TSLA
  /vendi settore energia — vende tutti i ticker in portafoglio di quel settore
  /vendi tutto         — forza SELL su tutte le posizioni aperte
  /stop                — invia segnale di stop all'agente
"""

import os
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
)

from command_bus import (
    cmd_queue, rep_queue, confirm_queue, stop_flag, agent_status, status_lock,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credenziali — caricate dal .env tramite load_dotenv() in bot_runner.py
# ---------------------------------------------------------------------------

BOT_TOKEN       = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Supporta più chat_id separati da virgola (es. "123,456,789")
_raw_chat_ids = os.environ.get("TELEGRAM_CHAT_ID", "0")
ALLOWED_CHAT_IDS: set[int] = {
    int(cid.strip()) for cid in _raw_chat_ids.split(",") if cid.strip().isdigit()
}

# ---------------------------------------------------------------------------
# Guard di sicurezza — DEVE essere il primo check di ogni handler
# ---------------------------------------------------------------------------

def _is_authorized(update: Update) -> bool:
    """Controlla che il chat_id sia tra quelli autorizzati."""
    return (
        len(ALLOWED_CHAT_IDS) > 0
        and 0 not in ALLOWED_CHAT_IDS
        and update.effective_chat is not None
        and update.effective_chat.id in ALLOWED_CHAT_IDS
    )


async def _deny(update: Update) -> None:
    """Risposta silenziosa per chat non autorizzate (non rivela info sull'agente)."""
    await update.message.reply_text("⛔ Non autorizzato.")


# ---------------------------------------------------------------------------
# Handler /start
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await _deny(update)
        return
    text = (
        "🤖 *WizCalifa Trading Bot* attivo\\!\n\n"
        "Comandi disponibili:\n"
        "`/status`        — stato dell'agente\n"
        "`/report`        — portfolio \\+ ultime decisioni\n"
        "`/compra settore [nome]` — compra top ticker del settore\n"
        "`/compra ticker [sym]`   — compra ticker specifico\n"
        "`/vendi settore [nome]`  — vende ticker del settore\n"
        "`/vendi ticker [sym]`    — vende ticker specifico\n"
        "`/vendi tutto`   — vendi tutte le posizioni\n"
        "`/stop`          — ferma l'agente"
    )
    await update.message.reply_text(text, parse_mode="MarkdownV2")


# ---------------------------------------------------------------------------
# Handler /status
# ---------------------------------------------------------------------------

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await _deny(update)
        return
    with status_lock:
        running  = agent_status["running"]
        cycles   = agent_status["cycle_count"]
        last_dec = agent_status["last_decision"] or "—"
        last_tk  = agent_status["last_ticker"]   or "—"

    emoji = "🟢" if running else "🔴"
    stato = "ATTIVO" if running else "FERMO"
    text  = (
        f"{emoji} *Agente: {stato}*\n"
        f"Cicli completati: `{cycles}`\n"
        f"Ultima decisione: `{last_dec}` su `{last_tk}`"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Handler /report
# ---------------------------------------------------------------------------

async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await _deny(update)
        return
    cmd_queue.put({
        "action":  "report",
        "ticker":  None,
        "chat_id": update.effective_chat.id,
    })
    await update.message.reply_text("⏳ Report accodato — arriverà al prossimo ciclo \\(max 3s\\)\\.", parse_mode="MarkdownV2")


# ---------------------------------------------------------------------------
# Handler /compra
# ---------------------------------------------------------------------------

async def cmd_compra(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await _deny(update)
        return

    args = context.args
    if not args or len(args) < 2:
        await update.message.reply_text(
            "⚠️ Uso corretto:\n`/compra settore energia` oppure `/compra ticker AAPL`",
            parse_mode="Markdown",
        )
        return

    tipo = args[0].strip().lower()
    target = " ".join(args[1:]).strip()

    if tipo == "settore":
        cmd_queue.put({
            "action":  "buy_sector",
            "target":  target,
            "chat_id": update.effective_chat.id,
        })
        await update.message.reply_text(
            f"⏳ Acquisto settore *{target}* accodato — l'LLM sceglierà i Top 3/4 ticker.",
            parse_mode="Markdown",
        )
    elif tipo == "ticker":
        cmd_queue.put({
            "action":  "buy_ticker",
            "target":  target.upper(),
            "chat_id": update.effective_chat.id,
        })
        await update.message.reply_text(
            f"⏳ Acquisto `{target.upper()}` accodato.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text("⚠️ Usa 'settore' o 'ticker'.", parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Handler /vendi
# ---------------------------------------------------------------------------

async def cmd_vendi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await _deny(update)
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "⚠️ Uso corretto:\n`/vendi settore energia`\n`/vendi ticker AAPL`\n`/vendi tutto`",
            parse_mode="Markdown",
        )
        return

    tipo = args[0].strip().lower()
    
    if tipo == "tutto":
        cmd_queue.put({
            "action":  "sell_all",
            "target":  None,
            "chat_id": update.effective_chat.id,
        })
        await update.message.reply_text(
            "⏳ *Vendita massiva* accodata — eseguirò tutte le posizioni al prossimo ciclo.",
            parse_mode="Markdown",
        )
        return

    if len(args) < 2:
        # Fallback per la vecchia sintassi `/vendi TSLA`
        cmd_queue.put({
            "action":  "sell_ticker",
            "target":  tipo.upper(),
            "chat_id": update.effective_chat.id,
        })
        await update.message.reply_text(
            f"⏳ Vendita di `{tipo.upper()}` accodata.",
            parse_mode="Markdown",
        )
        return

    target = " ".join(args[1:]).strip()
    if tipo == "settore":
        cmd_queue.put({
            "action":  "sell_sector",
            "target":  target,
            "chat_id": update.effective_chat.id,
        })
        await update.message.reply_text(
            f"⏳ Vendita settore *{target}* accodata — l'LLM filtrerà il portafoglio.",
            parse_mode="Markdown",
        )
    elif tipo == "ticker":
        cmd_queue.put({
            "action":  "sell_ticker",
            "target":  target.upper(),
            "chat_id": update.effective_chat.id,
        })
        await update.message.reply_text(
            f"⏳ Vendita `{target.upper()}` accodata.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text("⚠️ Usa 'settore', 'ticker' o 'tutto'.", parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Handler /stop
# ---------------------------------------------------------------------------

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await _deny(update)
        return
    stop_flag.set()
    cmd_queue.put({
        "action":  "stop",
        "ticker":  None,
        "chat_id": update.effective_chat.id,
    })
    await update.message.reply_text(
        "🛑 Segnale di *stop* inviato\\. L'agente si fermerà al termine del ciclo corrente\\.",
        parse_mode="MarkdownV2",
    )


# ---------------------------------------------------------------------------
# Handler bottoni inline — conferma vendita titoli protetti
# callback_data: "confirm:<answer>:<confirm_id>"  (answer = "yes" | "no")
# ---------------------------------------------------------------------------

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    if not _is_authorized(update):
        await query.answer("Non autorizzato.", show_alert=True)
        return

    data = query.data or ""
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != "confirm":
        await query.answer()
        return

    _, answer, confirm_id = parts
    confirm_queue.put({"confirm_id": confirm_id, "answer": answer})

    await query.answer("Ricevuto!")
    try:
        scelta = "Vendita CONFERMATA" if answer == "yes" else "Vendita ANNULLATA"
        await query.edit_message_text(
            text=(query.message.text or "Conferma") + f"\n\nRisposta: {scelta}",
            reply_markup=None,
        )
    except Exception as e:
        logger.error(f"on_callback edit error: {e}")


# ---------------------------------------------------------------------------
# Task di polling della rep_queue (Agent -> Bot)
# Gira in background nell'event loop asyncio del bot e invia i messaggi
# all'utente non appena l'agente li produce.
# ---------------------------------------------------------------------------

async def _poll_rep_queue(app: Application) -> None:
    """Legge rep_queue ogni secondo e spedisce i messaggi all'utente autorizzato."""
    while not stop_flag.is_set():
        try:
            while not rep_queue.empty():
                msg = rep_queue.get_nowait()
                target_ids = [msg["chat_id"]] if msg.get("chat_id") else list(ALLOWED_CHAT_IDS)
                text       = msg.get("text", "").strip()
                confirm_id = msg.get("confirm_id")
                if not (text and target_ids):
                    continue

                if len(text) > 4000:
                    text = text[:4000] + "\n...(troncato)"

                # Se e' una richiesta di conferma, aggiungi bottoni inline
                reply_markup = None
                if confirm_id:
                    reply_markup = InlineKeyboardMarkup([[
                        InlineKeyboardButton("Si, vendi", callback_data=f"confirm:yes:{confirm_id}"),
                        InlineKeyboardButton("No, tieni", callback_data=f"confirm:no:{confirm_id}"),
                    ]])

                for cid in target_ids:
                    try:
                        await app.bot.send_message(
                            chat_id=cid,
                            text=text,
                            parse_mode="Markdown",
                            reply_markup=reply_markup,
                        )
                    except Exception as e_md:
                        logger.error(f"send Markdown fallito ({e_md}) - retry plain.")
                        try:
                            await app.bot.send_message(
                                chat_id=cid,
                                text=text,
                                reply_markup=reply_markup,
                            )
                        except Exception as e_plain:
                            logger.error(f"send plain fallito per {cid}: {e_plain}")
        except Exception as e:
            logger.error(f"_poll_rep_queue error: {e}")
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Avvio del bot
# ---------------------------------------------------------------------------

async def start_bot() -> None:
    """
    Costruisce e avvia il bot Telegram.
    Bloccante finché stop_flag non viene impostato.
    Chiamato da bot_runner.py nel thread principale (asyncio.run).
    """
    if not BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN non trovato nel file .env\n"
            "Ottieni il token da @BotFather su Telegram e aggiungilo al .env."
        )
    if not ALLOWED_CHAT_IDS or ALLOWED_CHAT_IDS == {0}:
        raise ValueError(
            "TELEGRAM_CHAT_ID non trovato nel file .env\n"
            "Ottieni il tuo chat_id scrivendo a @userinfobot e aggiungilo al .env.\n"
            "Per più utenti, separali con virgola: TELEGRAM_CHAT_ID=123,456,789"
        )

    app = Application.builder().token(BOT_TOKEN).build()

    # Registra i handler
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("compra", cmd_compra))
    app.add_handler(CommandHandler("vendi",  cmd_vendi))
    app.add_handler(CommandHandler("stop",   cmd_stop))
    # Handler bottoni inline conferma vendita titoli protetti
    app.add_handler(CallbackQueryHandler(on_callback, pattern=r"^confirm:"))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    print("[telegram_bot] 📱 Bot avviato — in attesa di comandi...")
    print(f"[telegram_bot] 🔒 Chat_id autorizzati: {ALLOWED_CHAT_IDS}")

    # Avvia il polling della coda di risposta
    asyncio.create_task(_poll_rep_queue(app))

    # Blocca finché stop_flag non viene impostato (dal bot o dall'agente)
    while not stop_flag.is_set():
        await asyncio.sleep(1)

    print("[telegram_bot] 🛑 Stop ricevuto — chiudo il bot...")
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
