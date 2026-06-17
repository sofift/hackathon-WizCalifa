"""
main.py
Loop principale del Trading Agent.

Modalità standalone:  python main.py
Modalità bot:         python bot_runner.py  (avvia anche il bot Telegram)

Ogni ciclo:
  1. Controlla stop_flag (impostato dal bot /stop)
  2. Elabora i comandi in coda dalla cmd_queue (bot → agent)
  3. Esegue il grafo LangGraph (fetch → decide → execute → journal)
  4. Aggiorna agent_status (letto dal bot per /status)
"""

import os
import time
import datetime
from dotenv import load_dotenv

load_dotenv()

from journal import init_journal, log_decision, get_recent_decisions
from agent import build_graph
from state import AgentState
from tools import get_portfolio, place_order, get_position_qty

# ---------------------------------------------------------------------------
# Configurazione temporale
# ---------------------------------------------------------------------------

RUN_DURATION_SEC   = 300   # Durata totale della sessione (secondi)
CYCLE_INTERVAL_SEC = 3     # Pausa minima tra un ciclo e il successivo

# ---------------------------------------------------------------------------
# Elaborazione comandi Telegram (cmd_queue → rep_queue)
# ---------------------------------------------------------------------------

def _format_report(portfolio: dict) -> str:
    """Formatta il report portfolio + ultime decisioni per Telegram."""
    lines = ["📊 *REPORT AGENTE*\n"]

    if "error" in portfolio:
        lines.append(f"⚠️ Portfolio non disponibile: {portfolio['error']}")
    else:
        cash = portfolio.get("cash", 0)
        pval = portfolio.get("portfolio_value", 0)
        positions = portfolio.get("positions", [])
        lines.append(f"💰 Cash: `${cash:,.2f}`")
        lines.append(f"📈 Valore totale: `${pval:,.2f}`")
        if positions:
            lines.append("\n*Posizioni aperte:*")
            for p in positions:
                profit = p.get("profit_pct", 0)
                emoji  = "📈" if profit >= 0 else "📉"
                lines.append(
                    f"  {emoji} `{p['ticker']}` — qty: `{p['qty']}` | "
                    f"valore: `${p['market_value']:,.2f}` | P/L: `{profit:+.1f}%`"
                )
        else:
            lines.append("📭 Nessuna posizione aperta.")

    # Ultime 5 decisioni dal journal
    decisions = get_recent_decisions(limit=5)
    if decisions:
        lines.append("\n*Ultime decisioni:*")
        for d in decisions:
            dec_emoji = "🟢" if d["decision"] == "BUY" else "🔴" if d["decision"] == "SELL" else "⚪"
            price_str = f"@ ${d['price']:.2f}" if d["price"] else "@ N/D"
            lines.append(
                f"  {dec_emoji} `{d['ticker']}` {d['decision']} x{d['quantity']} "
                f"{price_str} — {d['timestamp'][:16]}"
            )

    return "\n".join(lines)


def _process_commands(live_portfolio: dict) -> dict:
    """
    Svuota la cmd_queue ed esegue ogni comando ricevuto dal bot Telegram.
    Mette i risultati in rep_queue per l'invio all'utente.
    Se è un comando per settore, restituisce un dizionario con gli intenti da passare al LangGraph.
    Chiamata all'inizio di ogni ciclo, PRIMA di invocare il grafo.
    """
    forced_state = {}
    try:
        from command_bus import cmd_queue, rep_queue
    except ImportError:
        return forced_state  # command_bus non disponibile (modalità standalone senza bot)

    while not cmd_queue.empty():
        try:
            cmd     = cmd_queue.get_nowait()
            action  = cmd.get("action", "")
            target  = cmd.get("target")
            chat_id = cmd.get("chat_id", int(os.environ.get("TELEGRAM_CHAT_ID", "0")))

            print(f"\n[main] 📨 Comando Telegram ricevuto: action={action} target={target}")

            # ── SELL_TICKER (ex sell): vendi un ticker specifico ─────────────
            if action in ("sell", "sell_ticker"):
                ticker = target or cmd.get("ticker")
                qty = get_position_qty(live_portfolio, ticker)
                if qty <= 0:
                    rep_queue.put({
                        "chat_id": chat_id,
                        "text": f"⚠️ Nessuna posizione aperta su `{ticker}` — niente da vendere.",
                    })
                else:
                    result = place_order(ticker, "sell", qty)
                    if "error" in result:
                        rep_queue.put({
                            "chat_id": chat_id,
                            "text": f"❌ Errore vendita `{ticker}`: {result['error']}",
                        })
                    else:
                        log_decision(
                            ticker=ticker,
                            price=None,
                            decision="SELL",
                            quantity=qty,
                            rationale="Comando forzato via Telegram (/vendi)",
                            order_id=result["order_id"],
                            outcome="ok",
                        )
                        rep_queue.put({
                            "chat_id": chat_id,
                            "text": (
                                f"✅ `{ticker}` venduto \\(x{qty}\\)\n"
                                f"Order ID: `{result['order_id']}`\n"
                                f"Status: `{result['status']}`"
                            ),
                        })

            # ── BUY_TICKER: acquista ticker specifico (20% cash o max) ─────────
            elif action == "buy_ticker":
                from tools import get_price
                ticker = target
                price_res = get_price(ticker)
                if "error" in price_res:
                    rep_queue.put({"chat_id": chat_id, "text": f"❌ Errore prezzo per `{ticker}`: {price_res['error']}"})
                else:
                    cash = float(live_portfolio.get("cash", 0))
                    # Alloca il 20% del cash per l'acquisto singolo
                    alloc = cash * 0.20
                    price = price_res["price"]
                    if "/" in ticker:
                        qty = round(alloc / price, 6)
                    else:
                        import math
                        qty = math.floor(alloc / price)
                        
                    if qty <= 0:
                        rep_queue.put({"chat_id": chat_id, "text": f"❌ Cash insufficiente per acquistare `{ticker}`."})
                    else:
                        result = place_order(ticker, "buy", qty)
                        if "error" in result:
                            rep_queue.put({"chat_id": chat_id, "text": f"❌ Errore acquisto `{ticker}`: {result['error']}"})
                        else:
                            log_decision(ticker, price, "BUY", qty, "Comando forzato via Telegram (/compra ticker)", result.get("order_id"), "ok")
                            rep_queue.put({"chat_id": chat_id, "text": f"✅ `{ticker}` acquistato \\(x{qty}\\)\nOrder ID: `{result.get('order_id')}`"})

            # ── BUY_SECTOR / SELL_SECTOR: delega a LangGraph ──────────────────
            elif action in ("buy_sector", "sell_sector"):
                forced_state["forced_sector_action"] = "BUY" if action == "buy_sector" else "SELL"
                forced_state["forced_sector_name"]   = target
                forced_state["forced_chat_id"]       = chat_id

            # ── SELL_ALL: vendi tutte le posizioni ────────────────────────
            elif action == "sell_all":
                positions = live_portfolio.get("positions", [])
                if not positions:
                    rep_queue.put({
                        "chat_id": chat_id,
                        "text": "⚠️ Nessuna posizione aperta da vendere.",
                    })
                else:
                    lines = ["💼 *Vendita massiva:*"]
                    for pos in positions:
                        t   = pos["ticker"]
                        qty = pos["qty"]
                        res = place_order(t, "sell", qty)
                        if "error" in res:
                            lines.append(f"  ❌ `{t}`: {res['error'][:60]}")
                        else:
                            log_decision(
                                ticker=t,
                                price=pos.get("avg_entry_price"),
                                decision="SELL",
                                quantity=qty,
                                rationale="Comando forzato via Telegram (/vendi tutto)",
                                order_id=res["order_id"],
                                outcome="ok",
                            )
                            lines.append(f"  ✅ `{t}` x{qty} — `{res['order_id']}`")
                    rep_queue.put({"chat_id": chat_id, "text": "\n".join(lines)})

            # ── REPORT: portfolio + journal ───────────────────────────────
            elif action == "report":
                fresh_portfolio = get_portfolio()
                report_text = _format_report(fresh_portfolio)
                rep_queue.put({"chat_id": chat_id, "text": report_text})

            # ── STOP: conferma all'utente (stop_flag già impostato dal bot) ─
            elif action == "stop":
                rep_queue.put({
                    "chat_id": chat_id,
                    "text": "🛑 Agente fermato. Sessione terminata.",
                })

        except Exception as e:
            print(f"[main] ⚠️  Errore elaborazione comando: {e}")

    return forced_state


# ---------------------------------------------------------------------------
# Loop principale — esportato per bot_runner.py
# ---------------------------------------------------------------------------

def run_agent_loop() -> None:
    """
    Loop temporale del Trading Agent.
    Può essere chiamato direttamente (standalone) o da bot_runner.py (con bot).
    """
    start_time = time.time()
    end_time   = start_time + RUN_DURATION_SEC

    print("=" * 55)
    print("  TRADING AGENT — Agentic AI Hackathon")
    print(f"  Durata: {RUN_DURATION_SEC}s  |  Intervallo ciclo: {CYCLE_INTERVAL_SEC}s")
    print(f"  Avvio: {datetime.datetime.now().strftime('%H:%M:%S')}  "
          f"| Fine prevista: {datetime.datetime.fromtimestamp(end_time).strftime('%H:%M:%S')}")
    print("=" * 55)

    init_journal()
    graph = build_graph()

    # Snapshot iniziale
    print("\n[main] Lettura portafoglio Alpaca iniziale...")
    initial_portfolio = get_portfolio()
    if "error" not in initial_portfolio:
        cash      = initial_portfolio.get("cash", 0)
        pval      = initial_portfolio.get("portfolio_value", 0)
        positions = initial_portfolio.get("positions", [])
        print(f"  💰 Cash: ${cash:,.2f}  |  Valore: ${pval:,.2f}")
        if positions:
            for p in positions:
                print(f"     • {p['ticker']:10s}  qty={p['qty']}  valore=${p['market_value']:,.2f}")
        else:
            print("  📭 Nessuna posizione aperta.")
    else:
        print(f"  ⚠️  Portafoglio non disponibile: {initial_portfolio['error']}")

    # Aggiorna agent_status — agente avviato
    try:
        from command_bus import agent_status, status_lock, stop_flag
        with status_lock:
            agent_status["running"]     = True
            agent_status["cycle_count"] = 0
        _has_bus = True
    except ImportError:
        stop_flag  = None
        _has_bus   = False

    session_analyzed: list[str] = []
    cycle_num = 0

    while time.time() < end_time:

        # ── Controllo stop_flag (impostato da /stop Telegram) ─────────────
        if _has_bus and stop_flag.is_set():
            print("\n[main] 🛑 Stop flag ricevuto dal bot — termino il loop.")
            break

        cycle_num += 1
        elapsed   = time.time() - start_time
        remaining = end_time - time.time()
        print(f"\n{'=' * 55}")
        print(f"🚀 CICLO #{cycle_num}  |  Trascorso: {elapsed:.0f}s  |  Rimanente: {remaining:.0f}s")
        print(f"{'=' * 55}")

        # ── Portfolio live ─────────────────────────────────────────────────
        live_portfolio = get_portfolio()
        if "error" not in live_portfolio:
            live_cash  = live_portfolio.get("cash", 0)
            live_pval  = live_portfolio.get("portfolio_value", 0)
            owned      = [p["ticker"] for p in live_portfolio.get("positions", [])]
            owned_str  = ", ".join(owned) if owned else "nessuna"
            print(f"  💼 Cash: ${live_cash:,.2f}  |  Valore: ${live_pval:,.2f}  |  Posizioni: {owned_str}")
            portfolio_snapshot = {
                "cash":            live_cash,
                "portfolio_value": live_pval,
                "positions":       live_portfolio.get("positions", []),
            }
        else:
            print(f"  ⚠️  Portfolio non disponibile: {live_portfolio['error']}")
            portfolio_snapshot = {}

        # ── Elabora comandi Telegram (PRIMA del grafo) ─────────────────────
        forced_state = _process_commands(live_portfolio if "error" not in live_portfolio else {})

        # ── Esegui grafo LangGraph ─────────────────────────────────────────
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
            "blacklist_tickers": [],
            "session_analyzed":  list(session_analyzed),
            "portfolio_snapshot": portfolio_snapshot,
            "forced_sector_action": forced_state.get("forced_sector_action"),
            "forced_sector_name":   forced_state.get("forced_sector_name"),
            # Passiamo anche il chat_id nello stato per rispondere, se necessario, dal grafo (come metadato in rationale)
            "forced_chat_id":       forced_state.get("forced_chat_id")
        }

        final_state = graph.invoke(cycle_state)

        # ── Aggiorna session_analyzed ──────────────────────────────────────
        session_analyzed = final_state.get("session_analyzed", session_analyzed)
        if cycle_num % 10 == 0 and session_analyzed:
            live_owned       = {p["ticker"] for p in live_portfolio.get("positions", [])}
            session_analyzed = [t for t in session_analyzed if t in live_owned]
            print(f"\n  🔄 Reset session_analyzed (ciclo {cycle_num})")

        # ── Aggiorna agent_status ──────────────────────────────────────────
        if _has_bus:
            with status_lock:
                agent_status["cycle_count"]   = cycle_num
                agent_status["last_decision"] = final_state.get("decision")
                agent_status["last_ticker"]   = final_state.get("ticker")

        # ── Notifica automatica BUY/SELL al bot ───────────────────────────
        if _has_bus:
            dec = final_state.get("decision")
            if dec in ("BUY", "SELL"):
                from command_bus import rep_queue as _rep_q
                # Notifica tutti i chat_id autorizzati (separati da virgola nel .env)
                _raw_ids = os.environ.get("TELEGRAM_CHAT_ID", "0")
                _all_ids = [
                    int(cid.strip()) for cid in _raw_ids.split(",")
                    if cid.strip().isdigit() and int(cid.strip()) != 0
                ]
                tk  = final_state.get("ticker", "?")
                qty = final_state.get("quantity", 0)
                rat = (final_state.get("rationale") or "")[:200]
                for chat_id in _all_ids:
                    _rep_q.put({
                        "chat_id": chat_id,
                        "text": (
                            f"{'🟢' if dec == 'BUY' else '🔴'} *{dec}* `{tk}` x{qty}\n"
                            f"_{rat}_"
                        ),
                    })

        # ── Pausa tra cicli ────────────────────────────────────────────────
        remaining_after = end_time - time.time()
        if remaining_after <= 0:
            print(f"\n⏰ Tempo scaduto dopo ciclo #{cycle_num}.")
            break
        if remaining_after < CYCLE_INTERVAL_SEC:
            print(f"\n⏰ Tempo rimanente ({remaining_after:.0f}s) < intervallo. Stop.")
            break
        print(f"\n⏳ Prossimo ciclo tra {CYCLE_INTERVAL_SEC}s... (Ctrl+C per interrompere)")
        time.sleep(CYCLE_INTERVAL_SEC)

    # ── Cleanup ────────────────────────────────────────────────────────────
    if _has_bus:
        with status_lock:
            agent_status["running"] = False
        stop_flag.set()  # segnala al bot che l'agente è terminato

    total = time.time() - start_time
    print(f"\n✅ Sessione terminata — {cycle_num} cicli in {total:.0f}s.")
    print(f"   Ticker analizzati: {session_analyzed}")


# ---------------------------------------------------------------------------
# Entry point standalone (senza bot Telegram)
# ---------------------------------------------------------------------------

def main() -> None:
    run_agent_loop()


if __name__ == "__main__":
    main()