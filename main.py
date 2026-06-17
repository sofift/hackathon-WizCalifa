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
from journal import init_watchlist, add_to_watchlist, remove_from_watchlist, get_watchlist
from agent import build_graph
from state import AgentState
from tools import get_portfolio, place_order, get_position_qty

# ---------------------------------------------------------------------------
# Configurazione temporale
# ---------------------------------------------------------------------------

RUN_DURATION_SEC   = 3600  # Durata totale della sessione (secondi) - impostata a 1 ora
CYCLE_INTERVAL_SEC = 3     # Pausa minima tra un ciclo e il successivo

# ---------------------------------------------------------------------------
# Elaborazione comandi Telegram (cmd_queue → rep_queue)
# ---------------------------------------------------------------------------

def _format_report(portfolio: dict, chat_id: int | None = None) -> str:
    """
    Formatta il report per Telegram con TRE sezioni distinte:
      1. Posizioni REALMENTE aperte (filled) — da get_portfolio()
      2. Ordini SOSPESI/IN ATTESA (inviati ma non eseguiti) — da get_open_orders()
         con motivo esplicito quando il mercato e' chiuso
      3. Ultime decisioni registrate nel journal (con avviso se ancora pendenti)
    """
    from tools import get_open_orders

    lines = ["📊 *REPORT AGENTE*\n"]

    # Sezione 1: posizioni reali aperte
    if "error" in portfolio:
        lines.append(f"⚠️ Portfolio non disponibile: {portfolio['error']}")
        open_orders_res = {"orders": []}
    else:
        cash = portfolio.get("cash", 0)
        pval = portfolio.get("portfolio_value", 0)
        positions = portfolio.get("positions", [])
        lines.append(f"💰 Cash: `${cash:,.2f}`")
        lines.append(f"📈 Valore totale: `${pval:,.2f}`")
        watchlist = set(get_watchlist(chat_id))
        if positions:
            lines.append("\n*Posizioni aperte:*")
            for p in positions:
                profit = p.get("profit_pct", 0)
                emoji  = "📈" if profit >= 0 else "📉"
                lock   = " 🔒" if p["ticker"].upper() in watchlist else ""
                lines.append(
                    f"  {emoji} `{p['ticker']}`{lock} — qty: `{p['qty']}` | "
                    f"valore: `${p['market_value']:,.2f}` | P/L: `{profit:+.1f}%`"
                )
        else:
            lines.append("\n📭 Nessuna posizione aperta.")

        open_orders_res = get_open_orders(user_chat_id=chat_id)

    # Sezione 2: ordini sospesi / in attesa
    if "error" in open_orders_res:
        lines.append(f"\n⚠️ Ordini in attesa non verificabili: {open_orders_res['error']}")
    else:
        pending = open_orders_res.get("orders", [])
        if pending:
            lines.append("\n*⏳ Ordini in attesa (non ancora eseguiti):*")
            for o in pending:
                side_emoji = "🟢" if o["side"] == "buy" else "🔴"
                if o.get("market_closed"):
                    motivo = "mercato chiuso — verrà eseguito all'apertura"
                elif o["filled_qty"] > 0:
                    motivo = f"parzialmente eseguito ({o['filled_qty']}/{o['qty']})"
                else:
                    motivo = f"in coda (status: {o['status']})"
                lines.append(
                    f"  {side_emoji} `{o['ticker']}` {o['side'].upper()} x{o['qty']} — _{motivo}_"
                )
        else:
            lines.append("\n✅ Nessun ordine in sospeso: tutto eseguito.")

    # Sezione 3: ultime decisioni dal journal con note di stato
    pending_tickers = set()
    if "error" not in open_orders_res:
        pending_tickers = {o["ticker"].upper() for o in open_orders_res.get("orders", [])}

    decisions = get_recent_decisions(limit=5, chat_id=chat_id)
    if decisions:
        lines.append("\n*Ultime decisioni:*")
        for d in decisions:
            dec_emoji = "🟢" if d["decision"] == "BUY" else "🔴" if d["decision"] == "SELL" else "⚪"
            price_str = f"@ ${d['price']:.2f}" if d["price"] else "@ N/D"
            nota = ""
            outcome = (d.get("outcome") or "").lower()
            if "mercato_chiuso" in outcome:
                nota = " ⏸️ _(in attesa: mercato chiuso)_"
            elif d["ticker"].upper() in pending_tickers and d["decision"] in ("BUY", "SELL"):
                nota = " ⏳ _(ordine non ancora eseguito)_"
            lines.append(
                f"  {dec_emoji} `{d['ticker']}` {d['decision']} x{d['quantity']} "
                f"{price_str} — {d['timestamp'][:16]}{nota}"
            )

    return "\n".join(lines)


def _process_commands(live_portfolio: dict, chat_id: int) -> dict:
    """
    Svuota la cmd_queue ed esegue ogni comando ricevuto dal bot Telegram.
    Mette i risultati in rep_queue per l'invio all'utente.
    Se è un comando per settore, restituisce un dizionario con gli intenti da passare al LangGraph.
    Chiamata all'inizio di ogni ciclo, PRIMA di invocare il grafo.
    """
    forced_state = {}
    try:
        from command_bus import get_cmd_queue, rep_queue
        cmd_queue = get_cmd_queue(chat_id)
    except ImportError:
        return forced_state  # command_bus non disponibile (modalità standalone senza bot)

    while not cmd_queue.empty():
        try:
            cmd     = cmd_queue.get_nowait()
            action  = cmd.get("action", "")
            target  = cmd.get("target")

            print(f"\n[main] 📨 Comando Telegram ricevuto: action={action} target={target}")

            # ── BUY_TICKER / SELL_TICKER / BUY_SECTOR / SELL_SECTOR: delega a LangGraph ──────────────────
            if action in ("buy_sector", "sell_sector", "buy_ticker", "sell_ticker"):
                from agent import resolve_ticker
                
                if action.endswith("_sector"):
                    forced_state["forced_action"] = "BUY" if action == "buy_sector" else "SELL"
                    forced_state["forced_target"] = target
                    forced_state["forced_type"]   = "sector"
                else:
                    forced_state["forced_action"] = "BUY" if action == "buy_ticker" else "SELL"
                    forced_state["forced_target"] = resolve_ticker(target, chat_id)
                    forced_state["forced_type"]   = "ticker"
                forced_state["forced_percentage"] = cmd.get("forced_percentage")
                forced_state["forced_chat_id"]    = chat_id
                break  # Essenziale per passare un solo comando al grafo per ciclo

            # ── SELL_ALL: vendi tutte le posizioni ────────────────────────
            elif action == "sell_all":
                # Usa il portfolio PERSONALE dell'utente
                user_portfolio = get_portfolio(user_chat_id=chat_id)
                if "error" in user_portfolio:
                    rep_queue.put({"chat_id": chat_id, "text": f"❌ Errore portfolio: {user_portfolio['error']}"})
                    continue
                positions = user_portfolio.get("positions", [])
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
                        res = place_order(t, "sell", qty, user_chat_id=chat_id)
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
                                chat_id=chat_id,
                            )
                            remove_from_watchlist(t, chat_id=chat_id)
                            lines.append(f"  ✅ `{t}` x{qty} — `{res['order_id']}`")
                    rep_queue.put({"chat_id": chat_id, "text": "\n".join(lines)})

            # ── BALANCE: bilancia il portafoglio su nuovi settori ─────────────
            elif action == "balance":
                from agent import get_llm, safe_invoke
                import json
                
                user_portfolio = get_portfolio(user_chat_id=chat_id)
                if "error" in user_portfolio:
                    rep_queue.put({"chat_id": chat_id, "text": f"❌ Errore portfolio: {user_portfolio['error']}"})
                    continue
                
                positions = user_portfolio.get("positions", [])
                pval = user_portfolio.get("portfolio_value", 1.0)
                
                if not positions:
                    rep_queue.put({"chat_id": chat_id, "text": "⚠️ Portafoglio vuoto. Impossibile bilanciare settori esistenti. Usa /compra settore per iniziare."})
                    continue
                
                pos_info = []
                for p in positions:
                    weight = p['market_value'] / pval
                    pos_info.append(f"{p['ticker']} ({weight*100:.1f}%)")
                
                pos_str = ", ".join(pos_info)
                
                llm = get_llm("fast", chat_id)
                prompt = f"""Sei un Portfolio Manager. L'utente ha questo portafoglio: {pos_str}.
Il tuo compito è bilanciare il portafoglio diversificandolo su ALTRI settori macroeconomici non presenti o sotto-rappresentati.
1. Analizza i settori dei ticker attuali e la loro percentuale.
2. Scegli 2 o 3 nuovi settori macro (es. healthcare, energy, financials, utilities, materials) che mancano per bilanciarlo.
3. Assegna a ciascun nuovo settore una percentuale del capitale totale (in decimale, es. 0.15) proporzionata ai pesi già esistenti in modo da raggiungere una maggiore diversificazione.
Restituisci SOLO ed ESCLUSIVAMENTE un array JSON in questo esatto formato:
[
  {{"sector": "healthcare", "percentage": 0.15}},
  {{"sector": "utilities", "percentage": 0.10}}
]
Non aggiungere testo prima o dopo il JSON."""
                try:
                    response = safe_invoke(llm, prompt)
                    content = response.content.strip()
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    content = content.strip()
                    
                    sectors_to_buy = json.loads(content)
                    
                    lines = ["⚖️ *Bilanciamento Portafoglio*"]
                    lines.append(f"Attuale: {pos_str}")
                    lines.append("\nNuovi settori da acquistare:")
                    
                    for item in sectors_to_buy:
                        sec = item.get("sector")
                        perc = item.get("percentage")
                        if sec and perc:
                            cmd_queue.put({
                                "action": "buy_sector",
                                "target": sec,
                                "forced_percentage": float(perc),
                                "chat_id": chat_id
                            })
                            lines.append(f"  • *{sec}*: {float(perc)*100:.1f}%")
                            
                    rep_queue.put({"chat_id": chat_id, "text": "\n".join(lines)})
                except Exception as e:
                    rep_queue.put({"chat_id": chat_id, "text": f"❌ Errore nel calcolo del bilanciamento: {e}\nRisposta LLM: {content}"})

            # ── REPORT: portfolio + journal ───────────────────────────────
            elif action == "report":
                # Usa il portfolio PERSONALE dell'utente
                fresh_portfolio = get_portfolio(user_chat_id=chat_id)
                report_text = _format_report(fresh_portfolio, chat_id=chat_id)
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

def run_agent_loop(chat_id: int | None = None) -> None:
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

    init_journal(chat_id)
    init_watchlist(chat_id)
    graph = build_graph()

    # Snapshot iniziale
    print(f"\n[main|{chat_id}] Lettura portafoglio Alpaca iniziale...")
    initial_portfolio = get_portfolio(user_chat_id=chat_id)
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
        from command_bus import get_agent_status, status_lock, stop_flag
        agent_status = get_agent_status(chat_id or 0)
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
            print(f"\n[main|{chat_id}] 🛑 Stop flag ricevuto dal bot — termino il loop.")
            break

        cycle_num += 1
        elapsed   = time.time() - start_time
        remaining = end_time - time.time()
        print(f"\n{'=' * 55}")
        print(f"🚀 CICLO #{cycle_num}  |  Trascorso: {elapsed:.0f}s  |  Rimanente: {remaining:.0f}s")
        print(f"{'=' * 55}")

        # ── Portfolio live ─────────────────────────────────────────────────
        live_portfolio = get_portfolio(user_chat_id=chat_id)
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
        forced_state = _process_commands(live_portfolio if "error" not in live_portfolio else {}, chat_id or 0)

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
            "forced_action":        forced_state.get("forced_action"),
            "forced_target":        forced_state.get("forced_target"),
            "forced_type":          forced_state.get("forced_type"),
            "forced_percentage":    forced_state.get("forced_percentage"),
            "chat_id":              chat_id,
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