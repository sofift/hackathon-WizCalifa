import sqlite3
import datetime

DB_PATH = "trade_journal.db"


def init_journal():
    """Crea la tabella se non esiste ancora."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS journal (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            ticker      TEXT    NOT NULL,
            price       REAL,
            decision    TEXT    NOT NULL,
            quantity    INTEGER,
            rationale   TEXT,
            order_id    TEXT,
            outcome     TEXT,
            sentiment   TEXT
        )
    """)
    # Aggiunge colonna sentiment se la tabella esisteva già (migrazione non distruttiva)
    try:
        conn.execute("ALTER TABLE journal ADD COLUMN sentiment TEXT")
    except Exception:
        pass  # colonna già presente
    conn.commit()
    conn.close()


def log_decision(
    ticker: str,
    price: float | None,
    decision: str,
    quantity: int | None,
    rationale: str | None,
    order_id: str | None = None,
    outcome: str | None = None,
    sentiment: str | None = None,
):
    """Inserisce una riga nel journal."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO journal (timestamp, ticker, price, decision, quantity, rationale, order_id, outcome, sentiment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.datetime.utcnow().isoformat(),
            ticker,
            price,
            decision,
            quantity,
            rationale,
            order_id,
            outcome,
            sentiment,
        ),
    )
    conn.commit()
    conn.close()


def print_journal():
    """Stampa il journal in modo leggibile (utile per il demo)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM journal ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()

    print("\n" + "═" * 70)
    print(" 📖 TRADE JOURNAL (ultime 10 operazioni)")
    print("═" * 70)
    for row in rows:
        # Formattazione pulita dei campi
        timestamp = row[1][:19].replace("T", " ") # Rimuove millisecondi
        ticker    = f"{row[2]:<8}"
        decision  = row[4]
        azione    = f"{decision} x{row[5]}"
        prezzo    = f"@ {row[3]:<8}" if row[3] else "@ N/A     "
        outcome   = row[8] if row[8] else "ok"
        
        # Aggiunta colori basilari per il terminale
        color = "\033[92m" if decision == "BUY" else "\033[91m" if decision == "SELL" else "\033[93m"
        reset = "\033[0m"
        
        print(f"  [{timestamp}] {ticker} | {color}{azione:<12}{reset} {prezzo} | Status: {outcome}")
    print("═" * 70 + "\n")


def get_recent_decisions(limit: int = 5) -> list[dict]:
    """
    Restituisce le ultime N decisioni dal journal come lista di dict.
    Usata dal bot Telegram per il comando /report.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, ticker, decision, quantity, price, outcome, sentiment "
        "FROM journal ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {
            "timestamp": row[0][:19].replace("T", " "),
            "ticker":    row[1],
            "decision":  row[2],
            "quantity":  row[3],
            "price":     row[4],
            "outcome":   row[5] or "ok",
            "sentiment": row[6] or "—",
        }
        for row in rows
    ]


def minutes_since_last_buy(ticker: str) -> float | None:
    """
    Restituisce i minuti trascorsi dall'ultimo BUY su ticker, o None se non trovato.
    Usata dall'agente per il cooldown anti-riacquisto.
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT timestamp FROM journal WHERE ticker=? AND decision='BUY' ORDER BY id DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    try:
        last_buy = datetime.datetime.fromisoformat(row[0])
        delta = datetime.datetime.utcnow() - last_buy
        return delta.total_seconds() / 60
    except Exception:
        return None


def reflect_on_past(current_prices: dict) -> dict:
    """
    Analizza gli esiti passati e restituisce un riassunto testuale e
    una mappa di affidabilità per-segnale.
    Usata dal nodo 'reflect' del grafo.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ticker, decision, price, sentiment, outcome FROM journal "
        "WHERE decision IN ('BUY','SELL') ORDER BY id DESC LIMIT 30"
    ).fetchall()
    conn.close()

    if not rows:
        return {"summary": "Nessuna decisione passata da analizzare.", "signal_reliability": {}}

    signal_counts: dict[str, dict] = {}  # sentiment -> {"ok": int, "total": int}
    lines = []

    for ticker, decision, entry_price, sentiment, outcome in rows:
        cur_price = current_prices.get(ticker)
        if cur_price is None or entry_price is None:
            continue
        pnl_pct = (cur_price - entry_price) / entry_price * 100
        success = pnl_pct > 0 if decision == "BUY" else pnl_pct < 0
        sig = sentiment or "NEUTRO"
        if sig not in signal_counts:
            signal_counts[sig] = {"ok": 0, "total": 0}
        signal_counts[sig]["total"] += 1
        if success:
            signal_counts[sig]["ok"] += 1
        lines.append(
            f"  {decision} {ticker} @ {entry_price:.2f} → ora {cur_price:.2f} "
            f"({pnl_pct:+.1f}%) [{sig}] → {'✅' if success else '❌'}"
        )

    reliability = {
        sig: (v["ok"] / v["total"]) if v["total"] > 0 else 0.5
        for sig, v in signal_counts.items()
    }
    summary = "Track record recente:\n" + "\n".join(lines[:10])
    return {"summary": summary, "signal_reliability": reliability}