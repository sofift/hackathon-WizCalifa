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
            outcome     TEXT
        )
    """)
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
):
    """Inserisce una riga nel journal."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO journal (timestamp, ticker, price, decision, quantity, rationale, order_id, outcome)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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