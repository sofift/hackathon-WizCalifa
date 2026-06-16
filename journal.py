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
    rows = conn.execute("SELECT * FROM journal ORDER BY id DESC LIMIT 20").fetchall()
    conn.close()

    print("\n===== TRADE JOURNAL (ultimi 20) =====")
    for row in rows:
        print(
            f"[{row[1]}] {row[2]} | {row[4]} x{row[5]} @ {row[3]} "
            f"| {row[6]} | order={row[7]} | outcome={row[8]}"
        )
    print("=====================================\n")