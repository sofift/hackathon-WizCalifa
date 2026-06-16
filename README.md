# 🤖 WizCalifa — Autonomous Trading Agent

> **Agentic AI Hackathon** — Trading agent autonomo basato su LangGraph + LLM + Alpaca Paper Trading

---

## 📌 Overview

**WizCalifa** è un agente di trading completamente autonomo che:

- 🔍 **Scansiona** il mercato leggendo il feed di notizie finanziarie in tempo reale
- 🧠 **Ragiona** usando un LLM (Llama 3.1) per analizzare sentiment e scegliere il ticker più promettente
- 📊 **Analizza** il trend tecnico tramite SMA5 (Media Mobile a 5 giorni)
- ⚡ **Esegue** ordini reali su Alpaca Paper Trading
- 💾 **Memorizza** ogni decisione in un journal SQLite
- 🔁 **Impara** dagli esiti passati grazie a una memoria riflessiva

L'agente opera in cicli autonomi senza intervento umano, seguendo una strategia **Momentum + Sentiment** con risk management deterministico.

---

## 🏗️ Architettura

### Grafo LangGraph

```
[START]
   │
   ▼
reflect            ← Valuta gli esiti passati (memoria riflessiva)
   │
   ▼
manage_portfolio   ← Gestisce le posizioni aperte (take-profit / stop-loss / momentum exit / sentiment)
   │
   ▼
scan_market        ← Legge il feed di notizie (50 articoli), conta le citazioni per ticker
   │
   ▼
select_ticker      ← LLM ordina i candidati per promettenza (analisi di mercato)
   │
   ▼
evaluate           ← Per ogni candidato: prezzo + SMA5 + sentiment LLM + regole deterministiche
   │
   ▼
execute_order      ← Invia l'ordine su Alpaca Paper Trading
   │
   ▼
write_journal      ← Registra la decisione nel journal SQLite
   │
   ▼
should_continue? ──── cicli rimasti → torna a reflect
                └──── cicli esauriti → [END]
```

### Struttura dei file

```
hackaton/
│
├── main.py              # Entry point — configura e avvia il grafo
├── agent.py             # Grafo LangGraph con tutti i nodi e la logica
├── state.py             # AgentState — stato condiviso tra i nodi
├── tools.py             # Tool deterministici: prezzi, notizie, ordini, portfolio
├── journal.py           # Persistenza SQLite delle decisioni
├── test_connection.py   # Verifica le credenziali Alpaca prima di avviare
├── requirements.txt     # Dipendenze Python
└── .env                 # Credenziali API (NON committare!)
```

---

## 🧠 Strategia: Momentum + Sentiment

### Acquisto (BUY)

L'agente compra solo se **tutti i segnali sono allineati**:

| Segnale | Fonte | Condizione |
|---|---|---|
| Trend tecnico | SMA5 (deterministico) | Prezzo > Media Mobile 5gg → RIALZISTA |
| Sentiment news | LLM (Llama 3.1) | Notizie positive → RIALZISTA |
| Anti-flip-flop | Portfolio Alpaca | Non possiedo già il ticker |
| Cooldown | Journal SQLite | Non l'ho comprato negli ultimi 60 min |

**Position sizing:**
- Segnale `RIALZISTA (ALTA)` → 15% del cash disponibile
- Segnale `RIALZISTA (BASSA)` → 5% del cash disponibile
- Se il segnale ha un track record negativo → sizing dimezzato (memoria riflessiva)

### Vendita (SELL) — 4 livelli in priorità

| # | Trigger | Condizione | Obiettivo |
|---|---|---|---|
| 🎯 1 | **Take-Profit** | gain ≥ +10% | Massimizzare il profitto |
| 🛑 2 | **Stop-Loss** | loss ≤ -5% | Limitare le perdite |
| 📉 3 | **Momentum Exit** | Trend RIBASSISTA + posizione in profitto | Proteggere i guadagni |
| 🔻 4 | **Sentiment Exit** | Notizie ribassiste (LLM) | Tagliare le perdite |

---

## ⚙️ Configurazione

### Parametri principali in `main.py`

```python
TICKER     = "AAPL"   # (non usato nel nuovo flusso multi-ticker)
MAX_CYCLES = 3        # Numero di cicli da eseguire
WAIT_SEC   = 30       # Pausa tra cicli (se loop esterno)
```

### Parametri strategici in `agent.py`

```python
NEWS_FEED_LIMIT     = 50     # Articoli del feed da analizzare per ciclo
TOP_N_CANDIDATES    = 8      # Ticker candidati passati all'LLM
BUY_COOLDOWN_MINUTES = 60   # Cooldown anti-riacquisto (minuti)
STOP_LOSS_PCT       = 0.05   # -5%  → stop-loss automatico
TAKE_PROFIT_PCT     = 0.10   # +10% → take-profit automatico
ALLOW_CRYPTO        = True   # Includi crypto nel feed
```

---

## 🚀 Setup e Avvio

### 1. Prerequisiti

- Python 3.11+
- Account Alpaca Paper Trading ([alpaca.markets](https://alpaca.markets))
- API Key Groq (per Llama 3.1)

### 2. Installazione dipendenze

```bash
pip install -r requirements.txt
```

### 3. Configurazione `.env`

Crea un file `.env` nella root del progetto:

```env
ALPACA_API_KEY=il_tuo_api_key_paper
ALPACA_SECRET_KEY=il_tuo_secret_key_paper
GROQ_API_KEY=il_tuo_groq_api_key
```

> ⚠️ Usa **le chiavi PAPER** di Alpaca, non quelle live. Nessun soldo reale viene utilizzato.

### 4. Verifica la connessione

```bash
python test_connection.py
```

Output atteso:
```
=============================================
  CONNESSIONE ALPACA OK
=============================================
  Account status : ACTIVE
  Cash           : 100000.0 USD
  ...
```

### 5. Avvia l'agente

```bash
python main.py
```

---

## 📋 Journal delle decisioni

Ogni decisione viene registrata in `trade_journal.db` (SQLite). Lo schema:

| Campo | Tipo | Descrizione |
|---|---|---|
| `timestamp` | TEXT | Data e ora UTC della decisione |
| `ticker` | TEXT | Simbolo del titolo |
| `price` | REAL | Prezzo al momento della decisione |
| `decision` | TEXT | BUY / SELL / HOLD |
| `quantity` | INTEGER | Numero di azioni/crypto |
| `rationale` | TEXT | Motivazione completa (sentiment + regola) |
| `order_id` | TEXT | ID ordine Alpaca (se eseguito) |
| `outcome` | TEXT | ok / price_error / order_error |
| `sentiment` | TEXT | Classificazione sentiment LLM |

---

## 🛠️ Stack Tecnologico

| Componente | Tecnologia | Versione |
|---|---|---|
| Orchestrazione agente | LangGraph | 1.2.5 |
| LLM | Llama 3.1 8B (via Groq) | - |
| Broker / Dati | Alpaca Paper Trading | alpaca-py 0.43.4 |
| Database decisioni | SQLite | built-in |
| Gestione env | python-dotenv | 1.2.2 |

---

## 👥 Team

**WizCalifa** — Agentic AI Hackathon 2026

---

## ⚠️ Disclaimer

Questo progetto è sviluppato esclusivamente per scopi accademici e di hackathon.  
Tutte le operazioni avvengono su **Alpaca Paper Trading** (simulazione).  
Nessun denaro reale viene movimentato.
