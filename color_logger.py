"""
color_logger.py
Aggiunge colore e prefisso utente (chat_id) ad ogni chiamata di print()
a seconda del thread che la esegue, sfruttando threading.local().
"""
import builtins
import threading

_local = threading.local()
_original_print = builtins.print
_print_lock = threading.Lock()

COLORS = [
    "\033[96m", # Ciano
    "\033[92m", # Verde
    "\033[93m", # Giallo
    "\033[95m", # Magenta
    "\033[94m", # Blu
]
RESET = "\033[0m"

_user_colors = {}
_color_idx = 0

def set_thread_chat_id(chat_id: int):
    """Assegna il chat_id al thread corrente."""
    _local.chat_id = chat_id

def colorized_print(*args, **kwargs):
    chat_id = getattr(_local, 'chat_id', None)
    if chat_id is None:
        _original_print(*args, **kwargs)
        return

    with _print_lock:
        global _color_idx
        if chat_id not in _user_colors:
            _user_colors[chat_id] = COLORS[_color_idx % len(COLORS)]
            _color_idx += 1
            
        color = _user_colors[chat_id]
        prefix = f"{color}[Utente {chat_id}]{RESET} "
        
        sep = kwargs.get('sep', ' ')
        msg = sep.join(str(a) for a in args)
        
        lines = msg.split('\n')
        out_lines = []
        for line in lines:
            if line.strip() == "" and not line.startswith(" "):
                # Mantieni righe completamente vuote pulite
                out_lines.append(line)
            else:
                out_lines.append(f"{prefix}{line}")
                
        out_str = '\n'.join(out_lines)
        
        kwargs_copy = {k: v for k, v in kwargs.items() if k != 'sep'}
        _original_print(out_str, **kwargs_copy)

def setup_logger():
    """Sostituisce builtins.print con la nostra versione colorata."""
    builtins.print = colorized_print
