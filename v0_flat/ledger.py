import json
from pathlib import Path
from datetime import datetime
from config_final import LEDGER_PATH, HYPOTHESIS_BUDGET
def _ensure_ledger():
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_PATH.exists(): LEDGER_PATH.touch()
def get_ledger_count():
    _ensure_ledger()
    count=0
    try:
        with open(LEDGER_PATH, "r") as f:
            for line in f:
                if line.strip(): count+=1
    except FileNotFoundError:
        count=0
    return count
def check_budget(n_new=1):
    count=get_ledger_count()
    return HYPOTHESIS_BUDGET - count >= n_new, HYPOTHESIS_BUDGET - count, count
def append_hypotheses(hypotheses):
    _ensure_ledger()
    can, rem, used = check_budget(len(hypotheses))
    if not can: raise RuntimeError(f"Budget exceeded: used {used}/{HYPOTHESIS_BUDGET}")
    with open(LEDGER_PATH, "a") as f:
        for h in hypotheses:
            entry={"timestamp": datetime.utcnow().isoformat(), "ledger_id": used+1, **h}
            f.write(json.dumps(entry)+"\n"); used+=1
    return used
def clear_ledger():
    if LEDGER_PATH.exists(): LEDGER_PATH.unlink()
    LEDGER_PATH.touch(); return 0
def load_ledger():
    _ensure_ledger(); entries=[]
    with open(LEDGER_PATH, "r") as f:
        for line in f:
            if line.strip():
                try: entries.append(json.loads(line))
                except: continue
    return entries
