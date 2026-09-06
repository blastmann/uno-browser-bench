#!/usr/bin/env python3
"""Score model JSON outputs against deterministic gold labels."""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

def extract_json(text: str):
    if not isinstance(text, str): return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I|re.S).strip()
    try: return json.loads(text)
    except Exception: pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return None

def norm(v): return re.sub(r"\s+", " ", str(v).strip().lower())
def flat_entities(x):
    out=[]
    for e in (x or []):
        if isinstance(e, str): out.append(norm(e))
        elif isinstance(e, dict):
            for k in ("name", "type"):
                if e.get(k): out.append(norm(e[k]))
            for v in (e.get("attributes") or {}).values(): out.append(norm(v))
    return set(out)
def f1(p,r):
    return 0.0 if p+r == 0 else 2*p*r/(p+r)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    rows=[json.loads(x) for x in a.input.read_text(encoding="utf-8").splitlines() if x.strip()]
    stats=Counter(); domain=defaultdict(lambda: Counter()); lengths=defaultdict(lambda: Counter()); entity_f1=[]; step_f1=[]
    for row in rows:
        pred=extract_json(row.get("output","")); gold=row.get("gold",{}); stats["total"]+=1
        if pred is not None: stats["json_valid"]+=1
        is_entity=row.get("task")=="entity"
        required=("page_type","entities") if is_entity else ("intent","entities","steps")
        valid=isinstance(pred,dict) and all(k in pred for k in required)
        if valid: stats["schema_valid"]+=1
        d=domain[row.get("domain","unknown")]; d["total"]+=1; d["json_valid"]+=pred is not None; d["schema_valid"]+=valid
        if is_entity and isinstance(pred,dict):
            g=flat_entities(gold.get("entities")); p=flat_entities(pred.get("entities")); tp=len(g&p); pp=tp/(len(p) or 1); rr=tp/(len(g) or 1); entity_f1.append(f1(pp,rr)); stats["field_exact"] += norm(pred.get("page_type"))==norm(gold.get("page_type")); d["entity_f1_sum"] += f1(pp,rr)
        if not is_entity and isinstance(pred,dict):
            stats["intent_correct"] += norm(pred.get("intent"))==norm(gold.get("intent")); d["intent_correct"] += norm(pred.get("intent"))==norm(gold.get("intent")); d["intent_total"] += 1
            g=set(map(norm,gold.get("steps",[]))); p=set(map(norm,pred.get("steps",[]))); tp=len(g&p); step_f1.append(f1(tp/(len(p) or 1),tp/(len(g) or 1))); d["step_f1_sum"] += step_f1[-1]
        lengths[row.get("trajectory_length",0)]["total"] += 1; lengths[row.get("trajectory_length",0)]["json_valid"] += pred is not None
    out={"input":str(a.input),"counts":dict(stats),"rates":{k:stats[k]/stats["total"] for k in ("json_valid","schema_valid")},"entity_f1":sum(entity_f1)/len(entity_f1) if entity_f1 else None,"intent_accuracy":stats["intent_correct"]/sum(1 for r in rows if r.get("task")=="trajectory") if any(r.get("task")=="trajectory" for r in rows) else None,"step_f1":sum(step_f1)/len(step_f1) if step_f1 else None,"by_domain":{k:dict(v) for k,v in domain.items()},"by_trajectory_length":{str(k):dict(v) for k,v in sorted(lengths.items())}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(out,ensure_ascii=False))
if __name__=="__main__": main()
