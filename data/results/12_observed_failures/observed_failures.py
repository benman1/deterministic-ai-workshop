"""Observed failure types of a small local LLM used as the slow tier: free generation of JSON, no constrained decoding.
usage: python observed_failures.py N OUT.jsonl MODEL"""
import json, re, sys, time, torch
from pathlib import Path
import meridian as m
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(4)
N, OUT, MODEL = int(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in Path("data/llm/test_inputs.jsonl").read_text().splitlines()][:N]
idmap = json.loads(Path("data/llm/test_id_map.json").read_text())
te = {r["id"]: r for r in m.load("test")}
tok = AutoTokenizer.from_pretrained(MODEL); mod = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
SYS = ("You route customer messages for a travel company. Reply with JSON only, in the form {\"route\": \"<route>\"}. "
       "Allowed routes: " + ", ".join(m.ROUTES) + ".")
out = open(OUT, "a"); tally = {}
for k, r in enumerate(rows):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["text"]}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    t = time.time()
    with torch.no_grad():
        g = mod.generate(**tok(p, return_tensors="pt"), max_new_tokens=40, do_sample=False)
    txt = tok.decode(g[0][len(tok(p)["input_ids"]):], skip_special_tokens=True).strip()
    sec = time.time() - t
    try:
        route = json.loads(txt)["route"]; kind = "valid" if route in m.ROUTES else "label_outside_set"
    except Exception:
        mm = re.search(r'"route"\s*:\s*"([^"]*)"', txt); route = mm.group(1) if mm else None
        kind = "malformed_json_but_label_found" if (route in m.ROUTES) else ("malformed_json" if txt else "empty")
    true = te[idmap[r["id"]]]["label"] if idmap[r["id"]] in te else None
    tally[kind] = tally.get(kind, 0) + 1
    out.write(json.dumps(dict(i=k, id=r["id"], kind=kind, route=route, true=true, sec=round(sec, 2), raw=txt[:120])) + "\n"); out.flush()
    print(f"{k+1}/{len(rows)} {kind:28s} sec {sec:5.1f} tally {tally}", flush=True)
