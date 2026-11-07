# Tune the fine-tuning schedule on a validation split of the TRAINING pool (never the test set), then rerun the notebook-18b comparison with the chosen schedule.
import os, json, math, time, csv, urllib.request, itertools, sys, datetime
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)
DATA = Path("banking77"); DATA.mkdir(exist_ok=True)
BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
for s in ("train", "test"):
    if not (DATA / f"{s}.csv").exists(): urllib.request.urlretrieve(BASE + f"{s}.csv", DATA / f"{s}.csv")
tr = list(csv.DictReader(open(DATA / "train.csv"))); te = list(csv.DictReader(open(DATA / "test.csv")))
classes = sorted({r["category"] for r in tr}); cid = {c: i for i, c in enumerate(classes)}; K = len(classes)
ytr = np.array([cid[r["category"]] for r in tr]); yte = np.array([cid[r["category"]] for r in te])
REPO = "https://raw.githubusercontent.com/benman1/deterministic-ai-workshop/main/data/results/18_distillation/"
LAB = Path("labels"); LAB.mkdir(exist_ok=True)
for f in ["sample_idx.json", "teacher_labels.jsonl", "teacher_labels_opus.jsonl"]:
    urllib.request.urlretrieve(REPO + f, LAB / f)
idx = np.array(json.load(open(LAB / "sample_idx.json")))
def labels(fname):
    T = {r["id"]: r["label"] for r in map(json.loads, open(LAB / fname))}
    return np.array([cid.get(T[i], -1) for i in range(len(idx))])
gold = ytr[idx]; haiku = labels("teacher_labels.jsonl"); opus = labels("teacher_labels_opus.jsonl")
MODEL = "BAAI/bge-small-en-v1.5"; dev = "cuda"; tok = AutoTokenizer.from_pretrained(MODEL); BS, MAXLEN = 32, 64
def enc(t): return tok(t, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
# validation: 1000 training messages that are NOT in the 2,000 sample
rest = np.setdiff1d(np.arange(len(tr)), idx); val_idx = np.random.default_rng(7).permutation(rest)[:1000]
val_texts = [tr[i]["text"] for i in val_idx]; y_val = ytr[val_idx]
sub = np.random.default_rng(123).permutation(len(te)); test_texts = [te[i]["text"] for i in sub]; y_test = yte[sub]
def ece(P, y, bins=10):
    conf = P.max(1); ok = P.argmax(1) == y; tot = 0
    for b in range(bins):
        i = (conf >= b / bins) & ((conf < (b + 1) / bins) | (b == bins - 1))
        if i.any(): tot += i.sum() * abs(conf[i].mean() - ok[i].mean())
    return tot / len(y)
def finetune(texts, y, seed, lr, epochs, eval_texts):
    torch.manual_seed(seed); np.random.seed(seed)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(texts) / BS); sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda"); keep = np.where(y >= 0)[0]
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(keep)
        for b in range(0, len(perm), BS):
            ii = perm[b:b + BS]; x = enc([texts[i] for i in ii]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(m(**x).logits.float(), torch.tensor(y[ii]).to(dev))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    m.eval(); P = []
    with torch.no_grad():
        for b in range(0, len(eval_texts), 128):
            P.append(torch.softmax(m(**enc(eval_texts[b:b + 128]).to(dev)).logits.float(), -1).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(P)
texts_all = [tr[i]["text"] for i in idx]; t0 = time.time(); out = {"grid": {}, "final": {}}
# stage 1: grid on validation, human labels, N=2000, seed 0
for lr, ep in itertools.product((1e-4, 3e-4, 1e-3), (12, 30)):
    P = finetune(texts_all, gold, 0, lr, ep, val_texts)
    acc = float((P.argmax(1) == y_val).mean()); e = float(ece(P, y_val)); conf = float(P.max(1).mean())
    out["grid"][f"{lr}|{ep}"] = dict(val_acc=acc, val_ece=e, mean_conf=conf)
    print(f"GRID lr {lr:g} epochs {ep:2d}: validation accuracy {acc:.3f}  ECE {e:.3f}  mean confidence {conf:.3f}  [{time.time()-t0:5.0f} s]", flush=True)
best = max(out["grid"], key=lambda k: out["grid"][k]["val_acc"]); BLR, BEP = float(best.split("|")[0]), int(best.split("|")[1]); out["chosen"] = dict(lr=BLR, epochs=BEP)
print(f"CHOSEN by validation accuracy: lr {BLR:g}, epochs {BEP}", flush=True)
# stage 2: the notebook-18b comparison on the test set with the chosen schedule
sources = {"human": gold, "Haiku": haiku, "Opus": opus}
for N in (500, 2000):
    for seed in (0, 1, 2):
        sel = np.random.default_rng(seed).permutation(len(idx))[:N]
        for name, lab in sources.items():
            P = finetune([texts_all[i] for i in sel], lab[sel], seed, BLR, BEP, test_texts)
            acc = float((P.argmax(1) == y_test).mean()); e = float(ece(P, y_test)); conf = float(P.max(1).mean())
            out["final"][f"{N}|{name}|{seed}"] = dict(acc=acc, ece=e, mean_conf=conf)
            print(f"FINAL N={N:5d} {name:6s} seed {seed}: accuracy {acc:.3f}  ECE {e:.3f}  mean confidence {conf:.3f}  [{time.time()-t0:5.0f} s]", flush=True)
json.dump(out, open("18b_tuned_results.json", "w"), indent=1)
json.dump(dict(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), torch=torch.__version__, transformers=transformers.__version__,
               gpu=torch.cuda.get_device_name(0), model=MODEL, encoder_revision=None, grid="lr {1e-4,3e-4,1e-3} x epochs {12,30}", selection="validation accuracy, 1000 training messages outside the 2000 sample, seed 0, human labels"), open("run_manifest.json", "w"), indent=1)
print("done", flush=True)
