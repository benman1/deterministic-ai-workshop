"""Does calibrating the Banking77 student on LLM labels give an honest confidence gate?
Student: LogisticRegression(C=10) on frozen bge-small embeddings, trained on 1,600 LLM-labeled messages.
Temperature fitted on (i) 400 held-out LLM-labeled messages, (ii) 200 held-out human-labeled messages. Threshold = smallest score whose
accuracy on the calibration set (its own labels) reaches 95%. Report on the 3,080 gold test messages: decided share and accuracy of decided.
usage: python exp_calibrated.py"""
import sys, json, numpy as np
sys.argv = ['x', 'none']
exec(open('exp_b77.py').read().split("mode = sys.argv[1]")[0])
from scipy.optimize import minimize_scalar
T = {}
for k in range(5):
    for l in open(D/'teacher'/f'out_{k}.jsonl'): r = json.loads(l); T[r['id']] = r['label']
tl = np.array([cid.get(T[i], -1) for i in range(2000)]); idx = np.array(sample); gold = ytr[idx]
def temp_fit(logit, y):
    f = lambda t: -np.mean(np.log(np.exp(logit/t - np.log(np.exp(logit/t).sum(1, keepdims=True)))[np.arange(len(y)), y] + 1e-12))
    return minimize_scalar(f, bounds=(.05, 20), method='bounded').x
def softmax(l): e = np.exp(l - l.max(1, keepdims=True)); return e/e.sum(1, keepdims=True)
def thresh(conf, ok, target=.95):
    for t in np.sort(np.unique(conf)):
        i = conf >= t
        if i.sum() >= 20 and ok[i].mean() >= target: return t
    return 1.01
rows = []
for seed in range(5):
    rng = np.random.default_rng(seed); perm = rng.permutation(2000)
    tr, cal = perm[:1600], perm[1600:]            # cal = 400 held-out
    cal_g = cal[:200]                              # 200 of those also have a human label
    m = fit(Etr[idx[tr]], tl[tr])
    ok_t = None
    L = lambda E: np.log(proba(m, E) + 1e-12)
    Pte = proba(m, Ete); okte = Pte.argmax(1) == yte
    out = dict(seed=seed, acc=float(okte.mean()))
    for name, ci, yc in (('uncal_teacherthr', cal, tl[cal]), ('teacher_cal', cal, tl[cal]), ('gold200_cal', cal_g, gold[cal_g])):
        t = 1.0 if name == 'uncal_teacherthr' else temp_fit(L(Etr[idx[ci]]), yc)
        Pc = softmax(L(Etr[idx[ci]])/t); Pt = softmax(L(Ete)/t)
        thr = thresh(Pc.max(1), Pc.argmax(1) == yc)
        dec = Pt.max(1) >= thr
        out[name] = dict(T=float(t), ece=float(ece(Pt, yte)), thr=float(thr), decided=float(dec.mean()), acc_decided=float(okte[dec].mean()) if dec.any() else None)
    rows.append(out); print(json.dumps(out), flush=True)
json.dump(rows, open(D/'result_calibrated.json', 'w'), indent=1)
for name in ('uncal_teacherthr', 'teacher_cal', 'gold200_cal'):
    print(name, 'ECE %.3f' % np.mean([r[name]['ece'] for r in rows]), 'decided %.3f' % np.mean([r[name]['decided'] for r in rows]),
          'acc decided %.3f' % np.mean([r[name]['acc_decided'] for r in rows if r[name]['acc_decided'] is not None]), 'T %.2f' % np.mean([r[name]['T'] for r in rows]))
