"""Banking77 on a laptop CPU: (a) LLM-label distillation vs gold labels, (b) self-training with three acceptance rules,
(c) teacher-error injection. Student = logistic regression (C=10, no tuning) on frozen bge-small embeddings.
usage: python exp_b77.py a|b|c"""
import json, sys, math, numpy as np, collections
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
D = Path('/private/tmp/claude-502/-Users-ben-Dropbox-projects-deterministic-ai/bd422d64-fbc7-49e1-9a07-432b93aca2e4/scratchpad/b77')
z = np.load(D/'emb_bge_small.npz'); Etr, Ete, ytr_s, yte_s = z['Etr'], z['Ete'], z['ytr'], z['yte']
classes = sorted(set(ytr_s)); cid = {c: i for i, c in enumerate(classes)}
ytr = np.array([cid[c] for c in ytr_s]); yte = np.array([cid[c] for c in yte_s]); K = len(classes)
sample = json.load(open(D.parent/'b77_gold'/'sample_idx.json'))
def wilson(k, n, zz=1.96):
    p = k/n; d = 1+zz*zz/n; c = (p+zz*zz/2/n)/d; h = zz*math.sqrt(p*(1-p)/n+zz*zz/4/n/n)/d; return c-h, c+h
def fit(X, y): return LogisticRegression(C=10, max_iter=2000).fit(X, y)
def proba(m, X):
    P = np.zeros((len(X), K)); P[:, m.classes_] = m.predict_proba(X); return P
def ece(P, y, bins=10):
    conf = P.max(1); ok = P.argmax(1) == y; t = 0
    for b in range(bins):
        i = (conf >= b/bins) & ((conf < (b+1)/bins) | (b == bins-1))
        if i.any(): t += i.sum()*abs(conf[i].mean()-ok[i].mean())
    return t/len(y)
def evaluate(m):
    P = proba(m, Ete); ok = P.argmax(1) == yte
    f1 = np.mean([2*((P.argmax(1) == c) & (yte == c)).sum()/max(1, (P.argmax(1) == c).sum()+(yte == c).sum()) for c in range(K)])
    return ok, float(ok.mean()), float(f1), float(ece(P, yte))
def mcnemar(a, b):
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); return n10, n01, (binomtest(n10, n10+n01, .5).pvalue if n10+n01 else 1.0)
mode = sys.argv[1]
if mode == 'a':
    T = {}
    for k in range(5):
        for l in open(D/'teacher'/f'out_{k}.jsonl'): r = json.loads(l); T[r['id']] = r['label']
    assert len(T) == 2000, len(T)
    bad = [v for v in T.values() if v not in cid]; print('teacher labels outside the 77 names:', len(bad), set(bad), flush=True)
    tl = np.array([cid.get(T[i], -1) for i in range(2000)]); idx = np.array(sample); gold = ytr[idx]
    k = int((tl == gold).sum()); lo, hi = wilson(k, 2000); print(f'teacher accuracy on {2000} train items: {k} ({k/2000:.3f}, Wilson {lo:.3f} to {hi:.3f})', flush=True)
    # per-class teacher recall: how many classes below 50%
    rec = [((tl == gold) & (gold == c)).sum()/max(1, (gold == c).sum()) for c in range(K)]; print('classes with teacher recall < 0.5:', sum(r < .5 for r in rec), flush=True)
    rng = np.random.default_rng(0); out = {}
    for N in (100, 500, 2000):
        reps = 1 if N == 2000 else 5; res = []
        for s in range(reps):
            sel = np.arange(N) if N == 2000 else rng.choice(2000, N, replace=False)
            row = {}
            for arm, lab in (('gold', gold[sel]), ('teacher', tl[sel])):
                m = fit(Etr[idx[sel]], lab); ok, acc, f1, e = evaluate(m); row[arm] = dict(ok=ok, acc=acc, f1=f1, ece=e)
            n10, n01, p = mcnemar(row['gold']['ok'], row['teacher']['ok']); row['mcnemar'] = (n10, n01, p)
            res.append(row)
            print(f"N={N} subset {s}: gold acc {row['gold']['acc']:.3f} F1 {row['gold']['f1']:.3f} ECE {row['gold']['ece']:.3f} | teacher-label acc {row['teacher']['acc']:.3f} F1 {row['teacher']['f1']:.3f} ECE {row['teacher']['ece']:.3f} | gold-only-right {n10} teacher-only-right {n01} McNemar p={p:.3g}", flush=True)
        g = [r['gold']['acc'] for r in res]; t = [r['teacher']['acc'] for r in res]
        lo, hi = wilson(round(np.mean(t)*len(yte)), len(yte))
        out[N] = dict(gold_mean=float(np.mean(g)), teacher_mean=float(np.mean(t)), teacher_wilson=[lo, hi], gold_range=[min(g), max(g)], teacher_range=[min(t), max(t)])
        print(f'N={N}: mean over {reps} subsets gold {np.mean(g):.3f} [{min(g):.3f},{max(g):.3f}] teacher {np.mean(t):.3f} [{min(t):.3f},{max(t):.3f}]', flush=True)
    json.dump(out, open(D/'result_a.json', 'w'), indent=1)
if mode == 'b':
    import csv
    from sklearn.feature_extraction.text import TfidfVectorizer
    G = D.parent/'b77_gold'; txt = [r['text'] for r in csv.DictReader(open(G/'train.csv'))]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(txt); Xtf = tf.transform(txt)
    fitq = lambda X, y: LogisticRegression(C=10, max_iter=500).fit(X, y)
    out = {}
    for seed in range(3):
        rng = np.random.default_rng(seed); perm = rng.permutation(len(ytr)); seedi, cal, pool = perm[:100], perm[100:400], perm[400:]
        m0 = fitq(Etr[seedi], ytr[seedi]); ok0, acc0, f10, e0 = evaluate(m0)
        mref = fitq(Etr[np.r_[seedi, cal]], ytr[np.r_[seedi, cal]]); _, accref, _, _ = evaluate(mref)
        print(f'seed {seed}: 100 gold only acc {acc0:.3f} ECE {e0:.3f}; 400 gold (reference) acc {accref:.3f}', flush=True)
        for rule in ('thr0.9', 'conformal', 'two_views'):
            m = m0; mt = fitq(Xtf[seedi], ytr[seedi])
            for rnd in range(1, 4):
                P = proba(m, Etr[pool]); conf = P.max(1); pred = P.argmax(1)
                if rule == 'thr0.9': acc_mask = conf >= 0.9
                elif rule == 'conformal':
                    Pc = proba(m, Etr[cal]); ns = np.sort(1-Pc[np.arange(len(cal)), ytr[cal]]); kq = min(len(ns)-1, math.ceil((len(ns)+1)*0.9)-1); q = ns[kq]
                    acc_mask = ((1-P) <= q).sum(1) == 1; pred = np.where(acc_mask, np.argmax((1-P) <= q, axis=1), pred)
                else:
                    Pt = np.zeros((len(pool), K)); Pt[:, mt.classes_] = mt.predict_proba(Xtf[pool])
                    acc_mask = (Pt.argmax(1) == P.argmax(1)) & (conf >= .5) & (Pt.max(1) >= .5)
                n = int(acc_mask.sum()); err = float((pred[acc_mask] != ytr[pool][acc_mask]).mean()) if n else float('nan')
                Xn = np.vstack([Etr[seedi], Etr[pool][acc_mask]]); yn = np.r_[ytr[seedi], pred[acc_mask]]
                m = fitq(Xn, yn)
                if rule == 'two_views': mt = fitq(Xtf[np.r_[seedi, pool[acc_mask]]], yn)
                ok, acc, f1, e = evaluate(m)
                out[f'{rule}|{seed}|{rnd}'] = dict(accepted=n, pseudo_error=err, acc=acc, f1=f1, ece=e)
                print(f'seed {seed} {rule:10s} round {rnd}: accepted {n:5d} of {len(pool)}, pseudo-label error among accepted {err:.3f}, test acc {acc:.3f} F1 {f1:.3f} ECE {e:.3f}', flush=True)
        out[f'ref|{seed}'] = dict(acc_100=acc0, acc_400=accref)
    json.dump(out, open(D/'result_b.json', 'w'), indent=1)
if mode == 'c':
    idx = np.array(sample); X = Etr[idx]; gold = ytr[idx]
    cent = np.stack([Etr[ytr == c].mean(0) for c in range(K)]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    S = cent @ cent.T; np.fill_diagonal(S, -1); nearest = S.argmax(1)
    out = {}
    for kind in ('random', 'confusable'):
        for r in (0.0, 0.1, 0.2, 0.3, 0.4):
            accs = []
            for seed in range(3):
                rng = np.random.default_rng(seed); y = gold.copy(); flip = rng.random(len(y)) < r
                y[flip] = (nearest[gold[flip]] if kind == 'confusable' else (gold[flip] + rng.integers(1, K, flip.sum())) % K)
                _, acc, _, _ = evaluate(fit(X, y)); accs.append(acc)
            out[f'{kind}|{r}'] = accs; print(f'{kind:10s} flip rate {r:.1f}: test acc mean {np.mean(accs):.3f} range [{min(accs):.3f},{max(accs):.3f}]', flush=True)
    json.dump(out, open(D/'result_c.json', 'w'), indent=1)
