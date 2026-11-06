import json, numpy as np
from collections import Counter
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
import meridian as m
R = Path("data/results"); r01 = json.loads((R/"01_encoders_and_baseline.json").read_text())
tr, te = m.load("train"), m.load("test"); _, ytr = m.texts_labels(tr); _, yte = m.texts_labels(te)
votes = {}
for n in ("bge-small","modernbert","lfm2.5-enc"):
    E = m.cached_embeddings(n); votes[n] = LogisticRegression(C=r01[n]["C"], max_iter=3000).fit(E["train"], ytr).predict(E["test"])
idmap = json.loads(Path("data/llm/test_id_map.json").read_text())
ans = {idmap[j["id"]]: j["route"] for j in map(json.loads, Path("data/llm/test_haiku_answers.jsonl").read_text().splitlines())}
# slow tier = recorded Haiku answers, mapped to route index as in notebook 10
votes["slow tier"] = np.array([m.ROUTE_ID[ans[r["id"]]] for r in te])
names = list(votes); W = np.stack([votes[n] != yte for n in names]).T   # True = wrong
N = len(yte)
def eff(Wb):
    c = np.corrcoef(Wb.T); iu = np.triu_indices(len(names), 1); rho = np.nanmean(c[iu]); return rho, len(names)/(1+(len(names)-1)*rho)
rho, ne = eff(W); print(f"N={N} error rates", {n: round(float(W[:, i].mean()), 3) for i, n in enumerate(names)})
rng = np.random.default_rng(0); bs = np.array([eff(W[rng.integers(0, N, N)]) for _ in range(2000)])
print(f"rho {rho:.3f}  95% bootstrap [{np.percentile(bs[:,0],2.5):.3f}, {np.percentile(bs[:,0],97.5):.3f}]")
print(f"effective voters {ne:.2f}  95% bootstrap [{np.nanpercentile(bs[:,1],2.5):.2f}, {np.nanpercentile(bs[:,1],97.5):.2f}]")
p = W.mean(0); exp_all = float(np.prod(p))*N; obs_all = int(W.all(1).sum())
print(f"all four wrong together: observed {obs_all}, expected under independence {exp_all:.2f}")
k_wrong = W.sum(1); pw = np.zeros(len(names)+1); pw[0] = 1
for q in p: pw = np.convolve(pw, [1-q, q])[:len(names)+1]
print("messages by number of wrong voters, observed vs independent:", [(k, int((k_wrong == k).sum()), round(float(pw[k])*N, 1)) for k in range(len(names)+1)])
# majority vote (strict majority of 4 = at least 3 same) vs best single voter, exact McNemar on all 200
M = np.stack([votes[n] for n in names]).T
top = np.array([Counter(r).most_common(1)[0] for r in M], dtype=object)
plur = np.array([t[0] for t in top]); cnt = np.array([t[1] for t in top])
best = names[int(np.argmin(p))]; print("best single voter:", best, f"error {p.min():.3f}")
for label, pred in (("plurality vote", plur),):
    a = (pred == yte); b = (votes[best] == yte)
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
    pv = binomtest(n10, n10+n01, 0.5).pvalue if n10+n01 else 1.0
    print(f"{label} right {a.sum()}/{N}, best voter right {b.sum()}/{N}; vote-only-right {n10}, voter-only-right {n01}, exact McNemar p={pv:.3f}")
