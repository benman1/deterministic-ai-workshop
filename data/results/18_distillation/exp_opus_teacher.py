"""Stronger teacher: Opus sub-agent labels vs Haiku labels on the same 2,000 Banking77 training messages; student (LR C=10 on bge-small) at 2,000.
usage: python exp_opus_teacher.py"""
import sys, json, numpy as np
sys.argv = ['x', 'none']
exec(open('exp_b77.py').read().split("mode = sys.argv[1]")[0])
def load(prefix):
    T = {}
    for k in range(5):
        for l in open(D/'teacher'/f'{prefix}{k}.jsonl'): r = json.loads(l); T[r['id']] = r['label']
    return np.array([cid.get(T[i], -1) for i in range(2000)])
idx = np.array(sample); gold = ytr[idx]
th, to = load('out_'), load('opus_out_')
res = {}
for name, tl in (('haiku', th), ('opus', to)):
    k = int((tl == gold).sum()); lo, hi = wilson(k, 2000); print(f'{name} label accuracy {k}/2000 = {k/2000:.3f} Wilson {lo:.3f} to {hi:.3f}', flush=True)
    res[name] = dict(k=k, lo=lo, hi=hi)
print('haiku vs opus agree on', int((th == to).sum()), 'of 2000', flush=True)
ok = {}
for name, lab in (('gold', gold), ('haiku', th), ('opus', to)):
    m = fit(Etr[idx], lab); o, acc, f1, e = evaluate(m); ok[name] = o; lo, hi = wilson(int(o.sum()), len(o))
    print(f'student on {name} labels: acc {acc:.3f} (Wilson {lo:.3f} to {hi:.3f}) F1 {f1:.3f} ECE {e:.3f}', flush=True)
    res['student_' + name] = dict(acc=acc, f1=f1, ece=e, lo=lo, hi=hi)
for a, b in (('opus', 'haiku'), ('gold', 'opus')):
    n10, n01, p = mcnemar(ok[a], ok[b]); print(f'McNemar {a} vs {b}: {a}-only right {n10}, {b}-only right {n01}, p={p:.3g}', flush=True); res[f'mcnemar_{a}_{b}'] = [n10, n01, p]
json.dump(res, open(D/'result_opus_teacher.json', 'w'), indent=1)
