import json,collections,math,numpy as np
from scipy.stats import binomtest
R="/Users/ben/deterministic-ai-workshop/data/results/17_grounding_checkers/"
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));return round((c-h)/d,3),round((c+h)/d,3)
def sc(name,model=None):
    o={}
    for l in open(R+name):
        d=json.loads(l)
        if model and d["model"]!=model: continue
        o[(d["pair"],d["side"])]=d["score"]
    return o
models=collections.Counter(json.loads(l)["model"] for l in open(R+"meridian_scores.jsonl")); print(models)
ref={"meridian":{"Qwen":sc("qwen3_1.7b_judge_meridian118.jsonl"),"MiniCheck":sc("meridian_scores.jsonl","minicheck")},
     "vitaminc":{"Qwen":sc("qwen3_1.7b_judge_300.jsonl"),"MiniCheck":sc("scores_300.jsonl","minicheck")}}
done={}
for l in open("twil_outputs.jsonl"):
    d=json.loads(l); done[(d["set"],d["pair"],d["side"])]=d
out={}
for name in("meridian","vitaminc"):
    recs=[d for d in done.values() if d["set"]==name]; by=collections.defaultdict(dict)
    for d in recs: by[d["pair"]][d["side"]]=d
    full=[p for p,s in by.items() if len(s)==2]
    ok=lambda d: d["verdict"] is not None and (d["verdict"]=="Yes")==bool(d["y"])
    k=sum(ok(d) for d in recs);n=len(recs)
    both={p for p in full if ok(by[p]["S"]) and ok(by[p]["R"])}
    toks=np.array([d["tokens"] for d in recs]);secs=np.array([d["sec"] for d in recs])
    cut=sum(d["cut"] for d in recs);unp=sum(d["verdict"] is None and not d["cut"] for d in recs)
    print(f"\n{name}: {n} claims, {len(full)} pairs; claim acc {k}/{n}={k/n:.3f} {wil(k,n)}; pairs both right {len(both)}/{len(full)}={len(both)/len(full):.3f} {wil(len(both),len(full))}")
    print(f"  tokens median {np.median(toks):.0f} p95 {np.percentile(toks,95):.0f}; cut {cut}; unparsed {unp}; mean sec {secs.mean():.1f}")
    for m,s in ref[name].items():
        qb={p for p in full if s[(p,"S")]>0.5 and s[(p,"R")]<=0.5}
        a=len(both-qb);b=len(qb-both);pv=binomtest(a,a+b,.5).pvalue if a+b else 1
        print(f"  vs {m}: {m} {len(qb)}/{len(full)}={len(qb)/len(full):.3f} {wil(len(qb),len(full))}; only TwIL {a}, only {m} {b}, McNemar p={pv:.3g}")
    # side-wise
    for side in("S","R"):
        r=[d for d in recs if d["side"]==side]; print(f"  side {side}: {sum(ok(d) for d in r)}/{len(r)}")
