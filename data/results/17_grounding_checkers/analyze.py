"""Calibration, split-conformal abstention and pair accuracy from scores_300.jsonl (300 VitaminC flip pairs).
Splits are by pair (a supported and a refuted example of one claim stay together). 200 random 50/50 splits."""
import json, math, random, collections, statistics as st
R=[json.loads(l) for l in open('scores_300.jsonl')]
by=collections.defaultdict(dict)
for r in R: by[r['model']].setdefault(r['pair'],{})[r['side']]=r
def ece(ps,ys,bins=10):
    tot=0
    for b in range(bins):
        idx=[i for i,p in enumerate(ps) if (p>=b/bins and (p<(b+1)/bins or b==bins-1))]
        if idx: tot+=len(idx)*abs(sum(ps[i] for i in idx)/len(idx)-sum(ys[i] for i in idx)/len(idx))
    return tot/len(ps)
def run(m,alpha,seed):
    d=by[m]; ids=sorted(d); random.Random(seed).shuffle(ids); cal,te=ids[:150],ids[150:]
    ex=lambda S:[(d[i][s]['score'],d[i][s]['y']) for i in S for s in 'SR']
    c=ex(cal); t=ex(te)
    ns=sorted(1-(p if y else 1-p) for p,y in c); k=math.ceil((len(ns)+1)*(1-alpha))-1; q=ns[min(k,len(ns)-1)]
    cov=sing=ok=0
    for p,y in t:
        S=[lab for lab,pr in ((1,p),(0,1-p)) if 1-pr<=q]
        cov+= y in S
        if len(S)==1: sing+=1; ok+= S[0]==y
    # threshold from calibration: best accuracy
    best=max(((sum((p>th)==bool(y) for p,y in c),th) for th in [i/100 for i in range(1,100)]))[1]
    tacc=sum((p>best)==bool(y) for p,y in t)/len(t)
    return cov/len(t), sing/len(t), (ok/sing if sing else float('nan')), tacc, ece([p for p,_ in t],[y for _,y in t])
for m in ('minicheck','hhem','nli'):
    for a in (0.10,0.05):
        res=[run(m,a,s) for s in range(200)]
        f=lambda j:(st.mean(r[j] for r in res), min(r[j] for r in res), max(r[j] for r in res))
        cv,sg,ac,ta,ec=f(0),f(1),f(2),f(3),f(4)
        print(f'{m} alpha={a}: coverage {cv[0]:.3f} (min {cv[1]:.3f}) | decided alone {sg[0]:.3f} | accuracy when decided {ac[0]:.3f} | tuned-threshold acc {ta[0]:.3f} | ECE10 {ec[0]:.3f}')
