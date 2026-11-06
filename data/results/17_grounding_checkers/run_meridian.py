"""Score VitaminC flip pairs with four small checkers. Streams one JSON line per example.
usage: python -I run_checkers.py N OUT.jsonl model1,model2   (models: ce,nli,hhem,minicheck)
A flip pair = one claim with evidence that SUPPORTS it and evidence (after a Wikipedia edit) that REFUTES it."""
import json, time, random, collections, sys, torch
torch.set_num_threads(4)
N=int(sys.argv[1]); OUT=sys.argv[2]; MODELS=sys.argv[3].split(",")
D='/private/tmp/claude-502/-Users-ben-Dropbox-projects-deterministic-ai/bd422d64-fbc7-49e1-9a07-432b93aca2e4/scratchpad/nb17/'
rows=[json.loads(l) for l in open(D+'vitc_test.jsonl')]
g=collections.defaultdict(list)
for r in rows:
    if r['revision_type']=='real': g[r['case_id']].append(r)
DROP={76,79}   # the subagent doubted these two (convoluted inference; vague exception)
P=[json.loads(l) for l in open(D+'meridian_pairs.jsonl')]; P=[r for r in P if r['id'] not in DROP][:N]
ex=[]
for i,r in enumerate(P):
    ex.append((i,r['id'],'S',r['claim'],r['current'],1)); ex.append((i,r['id'],'R',r['claim'],r['stale'],0))
print('pairs',len(P),'examples',len(ex),flush=True)
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
def load_ce():
    from sentence_transformers import CrossEncoder
    m=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return lambda c,e: float(m.predict([(c,e)])[0])
def load_nli():
    M='cross-encoder/nli-deberta-v3-base'; tok=AutoTokenizer.from_pretrained(M); m=AutoModelForSequenceClassification.from_pretrained(M).eval()
    ent=[k for k,v in m.config.id2label.items() if v.lower()=='entailment'][0]
    def f(c,e):
        with torch.no_grad(): return float(m(**tok(e,c,return_tensors='pt',truncation=True,max_length=512)).logits.softmax(-1)[0][ent])
    return f
def load_hhem():
    from transformers import T5ForTokenClassification
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg=AutoConfig.from_pretrained('google/flan-t5-base'); cfg.num_labels=2
    m=T5ForTokenClassification(cfg).eval()
    sd={k[3:] if k.startswith('t5.') else k:v for k,v in load_file(hf_hub_download('vectara/hallucination_evaluation_model','model.safetensors')).items()}
    r=m.load_state_dict(sd,strict=False); print('hhem missing',r.missing_keys,'unexpected',r.unexpected_keys,flush=True)
    tok=AutoTokenizer.from_pretrained('google/flan-t5-base')
    T="<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {t1}\n\nHypothesis: {t2}"
    def f(c,e):
        with torch.no_grad(): return float(m(**tok([T.format(t1=e,t2=c)],return_tensors='pt',padding=True,truncation=True,max_length=512)).logits[:,0,:].softmax(-1)[0][1])
    return f
def load_mc():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    M='lytang/MiniCheck-RoBERTa-Large'; cfg=AutoConfig.from_pretrained(M); tok=AutoTokenizer.from_pretrained(M)
    m=AutoModelForSequenceClassification.from_config(cfg).eval()
    r=m.load_state_dict(load_file(hf_hub_download(M,'model.safetensors',revision='refs/pr/1')),strict=False); print('mc missing',len(r.missing_keys),'unexpected',len(r.unexpected_keys),flush=True)
    def f(c,e):
        with torch.no_grad(): return float(m(**tok(e,c,return_tensors='pt',truncation=True,max_length=512)).logits.softmax(-1)[0][1])
    return f
L={'ce':load_ce,'nli':load_nli,'hhem':load_hhem,'minicheck':load_mc}
out=open(OUT,'a')
for name in MODELS:
    t0=time.time(); f=L[name](); print(f'{name}: loaded in {time.time()-t0:.1f}s',flush=True)
    tot=0; hit=0
    for i,cid,side,c,e,y in ex:
        t=time.time(); s=f(c,e); ms=(time.time()-t)*1000
        out.write(json.dumps(dict(model=name,pair=i,case=cid,side=side,y=y,score=s,ms=round(ms,1)))+'\n'); out.flush()
        tot+=1; hit+=int((s>0.5)==bool(y)) if name!='ce' else 0
        if tot%10==0: print(f'{name} {tot}/{len(ex)} last {s:.3f} {ms:.0f}ms running acc@0.5 {hit}/{tot}',flush=True)
    print(f'{name}: done {len(ex)} in {time.time()-t0:.1f}s',flush=True)
