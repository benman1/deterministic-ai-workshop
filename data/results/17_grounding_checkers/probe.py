import json, time, random, collections, sys, torch
torch.set_num_threads(4)
rows=[json.loads(l) for l in open('/private/tmp/claude-502/-Users-ben-Dropbox-projects-deterministic-ai/bd422d64-fbc7-49e1-9a07-432b93aca2e4/scratchpad/nb17/vitc_test.jsonl')]
g=collections.defaultdict(list)
for r in rows:
    if r['revision_type']=='real': g[r['case_id']].append(r)
pairs=[]
for cid,v in g.items():
    by={}
    for r in v:
        if r['label'] in('SUPPORTS','REFUTES'): by.setdefault((r['claim'],r['label']),r)
    claims={c for c,_ in by}
    for c in claims:
        if (c,'SUPPORTS') in by and (c,'REFUTES') in by:
            pairs.append((by[(c,'SUPPORTS')],by[(c,'REFUTES')])); break
print('flip pairs',len(pairs),flush=True)
random.seed(0); random.shuffle(pairs); P=pairs[:10]
ex=[(s['claim'],s['evidence'],1) for s,_ in P]+[(r['claim'],r['evidence'],0) for _,r in P]
def stamp(name,fn):
    t=time.time()
    try:
        out=fn(); print(f'{name}: loaded+scored {len(ex)} in {time.time()-t:.1f}s ->',out[:6],flush=True)
    except Exception as e:
        print(f'{name}: FAILED {type(e).__name__}: {str(e)[:300]}',flush=True)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def ce():
    from sentence_transformers import CrossEncoder
    m=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); return [round(float(x),3) for x in m.predict([(c,e) for c,e,_ in ex])]
def nli():
    tok=AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base'); m=AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base').eval()
    print('nli labels',m.config.id2label,flush=True)
    out=[]
    for c,e,_ in ex:
        with torch.no_grad(): lg=m(**tok(e,c,return_tensors='pt',truncation=True)).logits.softmax(-1)[0]
        out.append([round(float(x),3) for x in lg])
    return out
def hhem():
    from transformers import AutoConfig, T5ForTokenClassification
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg=AutoConfig.from_pretrained('google/flan-t5-base'); cfg.num_labels=2
    m=T5ForTokenClassification(cfg).eval()
    sd={k[3:] if k.startswith('t5.') else k:v for k,v in load_file(hf_hub_download('vectara/hallucination_evaluation_model','model.safetensors')).items()}
    r=m.load_state_dict(sd,strict=False); print('hhem missing',len(r.missing_keys),r.missing_keys[:3],'unexpected',len(r.unexpected_keys),r.unexpected_keys[:3],flush=True)
    tok=AutoTokenizer.from_pretrained('google/flan-t5-base')
    P="<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {t1}\n\nHypothesis: {t2}"
    out=[]
    for c,e,_ in ex:
        with torch.no_grad(): lg=m(**tok([P.format(t1=e,t2=c)],return_tensors='pt',padding=True)).logits[:,0,:].softmax(-1)[0]
        out.append(round(float(lg[1]),3))
    return out
def mc():
    tok=AutoTokenizer.from_pretrained('lytang/MiniCheck-RoBERTa-Large'); m=AutoModelForSequenceClassification.from_pretrained('lytang/MiniCheck-RoBERTa-Large').eval()
    print('minicheck labels',m.config.id2label,flush=True)
    out=[]
    for c,e,_ in ex:
        with torch.no_grad(): lg=m(**tok(e,c,return_tensors='pt',truncation=True)).logits.softmax(-1)[0]
        out.append([round(float(x),3) for x in lg])
    return out
print('labels',[y for *_,y in ex],flush=True)
for n,f in [('cross-encoder relevance',ce),('nli deberta',nli),('hhem',hhem),('minicheck roberta',mc)]: stamp(n,f)
