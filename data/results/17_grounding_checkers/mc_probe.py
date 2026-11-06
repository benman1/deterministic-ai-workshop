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
        out=fn(); print(f'{name}: loaded+scored {len(ex)} in {time.time()-t:.1f}s ->',out,flush=True)
    except Exception as e:
        print(f'{name}: FAILED {type(e).__name__}: {str(e)[:300]}',flush=True)
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
def mc():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    M='lytang/MiniCheck-RoBERTa-Large'
    cfg=AutoConfig.from_pretrained(M); tok=AutoTokenizer.from_pretrained(M)
    m=AutoModelForSequenceClassification.from_config(cfg).eval()
    sd=load_file(hf_hub_download(M,'model.safetensors',revision='refs/pr/1'))
    r=m.load_state_dict(sd,strict=False); print('missing',len(r.missing_keys),r.missing_keys[:3],'unexpected',len(r.unexpected_keys),r.unexpected_keys[:3],flush=True)
    print('labels',m.config.id2label,flush=True)
    out=[]
    for c,e,_ in ex:
        with torch.no_grad(): p=m(**tok(e,c,return_tensors='pt',truncation=True,max_length=512)).logits.softmax(-1)[0]
        out.append(round(float(p[1]),3)); print(len(out),out[-1],flush=True)
    return out
print('y',[y for *_,y in ex],flush=True)
stamp('minicheck roberta (safetensors PR1)',mc)
