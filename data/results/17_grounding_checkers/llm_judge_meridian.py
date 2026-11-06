"""Score VitaminC flip pairs with four small checkers. Streams one JSON line per example.
usage: python -I run_checkers.py N OUT.jsonl model1,model2   (models: ce,nli,hhem,minicheck)
A flip pair = one claim with evidence that SUPPORTS it and evidence (after a Wikipedia edit) that REFUTES it."""
import json, time, random, collections, sys, torch
torch.set_num_threads(4)
N=int(sys.argv[1]); OUT=sys.argv[2]; MODELS=[sys.argv[3]]
D='/private/tmp/claude-502/-Users-ben-Dropbox-projects-deterministic-ai/bd422d64-fbc7-49e1-9a07-432b93aca2e4/scratchpad/nb17/'
DROP={76,79}
P=[json.loads(l) for l in open(D+'meridian_pairs.jsonl')]; P=[r for r in P if r['id'] not in DROP][:N]
ex=[]
for i,r in enumerate(P):
    ex.append((i,r['id'],'S',r['claim'],r['current'],1)); ex.append((i,r['id'],'R',r['claim'],r['stale'],0))
print('pairs',len(P),'examples',len(ex),flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
name=MODELS[0]
t0=time.time(); tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32).eval()
print(f'{name} loaded in {time.time()-t0:.1f}s',flush=True)
Y=tok.encode('Yes',add_special_tokens=False)[0]; Nn=tok.encode('No',add_special_tokens=False)[0]
def judge(c,e):
    msg=[{'role':'user','content':f'Evidence: {e}\n\nClaim: {c}\n\nIs the claim supported by the evidence? Answer with one word, Yes or No.'}]
    try: p=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: p=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
    with torch.no_grad(): lg=m(**tok(p,return_tensors='pt',truncation=True,max_length=1024)).logits[0,-1]
    return float(torch.softmax(torch.stack([lg[Nn],lg[Y]]),0)[1])
out=open(OUT,'a'); hit=0; tot=0; pr=[]
for i,cid,side,c,e,y in ex:
    t=time.time(); s=judge(c,e); ms=(time.time()-t)*1000
    out.write(json.dumps(dict(model=name,pair=i,case=cid,side=side,y=y,score=s,ms=round(ms,1)))+'\n'); out.flush()
    tot+=1; hit+=int((s>0.5)==bool(y)); pr.append(s)
    if tot%10==0: print(f'{tot}/{len(ex)} last {s:.3f} {ms:.0f}ms acc@0.5 {hit}/{tot}',flush=True)
ok=sum(pr[2*k]>pr[2*k+1] for k in range(len(pr)//2)); print('pairs ranked right',ok,'of',len(pr)//2,flush=True)
