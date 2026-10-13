from explore1 import *
import random, numpy as np


F=lambda fn:(lambda q,h: fn([x for x in h if grounded(x)]))

def per_query(order_fn,E):
    out={}
    for k,e in E.items():
        g=gc[int(k)]; ids=[h['doc_id'] for h in order_fn(e['q'],e['hits'])]
        r=next((j+1 for j,d in enumerate(ids) if d in g),None)
        out[k]=(1.0/r if r else 0.0, 1.0 if (r and r<=3) else 0.0)
    return out

def boot(a,b,B=4000,seed=0):
    ks=list(a); rng=random.Random(seed); d_m=[];d_h=[]
    dm=np.array([a[k][0]-b[k][0] for k in ks]); dh=np.array([a[k][1]-b[k][1] for k in ks])
    n=len(ks); idx=np.array([[rng.randrange(n) for _ in range(n)] for _ in range(B)])
    bm=dm[idx].mean(1); bh=dh[idx].mean(1)
    f=lambda x:(round(x.mean(),4),round(float(np.percentile(x,2.5)),4),round(float(np.percentile(x,97.5)),4))
    return f(bm),f(bh)

base=F(lambda hs:hs)
cands={"RRF k=60 equal (original idea)":F(lambda hs:rrf(hs,60,(1,1))),
       "lenc g=0.5 wd=0.25":F(lambda hs:lenc(hs,0.5,wd=0.25)),
       "lenc g=0.75 wd=0.25":F(lambda hs:lenc(hs,0.75,wd=0.25)),
       "bm25 only":F(lambda hs:sorted(hs,key=lambda x:-x['bm25']))}

print("== paired diff vs 'server fused + ground filter'; (mean, 95% CI) for MRR and Hit@3, ALL 330 / TEST 167")

for nm,fn in cands.items():
    for tag,E in (("ALL",cache),("TEST",tst)):
        m,h=boot(per_query(fn,E),per_query(base,E))
        print(f"{nm:34s} {tag:4s} dMRR={m}  dHit@3={h}")

# Outlier hypothesis: does RRF help on queries with a BM25 outlier in the RAW (unfiltered) pool?
print("\n== outlier hypothesis (raw pool incl. poison): ratio = top bm25 / median bm25 of pool")
ratios={k:max(h['bm25'] for h in e['hits'])/max(np.median([h['bm25'] for h in e['hits']]),1e-9) for k,e in cache.items()}
thr=np.percentile(list(ratios.values()),75)
out={k:v for k,v in cache.items() if ratios[k]>=thr}; non={k:v for k,v in cache.items() if ratios[k]<thr}
rawfn_srv=lambda q,h:h
rawfn_rrf=lambda q,h:rrf(h[:5],60,(1,1))

for nm,E in (("outlier quartile (n=%d)"%len(out),out),("others (n=%d)"%len(non),non)):
    print(nm); print("  ",fmt(evaluate(rawfn_srv,E,gc,rows,"server fused (raw)"))); print("  ",fmt(evaluate(rawfn_rrf,E,gc,rows,"RRF pool5 (raw)")))
    
print("poison share among outlier-quartile top1:",np.mean([cache[k]['hits'][0]['doc_id']>=9000 for k in out]).round(3),"vs others",np.mean([cache[k]['hits'][0]['doc_id']>=9000 for k in non]).round(3))
