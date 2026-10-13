from explore1 import *
import statistics


def single(key):
    return lambda q,h: sorted(h,key=lambda x:-x[key])

F=lambda fn:(lambda q,h: fn([x for x in h if grounded(x)]))
print("== component rankers (grounded pool=20)  DEV / TEST")
for nm,fn in (("bm25 only",single('bm25')),("dense only",single('dense')),("server fused",lambda q,h:h)):
    for tag,E in (("dev",dev),("test",tst)):
        print(tag,fmt(evaluate(F(lambda hs,fn=fn:fn(None,hs)),E,gc,rows,nm)))

print("\n== weighted RRF (w_dense), k, pool=20, grounded, DEV")
best=[]
for k in (10,30,60):
    for wd in (0.0,0.25,0.5,0.75,1.0):
        m=evaluate(F(lambda hs,k=k,wd=wd: rrf(hs,k,(1.0,wd))),dev,gc,rows,f"wRRF k={k} w_dense={wd}")
        best.append((m['mrr'],m)); print(fmt(m))

print("\n== length-compensated linear (min-max) gamma, w_dense, DEV")
for g in (0.0,0.25,0.5,0.75,1.0):
    for wd in (0.25,0.5):
        m=evaluate(F(lambda hs,g=g,wd=wd: lenc(hs,g,wd=wd)),dev,gc,rows,f"lenc gamma={g} wd={wd}")
        print(fmt(m))

import numpy as np

bm=np.array([h['bm25'] for e in cache.values() for h in e['hits'] if grounded(h)])
dn=np.array([h['dense'] for e in cache.values() for h in e['hits'] if grounded(h)])
print("\nbm25 mean/std/max",bm.mean().round(2),bm.std().round(2),bm.max().round(1),"| dense mean/std/max",dn.mean().round(3),dn.std().round(3),dn.max().round(2))

rb=[];rd=[]
for k,e in dev.items():
    hs=[x for x in e['hits'] if grounded(x)]; g=gc[int(k)]
    gi=[i for i,x in enumerate(hs) if x['doc_id'] in g]
    if not gi: continue
    r1,r2=ranks(hs,'bm25'),ranks(hs,'dense'); rb.append(min(r1[i] for i in gi)); rd.append(min(r2[i] for i in gi))
    
print("gold-in-pool queries:",len(rb)," median gold rank  bm25:",statistics.median(rb)," dense:",statistics.median(rd)," mean bm25:",round(statistics.mean(rb),2)," dense:",round(statistics.mean(rd),2))
