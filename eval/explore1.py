from metrics import *
import math, itertools, zlib


rows=load_corpus(); gc=build_gold(rows)
cache=json.load(open('cache_clean.json',encoding='utf-8'))
full=[squash(r.get('BriefAnswer'))+squash(r.get('Answer')) for r in rows]

def grounded(h):
    d=h['doc_id']; return d<len(rows) and squash(h['text']) in full[d]

dev={k:v for k,v in cache.items() if int(zlib.crc32(k.encode()))%2==0}
tst={k:v for k,v in cache.items() if k not in dev}
print("dev",len(dev),"test",len(tst))

def ranks(hits,key):
    order=sorted(range(len(hits)),key=lambda i:-float(hits[i][key])); r={};prev=None;cur=0
    for p,i in enumerate(order,1):
        s=float(hits[i][key])
        if p==1 or s!=prev: cur=p
        r[i]=cur;prev=s
    return r

def rrf(hits,k=60,w=(1,1)):
    rb,rd=ranks(hits,'bm25'),ranks(hits,'dense')
    sc=[w[0]/(k+rb[i])+w[1]/(k+rd[i]) for i in range(len(hits))]
    return [hits[i] for i in sorted(range(len(hits)),key=lambda i:-sc[i])]

def lenc(hits,gamma,ref=300.0,wd=0.5):
    # length-compensated linear fusion on min-max scaled scores
    bm=[h['bm25']*(max(len(h['text']),1)/ref)**gamma for h in hits]
    dn=[h['dense'] for h in hits]
    nb=lambda xs:[(x-min(xs))/(max(xs)-min(xs)+1e-9) for x in xs]
    b,d=nb(bm),nb(dn)
    sc=[(1-wd)*b[i]+wd*d[i] for i in range(len(hits))]
    return [hits[i] for i in sorted(range(len(hits)),key=lambda i:-sc[i])]

def pool(fn,P,filt=False):
    def f(q,h):
        hs=[x for x in h if (grounded(x) or not filt)] if filt else h
        return fn(hs[:P])
    return f

ident=lambda x:x
for name,E in (("DEV",dev),("TEST",tst)):
    print("==",name)
    print(fmt(evaluate(pool(ident,20),E,gc,rows,"server fused")))
    for P in (5,10,20):
        print(fmt(evaluate(pool(rrf,P),E,gc,rows,f"RRF pool={P} (no filter)")))
    print(fmt(evaluate(pool(ident,20,True),E,gc,rows,"server fused + ground filter")))
    for P in (5,10,20):
        print(fmt(evaluate(pool(rrf,P,True),E,gc,rows,f"RRF pool={P} + ground filter")))
