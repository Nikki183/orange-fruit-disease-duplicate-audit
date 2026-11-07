import sys, os, glob, json, collections
import numpy as np, imagehash
from PIL import Image, ImageOps
doh, rauf, out = sys.argv[1:4]
def variants(im):
    v=[]
    for f in (im, ImageOps.mirror(im)):
        for r in (0,90,180,270): v.append(f.rotate(r, expand=True))
    return v
def load(root):
    recs=[]
    for p in sorted(glob.glob(os.path.join(root,'*','*'))):
        if p.lower().split('.')[-1] not in ('jpg','jpeg','png'): continue
        im=Image.open(p).convert('RGB')
        recs.append(dict(path=p, cls=os.path.basename(os.path.dirname(p)), size=im.size,
            ext=p.split('.')[-1].lower(),
            ph=[imagehash.phash(x) for x in variants(im)]))
    return recs
D=load(doh); R=load(rauf)
print('doh',len(D),'rauf',len(R))
def dist(a,b): return min(a['ph'][0]-h for h in b['ph'])
# size distribution
print(collections.Counter((r['cls'],r['size'],r['ext']) for r in D).most_common(40))
# match to rauf
for r in D:
    ds=[dist(r,q) for q in R]; j=int(np.argmin(ds)); r['rauf_d']=ds[j]; r['rauf_match']=R[j]['path']; r['rauf_cls']=R[j]['cls']
# within-doh pairwise
n=len(D); M=np.full((n,n),99)
for i in range(n):
    for j in range(i+1,n):
        M[i,j]=M[j,i]=dist(D[i],D[j])
np.save(os.path.join(out,'doh_pair.npy'),M)
for t in (0,4,6,8,10,12):
    # union-find clusters
    par=list(range(n))
    def f(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for i in range(n):
        for j in range(i+1,n):
            if M[i,j]<=t: par[f(i)]=f(j)
    cl=collections.Counter(f(i) for i in range(n))
    cross=sum(1 for i in range(n) for j in range(i+1,n) if M[i,j]<=t and D[i]['cls']!=D[j]['cls'])
    inr=sum(r['rauf_d']<=t for r in D)
    print(f't={t}: groups={len(cl)} imgs_in_dup_groups={sum(c for c in cl.values() if c>1)} crossclass_pairs={cross} doh_matching_rauf={inr}')
json.dump([{k:(str(v) if k=='size' else v) for k,v in r.items() if k!='ph'} for r in D], open(os.path.join(out,'doh_audit.json'),'w'), indent=0)
# per class rauf matches at t=8
print(collections.Counter((r['cls'], r['rauf_d']<=8, r['rauf_cls'] if r['rauf_d']<=8 else '-') for r in D))
