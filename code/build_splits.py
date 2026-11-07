"""Build image-level (random) and group-aware splits for the 4-class public orange dataset."""
import sys, os, json, collections, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
res=sys.argv[1]; T=4
CLS={'Black spot':'black_spot','Canker':'canker','Greening':'greening','healthy':'healthy','Scab':'scab'}
D=json.load(open(f'{res}/doh_audit.json')); M=np.load(f'{res}/doh_pair.npy'); n=len(D)
par=list(range(n))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
for i in range(n):
    for j in range(i+1,n):
        if M[i,j]<=T: par[f(i)]=f(j)
# link copies of the same Rauf original
byr=collections.defaultdict(list)
for i,r in enumerate(D):
    if r['rauf_d']<=T: byr[r['rauf_match']].append(i)
for v in byr.values():
    for i in v[1:]: par[f(i)]=f(v[0])
df=pd.DataFrame(dict(path=[r['path'] for r in D], cls=[CLS[r['cls']] for r in D], group=[f(i) for i in range(n)],
    rauf=[r['rauf_d']<=T for r in D]))
conf=df.groupby('group').cls.nunique(); bad=conf[conf>1].index
summ=dict(n_files=n, n_groups=df.group.nunique(), conflicting_groups=len(bad), files_in_conflicting=int(df.group.isin(bad).sum()),
          rauf_derived=int(df.rauf.sum()), files_per_class=df.cls.value_counts().to_dict(),
          groups_per_class=df.groupby('cls').group.nunique().to_dict())
df=df[~df.group.isin(bad) & (df.cls!='scab')].reset_index(drop=True)
# merge near-duplicates found by ResNet-50 embedding (cosine >= 0.95), computed on this same filtered table
E=np.load(f'{res}/emb.npy'); Sm=E@E.T; m=len(df); par2={g:g for g in df.group}
def f2(x):
    while par2[x]!=x: par2[x]=par2[par2[x]]; x=par2[x]
    return x
ii,jj=np.where(np.triu(Sm,1)>=0.95)
for a,b in zip(ii,jj): par2[f2(df.group[a])]=f2(df.group[b])
summ['hash_groups_4class']=int(df.group.nunique()); summ['embedding_merged_pairs']=int(len(ii))
df['group']=[f2(g) for g in df.group]
summ.update(after_n=len(df), after_groups=int(df.group.nunique()), after_files_per_class=df.cls.value_counts().to_dict(),
    after_groups_per_class=df.groupby('cls').group.nunique().to_dict(),
    group_size=df.groupby('group').size().describe().to_dict())
df.to_csv(f'{res}/dataset4.csv',index=False)
splits=[]
for s in range(5):  # E1 random image-level 70/10/20
    tr,te=train_test_split(df.index,test_size=0.2,stratify=df.cls,random_state=s)
    tr,va=train_test_split(tr,test_size=0.125,stratify=df.cls[tr],random_state=s)
    splits.append(dict(name=f'random_s{s}',regime='random',train=list(map(int,tr)),val=list(map(int,va)),test=list(map(int,te))))
sg=StratifiedGroupKFold(5,shuffle=True,random_state=0)
for k,(tr,te) in enumerate(sg.split(df.index,df.cls,df.group)):
    sg2=StratifiedGroupKFold(8,shuffle=True,random_state=k)
    a,b=next(sg2.split(tr,df.cls[tr],df.group[tr]))
    splits.append(dict(name=f'group_f{k}',regime='group',train=list(map(int,tr[a])),val=list(map(int,tr[b])),test=list(map(int,te))))
for sp in splits:
    g=lambda ix:set(df.group[ix]); sp['test_groups_in_train']=len(g(sp['test'])&g(sp['train']))
    sp['test_imgs_with_dup_in_train']=int(df.loc[sp['test'],'group'].isin(g(sp['train'])).sum())
json.dump(splits,open(f'{res}/splits.json','w'))
summ['leak']={sp['name']:(len(sp['test']),sp['test_imgs_with_dup_in_train']) for sp in splits}
json.dump(summ,open(f'{res}/dataset_summary.json','w'),indent=1,default=str); print(json.dumps(summ,indent=1,default=str))
