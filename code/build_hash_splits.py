"""Hash-only grouping (exact/near-exact copies via pHash, no embedding merge), for comparison with the full duplicate-aware split.
Adds 'hashgroup_f0..4' to splits.json and reports how many test images still have an embedding near-duplicate in training.
usage: build_hash_splits.py RESULTS_DIR"""
import sys, json, collections, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
res = sys.argv[1]; T = 4
D = json.load(open(f'{res}/doh_audit.json')); M = np.load(f'{res}/intermediate/doh_pair.npy')
df = pd.read_csv(f'{res}/dataset4.csv'); idx = {r['path']: i for i, r in enumerate(D)}
n = len(D); par = list(range(n))
def f(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for i in range(n):
    for j in range(i + 1, n):
        if M[i, j] <= T: par[f(i)] = f(j)
byr = collections.defaultdict(list)
for i, r in enumerate(D):
    if r['rauf_d'] <= T: byr[r['rauf_match']].append(i)
for v in byr.values():
    for i in v[1:]: par[f(i)] = f(v[0])
df['hgroup'] = [f(idx[p]) for p in df.path]
E = np.load(f'{res}/intermediate/emb.npy'); S = E @ E.T
splits = json.load(open(f'{res}/splits.json')); splits = [s for s in splits if s['regime'] != 'hashgroup']
sg = StratifiedGroupKFold(5, shuffle=True, random_state=0); out = {}
for k, (tr, te) in enumerate(sg.split(df.index, df.cls, df.hgroup)):
    a, b = next(StratifiedGroupKFold(8, shuffle=True, random_state=k).split(tr, df.cls[tr], df.hgroup[tr]))
    trn, val = tr[a], tr[b]
    near = int(sum(S[t, trn].max() >= 0.95 for t in te))
    exact = int(df.loc[te, 'hgroup'].isin(set(df.hgroup[trn])).sum())
    splits.append(dict(name=f'hashgroup_f{k}', regime='hashgroup', train=list(map(int, trn)), val=list(map(int, val)), test=list(map(int, te)),
                       test_imgs_with_dup_in_train=exact, test_imgs_with_near_dup_in_train=near))
    out[f'hashgroup_f{k}'] = dict(n_test=len(te), exact_copy_in_train=exact, embedding_near_duplicate_in_train=near)
json.dump(splits, open(f'{res}/splits.json', 'w'))
json.dump(out, open(f'{res}/hash_split_summary.json', 'w'), indent=1); print(json.dumps(out, indent=1))
