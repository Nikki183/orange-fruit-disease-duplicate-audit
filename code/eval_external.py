"""Evaluate every saved group-fold model on the external Uganda test set.
usage: eval_external.py RESULTS_DIR WORK_DIR"""
import sys, os, glob, json, pandas as pd, numpy as np
from ultralytics import YOLO
res, work = sys.argv[1:3]
ex = pd.read_csv(f'{res}/uganda_test.csv')
for w in sorted(glob.glob(f'{work}/runs/*__group_f*/weights/best.pt')):
    tag = w.split('/runs/')[1].split('/')[0]; m, s = tag.split('__')
    o = f'{res}/preds/uganda__{tag}.csv'
    if os.path.exists(o): continue
    model = YOLO(w); names = model.names; P = []
    for k in range(0, len(ex), 64):
        P += [r.probs.data.cpu().numpy() for r in model.predict(list(ex.path[k:k+64]), imgsz=224, device='mps', verbose=False)]
    P = np.array(P)
    d = pd.DataFrame(dict(model=m, split=s, regime='group', idx=-1, path=ex.path, true=ex.cls, pred=[names[int(a)] for a in P.argmax(1)],
                          conf=P.max(1), dup_in_train=False))
    for j, c in names.items(): d[f'p_{c}'] = P[:, j]
    d['source'] = 'uganda'; d.to_csv(o, index=False); print(tag, (d.true == d.pred).mean())
