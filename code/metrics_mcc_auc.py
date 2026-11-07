"""MCC and AUC (with macro-F1) for every evaluation, as recommended by Vallejo et al. (2026) and Demeyer et al. (2026).
usage: metrics_mcc_auc.py RESULTS_DIR MODELS_DIR"""
import sys, os, glob, numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
from ultralytics import YOLO
res, mdir = sys.argv[1:3]; base = os.path.join(res, '..'); C = ['black_spot', 'canker', 'greening', 'healthy']
rows = []
P = pd.concat([pd.read_csv(f) for f in glob.glob(f'{res}/preds/*-cls__*.csv')]); P = P[P.source == 'public']
for (m, s, r), d in P.groupby(['model', 'split', 'regime']):
    pr = d[[f'p_{c}' for c in C]].values; pr = pr / pr.sum(1, keepdims=True)
    rows.append(dict(eval='within_source_4class', model=m, split=s, regime=r, mcc=matthews_corrcoef(d.true, d.pred),
                     auc=roc_auc_score(d.true, pr, multi_class='ovr', average='macro', labels=C), f1=f1_score(d.true, d.pred, labels=C, average='macro')))
    if r == 'group':
        h = d[d.true.isin(['canker', 'healthy'])]; sc = h.p_canker / (h.p_canker + h.p_healthy); two = np.where(sc > 0.5, 'canker', 'healthy')
        rows.append(dict(eval='within_source_2class', model=m, split=s, regime=r, mcc=matthews_corrcoef(h.true, two), auc=roc_auc_score(h.true == 'canker', sc), f1=f1_score(h.true, two, average='macro')))
U = pd.concat([pd.read_csv(f) for f in glob.glob(f'{res}/preds/uganda__*.csv')])
for (m, s), d in U.groupby(['model', 'split']):
    sc = d.p_canker / (d.p_canker + d.p_healthy); two = np.where(sc > 0.5, 'canker', 'healthy')
    rows.append(dict(eval='public_to_uganda_2class', model=m, split=s, regime='group', mcc=matthews_corrcoef(d.true, two), auc=roc_auc_score(d.true == 'canker', sc), f1=f1_score(d.true, two, average='macro')))
Ut = pd.read_csv(f'{res}/uganda_test.csv'); D = pd.read_csv(f'{res}/dataset4.csv'); D = D[D.cls.isin(['canker', 'healthy'])]
for w in sorted(glob.glob(f'{mdir}/yolo11n-cls__ugandatrain_s*/best.pt')):
    s = w.split('__')[1].split('/')[0]; mdl = YOLO(w); ci = {v: k for k, v in mdl.names.items()}
    for ev, paths, truth in (('uganda_within_2class', [os.path.join(base, p) for p in Ut.path], Ut.cls.values), ('uganda_to_public_2class', [os.path.join(base, p) for p in D.path], D.cls.values)):
        sc = []
        for k in range(0, len(paths), 64): sc += [float(r.probs.data[ci['canker']]) for r in mdl.predict(paths[k:k+64], imgsz=224, device='mps', verbose=False)]
        sc = np.array(sc); two = np.where(sc > 0.5, 'canker', 'healthy')
        rows.append(dict(eval=ev, model='yolo11n-cls', split=s, regime='ugandatrain', mcc=matthews_corrcoef(truth, two), auc=roc_auc_score(truth == 'canker', sc), f1=f1_score(truth, two, average='macro')))
R = pd.DataFrame(rows); R.to_csv(f'{res}/tables/mcc_auc_per_run.csv', index=False)
S = R.groupby(['eval', 'regime', 'model'])[['f1', 'mcc', 'auc']].agg(['mean', 'std']).round(3); S.to_csv(f'{res}/tables/mcc_auc_summary.csv'); print(S.to_string())
