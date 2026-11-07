"""Hash-only grouping vs duplicate-aware (hash + embedding) grouping, nano models.
usage: analysis_hashgroup.py RESULTS_DIR"""
import sys, glob, json, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
res = sys.argv[1]; C = ['black_spot', 'canker', 'greening', 'healthy']
P = pd.concat([pd.read_csv(f) for f in glob.glob(f'{res}/preds/*-cls__*.csv')]); P = P[P.source == 'public']
names = {'yolov8n-cls': 'YOLOv8n-cls', 'yolo11n-cls': 'YOLO11n-cls'}
rows, out = [], []
for (m, s, r), d in P[P.model.isin(names) & P.regime.isin(['hashgroup', 'group', 'random'])].groupby(['model', 'split', 'regime']):
    x = dict(model=m, split=s, regime=r, acc=(d.true == d.pred).mean(), f1=f1_score(d.true, d.pred, labels=C, average='macro'))
    if r == 'hashgroup':
        a, b = d[d.dup_in_train], d[~d.dup_in_train]
        x.update(n_near=len(a), acc_near=(a.true == a.pred).mean(), acc_rest=(b.true == b.pred).mean())
    rows.append(x)
R = pd.DataFrame(rows); R.to_csv(f'{res}/tables/hashgroup_per_run.csv', index=False)
for m in names:
    h, g, rnd = (R[(R.model == m) & (R.regime == k)] for k in ('hashgroup', 'group', 'random'))
    msd = lambda v: f'{v.mean()*100:.1f} ± {v.std(ddof=1)*100:.1f}'
    out.append({'Model': names[m],
                'Random image-level macro-F1 (%)': msd(rnd.f1),
                'Hash-only group macro-F1 (%)': msd(h.f1),
                'Hash + embedding group macro-F1 (%)': msd(g.f1),
                'Δ hash-only − full (pp)': f'{(h.f1.mean()-g.f1.mean())*100:.1f}',
                'p (Welch)': f'{stats.ttest_ind(h.f1, g.f1, equal_var=False).pvalue:.3f}',
                'Test images with a near-duplicate in training (per fold)': f'{int(h.n_near.min())}–{int(h.n_near.max())}',
                'Accuracy, near-duplicate in training (%)': msd(h.acc_near),
                'Accuracy, no near-duplicate (%)': msd(h.acc_rest),
                'p (paired)': f'{stats.ttest_rel(h.acc_near, h.acc_rest).pvalue:.3f}'})
T = pd.DataFrame(out); T.to_csv(f'{res}/tables/table_hashgroup.csv', index=False); print(T.T.to_string())
