"""Figure 6: two-class (canker vs healthy) MCC for within- and cross-source tests in both directions.
usage: fig_transfer.py RESULTS_DIR OUT_PREFIX"""
import sys, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
res, out = sys.argv[1:3]
R = pd.read_csv(f'{res}/tables/mcc_auc_per_run.csv')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600})
COL = {'yolov8n-cls': '#2a78d6', 'yolov8s-cls': '#2a78d6', 'yolo11n-cls': '#eb6834', 'yolo11s-cls': '#eb6834'}
MK = {'yolov8n-cls': 'o', 'yolo11n-cls': 'o', 'yolov8s-cls': 's', 'yolo11s-cls': 's'}
conds = [('within_source_2class', 'Train public\ntest public'), ('public_to_uganda_2class', 'Train public\ntest Uganda'),
         ('uganda_within_2class', 'Train Uganda\ntest Uganda'), ('uganda_to_public_2class', 'Train Uganda\ntest public')]
fig, ax = plt.subplots(figsize=(3.5, 2.7)); rng = np.random.default_rng(1)
for i, (ev, lab) in enumerate(conds):
    d = R[R['eval'] == ev]; ms = sorted(d.model.unique()); off = np.linspace(-0.27, 0.27, len(ms)) if len(ms) > 1 else [0]
    for o, m in zip(off, ms):
        v = d[d.model == m].mcc.values; x = i + o
        ax.scatter(x + rng.uniform(-0.03, 0.03, len(v)), v, s=9, color=COL[m], marker=MK[m], alpha=0.45, lw=0)
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        ax.errorbar(x, v.mean(), yerr=h, fmt=MK[m], ms=5, color=COL[m], mec='white', mew=0.8, lw=1.2, capsize=0)
ax.axhline(0, color='#52514e', lw=0.6, ls=(0, (2, 2)))
ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[1] for c in conds], fontsize=7)
ax.set_ylabel('Matthews correlation coefficient'); ax.set_ylim(-0.35, 1.05); ax.set_xlim(-0.55, 3.55)
fig.tight_layout()
for e in ('pdf', 'tiff', 'png'): fig.savefig(f'{out}.{e}')
print('ok')
