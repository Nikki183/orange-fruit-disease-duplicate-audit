"""Aggregate predictions into tables, statistics and figures for the manuscript.
usage: analysis.py RESULTS_DIR OUT_DIR"""
import sys, os, glob, json, itertools
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
CLASSES = ['black_spot', 'canker', 'greening', 'healthy']
LABEL = {'black_spot': 'Black spot', 'canker': 'Canker', 'greening': 'Greening', 'healthy': 'Healthy'}
MODELS = ['yolov8n-cls', 'yolo11n-cls', 'yolov8s-cls', 'yolo11s-cls']
MNAME = {'yolov8n-cls': 'YOLOv8n-cls', 'yolo11n-cls': 'YOLO11n-cls', 'yolov8s-cls': 'YOLOv8s-cls', 'yolo11s-cls': 'YOLO11s-cls'}
COL = {'v8': '#2a78d6', '11': '#eb6834'}           # categorical slots 1 and 2 (validated palette)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 600, 'figure.dpi': 150})
rng = np.random.default_rng(2026)

P = pd.concat([pd.read_csv(f) for f in glob.glob(f'{res}/preds/*.csv')], ignore_index=True)
P = P[P.regime.isin(['random', 'group']) | (P.source == 'uganda')].copy(); P['dup_in_train'] = P.dup_in_train.astype(str).str.lower().eq('true')
J = pd.DataFrame([json.load(open(f)) for f in glob.glob(f'{res}/preds/*.json')])
pub = P[P.source == 'public']

def metrics(d):
    return dict(acc=accuracy_score(d.true, d.pred), macro_f1=f1_score(d.true, d.pred, labels=CLASSES, average='macro', zero_division=0), n=len(d))

# ---------- per-run metrics ----------
rows = []
for (m, s, r), d in pub.groupby(['model', 'split', 'regime']):
    x = metrics(d); x.update(model=m, split=s, regime=r)
    if r == 'random':
        a, b = d[d.dup_in_train], d[~d.dup_in_train]
        x.update(acc_dup=accuracy_score(a.true, a.pred), n_dup=len(a), acc_nodup=accuracy_score(b.true, b.pred), n_nodup=len(b))
    rows.append(x)
R = pd.DataFrame(rows); R.to_csv(f'{out}/per_run_metrics.csv', index=False)

def msd(v): return f'{np.mean(v)*100:.1f} ± {np.std(v, ddof=1)*100:.1f}'
def ci_t(v):
    v = np.asarray(v); h = stats.t.ppf(0.975, len(v)-1) * v.std(ddof=1) / np.sqrt(len(v)); return v.mean()-h, v.mean()+h

# ---------- Table: regime x model ----------
T = []
for m in MODELS:
    for r in ['random', 'group']:
        d = R[(R.model == m) & (R.regime == r)]
        if d.empty: continue
        lo, hi = ci_t(d.macro_f1)
        T.append(dict(Model=MNAME[m], Split=('Random image-level' if r == 'random' else 'Duplicate-aware group'),
                      runs=len(d), Accuracy=msd(d.acc), MacroF1=msd(d.macro_f1), MacroF1_CI95=f'{lo*100:.1f}–{hi*100:.1f}'))
    a = R[(R.model == m) & (R.regime == 'random')].macro_f1; b = R[(R.model == m) & (R.regime == 'group')].macro_f1
    if len(a) > 1 and len(b) > 1:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        T[-1]['Welch_p_vs_random'] = f'{p:.3g}'; T[-1]['Delta_F1_pp'] = f'{(a.mean()-b.mean())*100:.1f}'
pd.DataFrame(T).to_csv(f'{out}/table_regime_model.csv', index=False)

# ---------- leaked vs unleaked test images in random splits ----------
L = []
for m in MODELS:
    d = R[(R.model == m) & (R.regime == 'random')]
    if d.empty: continue
    pp = pub[(pub.model == m) & (pub.regime == 'random')]
    a, b = pp[pp.dup_in_train], pp[~pp.dup_in_train]
    # two-proportion comparison pooled over the five random splits (images can recur across splits -> descriptive)
    L.append(dict(Model=MNAME[m], n_dup=len(a), acc_dup=f'{(a.true==a.pred).mean()*100:.1f}', n_nodup=len(b),
                  acc_nodup=f'{(b.true==b.pred).mean()*100:.1f}',
                  per_split_dup=msd(d.acc_dup), per_split_nodup=msd(d.acc_nodup),
                  paired_t_p=f'{stats.ttest_rel(d.acc_dup, d.acc_nodup).pvalue:.3g}'))
pd.DataFrame(L).to_csv(f'{out}/table_leak_subsets.csv', index=False)

# ---------- model comparisons on the group folds ----------
G = R[R.regime == 'group']; C = []
n_test = G.groupby('split').n.first().mean(); n_train = 578 - n_test  # approximate ratio for Nadeau-Bengio correction
def corrected_t(diff, k=5, ratio=None):
    diff = np.asarray(diff); var = diff.var(ddof=1); ratio = ratio or (1/(k-1))
    t = diff.mean() / np.sqrt((1/len(diff) + ratio) * var) if var > 0 else np.inf
    return t, 2*stats.t.sf(abs(t), len(diff)-1)
oof = pub[pub.regime == 'group']
for a, b in [('yolov8n-cls', 'yolo11n-cls'), ('yolov8s-cls', 'yolo11s-cls'), ('yolov8n-cls', 'yolov8s-cls'), ('yolo11n-cls', 'yolo11s-cls')]:
    da = G[G.model == a].set_index('split').macro_f1; db = G[G.model == b].set_index('split').macro_f1
    common = da.index.intersection(db.index)
    if len(common) < 2: continue
    diff = (db[common] - da[common]).values
    t, p = corrected_t(diff, ratio=n_test/n_train)
    # McNemar on pooled out-of-fold predictions (each image predicted exactly once per model)
    xa = oof[oof.model == a].set_index('idx'); xb = oof[oof.model == b].set_index('idx')
    ix = xa.index.intersection(xb.index); ca = (xa.loc[ix].true == xa.loc[ix].pred); cb = (xb.loc[ix].true == xb.loc[ix].pred)
    n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    pm = stats.binomtest(min(n01, n10), n01+n10, 0.5).pvalue if n01+n10 > 0 else 1.0
    C.append(dict(Comparison=f'{MNAME[b]} − {MNAME[a]}', mean_diff_F1_pp=f'{diff.mean()*100:.1f}', SD_pp=f'{diff.std(ddof=1)*100:.1f}',
                  corrected_t_p=f'{p:.3g}', mcnemar_b=n01, mcnemar_c=n10, mcnemar_exact_p=f'{pm:.3g}'))
pd.DataFrame(C).to_csv(f'{out}/table_model_comparison.csv', index=False)

# ---------- per-class (pooled out-of-fold, group regime) ----------
PC = []
for m in MODELS:
    d = oof[oof.model == m]
    if d.empty: continue
    p_, r_, f_, s_ = precision_recall_fscore_support(d.true, d.pred, labels=CLASSES, zero_division=0)
    for c, pi, ri, fi, si in zip(CLASSES, p_, r_, f_, s_):
        PC.append(dict(Model=MNAME[m], Class=LABEL[c], Precision=round(pi*100, 1), Recall=round(ri*100, 1), F1=round(fi*100, 1), n=int(si)))
pd.DataFrame(PC).to_csv(f'{out}/table_per_class_group.csv', index=False)

# ---------- efficiency ----------
J['model'] = J.tag.str.split('__').str[0]
E = J.groupby('model').agg(params=('params', 'first'), latency_ms=('latency_ms_median', 'median'), train_s=('train_s', 'median')).reindex(MODELS).dropna()
E.to_csv(f'{out}/table_efficiency.csv')

# ---------- cross-source (Uganda) ----------
U = P[P.source == 'uganda']
if not U.empty:
    UR = []
    for (m, s), d in U.groupby(['model', 'split']):
        h, c = d[d.true == 'healthy'], d[d.true == 'canker']
        UR.append(dict(model=m, split=s, acc4=accuracy_score(d.true, d.pred),
                       macro_f1_2=f1_score(d.true, d.pred, labels=['canker', 'healthy'], average='macro', zero_division=0),
                       bin_acc=((d.true == 'healthy') == (d.pred == 'healthy')).mean(),
                       canker_recall=(c.pred == 'canker').mean(), diseased_sens=(c.pred != 'healthy').mean(),
                       healthy_spec=(h.pred == 'healthy').mean(), n_h=len(h), n_c=len(c)))
    UR = pd.DataFrame(UR); UR.to_csv(f'{out}/per_run_uganda.csv', index=False)
    UT = []
    for m in MODELS:
        d = UR[UR.model == m]; g = R[(R.model == m) & (R.regime == 'group')]
        if d.empty: continue
        # matched within-source reference: same models, healthy+canker images only, group folds
        w = pub[(pub.model == m) & (pub.regime == 'group') & pub.true.isin(['healthy', 'canker'])]
        wf = [f1_score(x.true, x.pred, labels=['canker', 'healthy'], average='macro', zero_division=0) for _, x in w.groupby('split')]
        UT.append(dict(Model=MNAME[m], within_source_F1_HC=msd(wf), cross_source_F1_HC=msd(d.macro_f1_2),
                       drop_pp=f'{(np.mean(wf)-d.macro_f1_2.mean())*100:.1f}',
                       paired_t_p=f'{stats.ttest_rel(wf, d.sort_values("split").macro_f1_2).pvalue:.3g}' if len(wf) == len(d) else '',
                       canker_recall=msd(d.canker_recall), diseased_sensitivity=msd(d.diseased_sens), healthy_specificity=msd(d.healthy_spec),
                       binary_acc=msd(d.bin_acc), n_healthy=int(d.n_h.iloc[0]), n_canker=int(d.n_c.iloc[0])))
    pd.DataFrame(UT).to_csv(f'{out}/table_cross_source.csv', index=False)
    # where do the Uganda images go?
    dist = U.groupby(['model', 'true']).pred.value_counts(normalize=True).unstack(fill_value=0).round(3)
    dist.to_csv(f'{out}/uganda_prediction_distribution.csv')

# ================= FIGURES =================
def style(ax):
    pass

# Fig 2: macro-F1 by split regime and model (dots = runs, bar = mean, whisker = 95% CI)
fig, ax = plt.subplots(figsize=(3.5, 2.6))
xs = {'random': 0, 'group': 1}; off = {'yolov8n-cls': -0.27, 'yolo11n-cls': -0.09, 'yolov8s-cls': 0.09, 'yolo11s-cls': 0.27}
for m in MODELS:
    fam = '11' if '11' in m else 'v8'; mk = 'o' if 'n-' in m else 's'
    for r in ['random', 'group']:
        d = R[(R.model == m) & (R.regime == r)].macro_f1 * 100
        if d.empty: continue
        x = xs[r] + off[m]
        ax.scatter(np.full(len(d), x) + rng.uniform(-0.025, 0.025, len(d)), d, s=10, color=COL[fam], marker=mk, alpha=0.45, lw=0)
        lo, hi = ci_t(d / 100)
        ax.errorbar(x, d.mean(), yerr=[[d.mean()-lo*100], [hi*100-d.mean()]], fmt=mk, ms=5, color=COL[fam], mec='white', mew=0.8, capsize=0, lw=1.2,
                    label=MNAME[m] if r == 'random' else None)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Random image-level\nsplit (5 seeds)', 'Duplicate-aware\ngroup split (5 folds)'])
ax.set_ylabel('Macro-F1 on held-out images (%)'); style(ax)
fig.tight_layout(); fig.savefig(f'{out}/Fig2_regime.png'); fig.savefig(f'{out}/Fig2_regime.tiff'); fig.savefig(f'{out}/Fig2_regime.pdf'); plt.close(fig)

# Fig 3: accuracy on test images with vs without a copy in training (random splits)
if len(L):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for k, m in enumerate(MODELS):
        d = R[(R.model == m) & (R.regime == 'random')]
        if d.empty: continue
        fam = '11' if '11' in m else 'v8'; mk = 'o' if 'n-' in m else 's'
        for j, col in enumerate(['acc_dup', 'acc_nodup']):
            v = d[col] * 100; x = j + (k - 1.5) * 0.14
            ax.scatter(np.full(len(v), x), v, s=9, color=COL[fam], marker=mk, alpha=0.4, lw=0)
            ax.plot(x, v.mean(), mk, ms=5, color=COL[fam], mec='white', mew=0.8, label=MNAME[m] if j == 0 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Test image has a copy\nin the training set', 'Test image has no copy\nin the training set'])
    ax.set_ylabel('Accuracy (%)'); style(ax)
    fig.tight_layout(); fig.savefig(f'{out}/Fig3_leak.png'); fig.savefig(f'{out}/Fig3_leak.tiff'); fig.savefig(f'{out}/Fig3_leak.pdf'); plt.close(fig)

# Fig 4: confusion matrices: YOLO11n-cls pooled out-of-fold (group) and Uganda
def cm_plot(ax, d, rows, cols, title):
    M = pd.crosstab(d.true, d.pred).reindex(index=rows, columns=cols, fill_value=0).values
    N = M / M.sum(1, keepdims=True)
    ax.imshow(N, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f'{M[i, j]}', ha='center', va='center', fontsize=7, color='white' if N[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([LABEL[c] for c in cols], rotation=35, ha='right')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([LABEL[c] for c in rows])
    ax.set_xlabel('Predicted class'); ax.set_title(title, fontsize=8, color=INK)
    for s in ax.spines.values(): s.set_visible(False)
best = 'yolo11n-cls'
d1 = oof[oof.model == best]
if not d1.empty:
    ncol = 2 if not U.empty else 1
    fig, axs = plt.subplots(1, ncol, figsize=(3.5*ncol*0.95, 2.9), squeeze=False)
    cm_plot(axs[0, 0], d1, CLASSES, CLASSES, '(a) Public set, group folds'); axs[0, 0].set_ylabel('True class')
    if not U.empty:
        cm_plot(axs[0, 1], U[U.model == best], ['canker', 'healthy'], CLASSES, '(b) Independent Uganda set')
    fig.tight_layout(); fig.savefig(f'{out}/Fig4_confusion.png'); fig.savefig(f'{out}/Fig4_confusion.tiff'); fig.savefig(f'{out}/Fig4_confusion.pdf'); plt.close(fig)

# Fig 5: within- vs cross-source macro-F1 (healthy vs canker)
if not U.empty:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for k, m in enumerate(MODELS):
        d = UR[UR.model == m].sort_values('split')
        if d.empty: continue
        w = pub[(pub.model == m) & (pub.regime == 'group') & pub.true.isin(['healthy', 'canker'])]
        wf = np.array([f1_score(x.true, x.pred, labels=['canker', 'healthy'], average='macro', zero_division=0) for _, x in w.groupby('split')]) * 100
        cf = d.macro_f1_2.values * 100
        fam = '11' if '11' in m else 'v8'; mk = 'o' if 'n-' in m else 's'; x0 = (k - 1.5) * 0.14
        for a_, b_ in zip(wf, cf): ax.plot([0 + x0, 1 + x0], [a_, b_], color=COL[fam], alpha=0.25, lw=0.7)
        ax.plot([0 + x0, 1 + x0], [wf.mean(), cf.mean()], marker=mk, color=COL[fam], ms=5, mec='white', mew=0.8, lw=1.6, label=MNAME[m])
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Within source\n(public, group folds)', 'Cross source\n(Uganda phone images)'])
    ax.set_ylabel('Macro-F1, canker vs healthy (%)'); style(ax); ax.set_xlim(-0.5, 1.5)
    ax.legend(frameon=False, fontsize=6.5, loc='lower left')
    fig.tight_layout(); fig.savefig(f'{out}/Fig5_cross_source.png'); fig.savefig(f'{out}/Fig5_cross_source.tiff'); fig.savefig(f'{out}/Fig5_cross_source.pdf'); plt.close(fig)

for f in sorted(glob.glob(f'{out}/table_*.csv')):
    print('\n##', os.path.basename(f))
    try: print(pd.read_csv(f).to_string(index=False))
    except Exception: print('(empty)')
