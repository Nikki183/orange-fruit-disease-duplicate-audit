"""Figure 1 (counts chart, vector) and Figure 2 (photo comparisons) as separate files, per the journal's artwork rules.
usage: figs_split_fig1.py RESULTS_DIR OUT_DIR"""
import sys, os, json, numpy as np, pandas as pd
from collections import Counter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
res, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
INK2, GRID, C1, C2 = '#52514e', '#e4e3df', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42, 'axes.edgecolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600})
t = pd.read_csv(f'{res}/table1_dataset.csv'); t = t[t.Class != 'Total']
fig, ax = plt.subplots(figsize=(3.5, 2.6)); x = np.arange(len(t)); w = 0.38
ax.bar(x - w/2 - 0.01, t.Files, w, color=C1, label='Files in folder')
ax.bar(x + w/2 + 0.01, t['Distinct images (pHash groups)'], w, color=C2, label='Distinct images')
for xi, a, b in zip(x, t.Files, t['Distinct images (pHash groups)']):
    ax.text(xi - w/2, a + 3, str(a), ha='center', fontsize=6.5, color=INK2); ax.text(xi + w/2, b + 3, str(b), ha='center', fontsize=6.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(t.Class, rotation=30, ha='right'); ax.set_ylim(0, 245); ax.set_ylabel('Number of images')

fig.tight_layout(); fig.savefig(f'{out}/Figure_1.pdf'); fig.savefig(f'{out}/Figure_1.tiff'); fig.savefig(f'{out}/Figure_1.png'); plt.close(fig)
D = json.load(open(f'{res}/doh_audit.json'))
k = Counter(r['rauf_match'] for r in D if r['rauf_d'] <= 4 and 'black_spot' in r['rauf_match']).most_common(1)[0][0]
cop = [r['path'] for r in D if r['rauf_match'] == k and r['rauf_d'] <= 4][:3]
rowA = [(k, 'Rauf et al. original')] + [(c, os.path.basename(c)) for c in cop]
rowB = [(f'data/doh/Orange Fruit Diseases/{c}/{f}', f'"{c}" folder') for c, f in [('Greening', 'Greening 2.jpg'), ('Scab', 'Scab 15.jpg'), ('Scab', 'Scab 30.jpg'), ('Scab', 'Scab 45.jpg')]]
fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.0))
for r, row in enumerate((rowA, rowB)):
    for j, (p, lab) in enumerate(row):
        a = axs[r, j]; a.imshow(Image.open(os.path.join(res, '..', p)).convert('RGB').resize((300, 300))); a.axis('off'); a.set_title(lab, fontsize=7)
fig.text(0.01, 0.965, '(a)', fontsize=9, weight='bold'); fig.text(0.01, 0.48, '(b)', fontsize=9, weight='bold')
fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.02, wspace=0.05, hspace=0.25)
fig.savefig(f'{out}/Figure_2.tiff', dpi=600); fig.savefig(f'{out}/Figure_2.png', dpi=600); plt.close(fig); print('ok')
