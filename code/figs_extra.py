import sys, os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
res, out = sys.argv[1:3]
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'; C1, C2 = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 600, 'pdf.fonttype': 42})
t = pd.read_csv(f'{res}/table1_dataset.csv'); t = t[t.Class != 'Total']
D = json.load(open(f'{res}/doh_audit.json'))
fig = plt.figure(figsize=(7.2, 3.4)); gs = fig.add_gridspec(2, 6, width_ratios=[2.2, 0.25, 1, 1, 1, 1])
ax = fig.add_subplot(gs[:, 0]); x = np.arange(len(t)); w = 0.38
ax.bar(x - w/2 - 0.01, t.Files, w, color=C1, label='Files in folder')
ax.bar(x + w/2 + 0.01, t['Distinct images (pHash groups)'], w, color=C2, label='Distinct images')
for xi, a, b in zip(x, t.Files, t['Distinct images (pHash groups)']):
    ax.text(xi - w/2, a + 3, str(a), ha='center', fontsize=6.5, color=INK2); ax.text(xi + w/2, b + 3, str(b), ha='center', fontsize=6.5, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(t.Class, rotation=30, ha='right'); ax.set_ylim(0, 245); ax.set_ylabel('Number of images'); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=7, loc='upper left', ncol=2); fig.text(0.01, 0.96, '(a) Files and distinct images per class', fontsize=8)
# (b) one Rauf original and its copies
cop = []
if True:  # pick the original with most copies
    from collections import Counter
    k = Counter(r['rauf_match'] for r in D if r['rauf_d'] <= 4 and 'black_spot' in r['rauf_match']).most_common(1)[0][0]; orig = k
    cop = [r['path'] for r in D if r['rauf_match'] == k and r['rauf_d'] <= 4][:3]
for j, (p, lab) in enumerate([(orig, 'Rauf et al. original')] + [(c, os.path.basename(c)) for c in cop]):
    a = fig.add_subplot(gs[0, 2 + j]); a.imshow(Image.open(os.path.join(res, '..', p) if not os.path.isabs(p) else p).convert('RGB').resize((200, 200))); a.axis('off')
    a.set_title(lab, fontsize=6)
fig.text(0.42, 0.96, '(b) One original photograph and three of its copies in the public collection', fontsize=8)
# (c) label conflict
pairs = [('Greening', 'Greening 2.jpg'), ('Scab', 'Scab 15.jpg'), ('Scab', 'Scab 30.jpg'), ('Scab', 'Scab 45.jpg')]
for j, (c, f) in enumerate(pairs):
    a = fig.add_subplot(gs[1, 2 + j]); a.imshow(Image.open(os.path.join(res, '..', 'data/doh/Orange Fruit Diseases', c, f)).convert('RGB').resize((200, 200))); a.axis('off')
    a.set_title(f'"{c}" folder', fontsize=6)
fig.text(0.42, 0.485, '(c) One photograph filed under both greening and scab', fontsize=8)
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.86, wspace=0.08, hspace=0.55)
fig.savefig(f'{out}/Fig1_audit.png'); fig.savefig(f'{out}/Fig1_audit.tiff'); plt.close(fig)
# Fig 6 hue
def hue(p):
    im = Image.open(p).convert('RGB').resize((128, 128)); a = np.asarray(im.crop((32, 32, 96, 96)).convert('HSV')).reshape(-1, 3).astype(float)
    a = a[(a[:, 1] > 60) & (a[:, 2] > 50)]; return np.median(a[:, 0]) * 360 / 255 if len(a) else np.nan
d = pd.read_csv(f'{res}/dataset4.csv'); d['path'] = [os.path.join(res, '..', p) for p in d.path]; u = pd.read_csv(f'{res}/uganda_test.csv')
groups = [('Public', 'healthy', 'Healthy'), ('Public', 'canker', 'Canker'), ('Public', 'black_spot', 'Black spot'), ('Public', 'greening', 'Greening'),
          ('Uganda', 'healthy', 'Healthy'), ('Uganda', 'canker', 'Canker')]
vals = [np.array([hue(p) for p in (d if s == 'Public' else u).query('cls==@c').path]) for s, c, _ in groups]
fig, ax = plt.subplots(figsize=(3.5, 2.6)); rng = np.random.default_rng(0)
for i, (v, (s, c, lab)) in enumerate(zip(vals, groups)):
    v = v[~np.isnan(v)]; col = C1 if s == 'Public' else C2
    ax.scatter(rng.normal(i, 0.07, len(v)), v, s=3, color=col, alpha=0.35, lw=0)
    q = np.percentile(v, [25, 50, 75]); ax.plot([i - 0.25, i + 0.25], [q[1]] * 2, color=INK, lw=1.4); ax.plot([i, i], [q[0], q[2]], color=INK, lw=1)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([f'{g[0]}\n{g[2].lower()}' for g in groups], fontsize=7)
ax.set_ylabel('Median hue of central region (°)'); ax.set_ylim(0, 185)
fig.tight_layout(); fig.savefig(f'{out}/Fig6_hue.png'); fig.savefig(f'{out}/Fig6_hue.tiff'); fig.savefig(f'{out}/Fig6_hue.pdf'); plt.close(fig); print('ok')
