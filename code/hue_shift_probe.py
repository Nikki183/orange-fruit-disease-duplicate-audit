"""Causal colour probe: rotate image hue and record how predictions of the group-fold models change.
(a) Uganda images shifted -35 deg (yellow-green -> orange); (b) public healthy images shifted +35 deg (orange -> yellow-green).
usage: hue_shift_probe.py RESULTS_DIR MODELS_DIR"""
import sys, glob, os, numpy as np, pandas as pd
from PIL import Image
from ultralytics import YOLO
res, mdir = sys.argv[1:3]
def shift(im, deg):
    h, s, v = im.convert('RGB').convert('HSV').split()
    h = h.point(lambda x: int(x + deg * 255 / 360) % 256)
    return Image.merge('HSV', (h, s, v)).convert('RGB')
U = pd.read_csv(f'{res}/uganda_test.csv'); U['path'] = [os.path.join(res, '..', p) for p in U.path]
D = pd.read_csv(f'{res}/dataset4.csv'); D['path'] = [os.path.join(res, '..', p) for p in D.path]
H = D[D.cls == 'healthy']
uimg = {d: [shift(Image.open(p).resize((320, 320)), d) if d else Image.open(p).convert('RGB').resize((320, 320)) for p in U.path] for d in (0, -35)}
himg = {d: [shift(Image.open(p).resize((320, 320)), d) if d else Image.open(p).convert('RGB').resize((320, 320)) for p in H.path] for d in (0, 35)}
rows = []
for w in sorted(glob.glob(f'{mdir}/*__group_f*/best.pt')):
    tag = w.split('/')[-2]; m, s = tag.split('__'); model = YOLO(w); nm = model.names
    def pred(ims):
        out = []
        for k in range(0, len(ims), 64): out += [r.probs.data.cpu().numpy() for r in model.predict(ims[k:k+64], imgsz=224, device='mps', verbose=False)]
        return np.array(out)
    ci = {v: k for k, v in nm.items()}
    for d in (0, -35):
        P = pred(uimg[d]); top = [nm[int(i)] for i in P.argmax(1)]
        two = np.where(P[:, ci['canker']] > P[:, ci['healthy']], 'canker', 'healthy')
        for c in ('canker', 'healthy'):
            msk = (U.cls == c).values
            rows.append(dict(model=m, split=s, set='uganda', shift=d, true=c, pred_greening=np.mean([t == 'greening' for t, k in zip(top, msk) if k]),
                             pred_healthy=np.mean([t == 'healthy' for t, k in zip(top, msk) if k]), two_class_correct=np.mean(two[msk] == c)))
    for d in (0, 35):
        P = pred(himg[d]); top = [nm[int(i)] for i in P.argmax(1)]
        rows.append(dict(model=m, split=s, set='public_healthy', shift=d, true='healthy', pred_greening=np.mean([t == 'greening' for t in top]),
                         pred_healthy=np.mean([t == 'healthy' for t in top]), two_class_correct=np.nan))
    print(tag, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{res}/tables/hue_shift_probe_per_model.csv', index=False)
print((R.drop(columns=['split']).groupby(['set', 'true', 'shift']).agg(['mean', 'std']) * 100).round(1).to_string())
