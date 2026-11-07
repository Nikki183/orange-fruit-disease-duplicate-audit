"""Shortcut probe: predict held-out (group-fold) test images after masking the central region
(background/context only) or the border (central region only).
usage: occlusion_probe.py RESULTS_DIR WORK_DIR"""
import sys, os, glob, json, numpy as np, pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO
res, work = sys.argv[1:3]
df = pd.read_csv(f'{res}/dataset4.csv'); df['path'] = [os.path.abspath(os.path.join(res, '..', p)) for p in df.path]
splits = {s['name']: s for s in json.load(open(f'{res}/splits.json'))}
GRAY = (124, 116, 104)  # ImageNet mean RGB
def mask(im, mode, frac=0.7):
    im = im.convert('RGB').resize((224, 224)); w, h = im.size
    box = [w*(1-frac)/2, h*(1-frac)/2, w*(1+frac)/2, h*(1+frac)/2]
    if mode == 'center':
        ImageDraw.Draw(im).ellipse(box, fill=GRAY); return im
    out = Image.new('RGB', im.size, GRAY); m = Image.new('L', im.size, 0); ImageDraw.Draw(m).ellipse(box, fill=255)
    out.paste(im, (0, 0), m); return out
rows = []
for w in sorted(glob.glob(f'{work}/runs/*__group_f*/weights/best.pt')):
    tag = w.split('/runs/')[1].split('/')[0]; m, s = tag.split('__'); model = YOLO(w); names = model.names
    te = splits[s]['test']; truth = list(df.cls[te])
    for mode in ('none', 'center', 'border'):
        ims = [Image.open(p).convert('RGB').resize((224, 224)) if mode == 'none' else mask(Image.open(p), mode) for p in df.path[te]]
        pr = [names[int(r.probs.top1)] for r in model.predict(ims, imgsz=224, device='mps', verbose=False)]
        rows.append(dict(model=m, split=s, mode=mode, acc=float(np.mean([a == b for a, b in zip(truth, pr)])),
                         majority=float(pd.Series(truth).value_counts(normalize=True).iloc[0]), n=len(te)))
    print(tag, rows[-3:])
pd.DataFrame(rows).to_csv(f'{res}/occlusion_probe.csv', index=False)
