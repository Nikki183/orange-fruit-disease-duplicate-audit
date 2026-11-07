"""Deduplicate the Uganda set, check overlap with the public set, and draw a class-balanced test sample.
usage: prep_uganda.py UGANDA_ROOT RESULTS_DIR OUT_IMG_DIR N_PER_CLASS"""
import sys, os, glob, json, collections, random
import imagehash, pandas as pd
from PIL import Image, ImageOps
root, res, outd, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
Image.MAX_IMAGE_PIXELS = None
MAP = {'canker': 'canker', 'healthy': 'healthy', 'fresh': 'healthy', 'melanose': 'melanose'}
recs = []
for p in sorted(glob.glob(os.path.join(root, '**', '*.*'), recursive=True)):
    if p.lower().rsplit('.', 1)[-1] not in ('jpg', 'jpeg', 'png'): continue
    folder = os.path.basename(os.path.dirname(p)).lower()
    cls = next((v for k, v in MAP.items() if k in folder), None)
    recs.append(dict(path=p, folder=os.path.basename(os.path.dirname(p)), cls=cls))
df = pd.DataFrame(recs); print(df.groupby(['folder', 'cls'], dropna=False).size())
summary = dict(files_per_folder=df.folder.value_counts().to_dict())
df = df[df.cls.isin(['canker', 'healthy'])]
random.seed(2026)
pool = {c: list(df[df.cls == c].path) for c in ('canker', 'healthy')}
for c in pool: random.shuffle(pool[c])
def ph(im): return [imagehash.phash(x) for x in (im, ImageOps.mirror(im), im.rotate(90, expand=True), im.rotate(180), im.rotate(270, expand=True))]
pub = json.load(open(f'{res}/doh_audit.json'))
pub_h = [imagehash.phash(Image.open(os.path.join(res, '..', r['path'])).convert('RGB')) for r in pub]
os.makedirs(outd, exist_ok=True); kept = []; stats = collections.Counter()
for c in ('canker', 'healthy'):
    hashes = []
    for p in pool[c]:
        if sum(1 for k in kept if k['cls'] == c) >= N: break
        try: im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        except Exception: stats['unreadable'] += 1; continue
        hs = ph(im)
        if any(min(h - q for q in hs) <= 4 for h in hashes): stats[f'{c}_dup_within_uganda'] += 1; continue
        if any(min(h - q for q in hs) <= 4 for h in pub_h): stats[f'{c}_overlap_public'] += 1; continue
        hashes.append(hs[0]); im.thumbnail((640, 640))
        q = os.path.join(outd, c); os.makedirs(q, exist_ok=True); o = os.path.join(q, f'ug_{c}_{len(hashes):04d}.jpg')
        im.save(o, quality=95); kept.append(dict(path=os.path.abspath(o), cls=c, source_file=os.path.relpath(p, root)))
k = pd.DataFrame(kept); k.to_csv(f'{res}/uganda_test.csv', index=False)
summary.update(dict(sampled=k.cls.value_counts().to_dict(), screening=dict(stats), n_per_class_target=N))
json.dump(summary, open(f'{res}/uganda_summary.json', 'w'), indent=1); print(summary)
