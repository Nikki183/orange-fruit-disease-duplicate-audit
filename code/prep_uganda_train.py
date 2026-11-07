"""Draw an Uganda training pool disjoint from the Uganda test set (pHash <= 4 and ResNet-50 cosine >= 0.95 excluded).
usage: prep_uganda_train.py UGANDA_RAW_ROOT RESULTS_DIR OUT_DIR N_PER_CLASS"""
import sys, os, glob, random, json, numpy as np, pandas as pd, imagehash, torch, torchvision as tv
from PIL import Image, ImageOps
root, res, outd, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); Image.MAX_IMAGE_PIXELS = None
T = pd.read_csv(f'{res}/uganda_test.csv'); used = set(T.source_file)
tpaths = [os.path.join(res, '..', p) for p in T.path]
th = [imagehash.phash(Image.open(p).convert('RGB')) for p in tpaths]
random.seed(7); kept = []; stats = {'skipped_test_file': 0, 'phash_near_test_or_kept': 0}
for c, folder in (('canker', 'Citrus_Canker'), ('healthy', 'Healthy')):
    pool = sorted(glob.glob(os.path.join(root, 'Orange_Citrus_Disease_Dataset', folder, '*'))); random.shuffle(pool); hs = []
    for p in pool:
        if sum(k['cls'] == c for k in kept) >= N: break
        rel = os.path.relpath(p, root)
        if rel in used: stats['skipped_test_file'] += 1; continue
        try: im = ImageOps.exif_transpose(Image.open(p)).convert('RGB')
        except Exception: continue
        h = imagehash.phash(im)
        if any(h - q <= 4 for q in th) or any(h - q <= 4 for q in hs): stats['phash_near_test_or_kept'] += 1; continue
        hs.append(h); im.thumbnail((640, 640)); d = os.path.join(outd, c); os.makedirs(d, exist_ok=True)
        o = os.path.join(d, f'ugtr_{c}_{len(hs):04d}.jpg'); im.save(o, quality=95); kept.append(dict(path=o, cls=c, source_file=rel))
K = pd.DataFrame(kept)
m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity(); m.eval(); tf = tv.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
def emb(ps):
    out = []
    with torch.no_grad():
        for i in range(0, len(ps), 32): out.append(m(torch.stack([tf(Image.open(p).convert('RGB')) for p in ps[i:i+32]])))
    return torch.nn.functional.normalize(torch.cat(out), dim=1).numpy()
Et, Ek = emb(tpaths), emb(list(K.path)); near = (Ek @ Et.T).max(1) >= 0.95
stats['embedding_near_test_removed'] = int(near.sum()); K = K[~near].reset_index(drop=True)
K['path'] = [os.path.relpath(p, os.path.join(res, '..')) for p in K.path]
K.to_csv(f'{res}/uganda_train.csv', index=False); stats['kept'] = K.cls.value_counts().to_dict()
json.dump(stats, open(f'{res}/uganda_train_summary.json', 'w'), indent=1); print(stats)
