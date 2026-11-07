"""Train two-class (canker/healthy) YOLO11n-cls on Uganda images; test within Uganda and on the public collection.
usage: run_uganda_trained.py RESULTS_DIR WORK_DIR SEED"""
import sys, os, shutil, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ultralytics import YOLO
res, work, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); base = os.path.join(res, '..')
K = pd.read_csv(f'{res}/uganda_train.csv'); tr, va = train_test_split(K.index, test_size=0.15, stratify=K.cls, random_state=seed)
root = os.path.join(work, 'ds', f'ug_s{seed}'); shutil.rmtree(root, ignore_errors=True)
for part, ix in (('train', tr), ('val', va), ('test', va[:4])):
    for i in ix:
        d = os.path.join(root, part, K.cls[i]); os.makedirs(d, exist_ok=True)
        os.symlink(os.path.abspath(os.path.join(base, K.path[i])), os.path.join(d, os.path.basename(K.path[i])))
m = YOLO('yolo11n-cls.pt')
m.train(data=root, epochs=30, imgsz=224, batch=32, seed=seed, deterministic=True, device='mps', project=os.path.join(work, 'runs'), name=f'yolo11n-cls__ugandatrain_s{seed}', exist_ok=True, verbose=False, plots=False, workers=2)
best = YOLO(os.path.join(str(m.trainer.save_dir), 'weights', 'best.pt')); nm = best.names
def evaluate(paths, truth, src):
    pr = []
    for k in range(0, len(paths), 64): pr += [nm[int(r.probs.top1)] for r in best.predict(paths[k:k+64], imgsz=224, device='mps', verbose=False)]
    return pd.DataFrame(dict(model='yolo11n-cls', split=f'ugandatrain_s{seed}', regime='ugandatrain', path=paths, true=truth, pred=pr, source=src))
U = pd.read_csv(f'{res}/uganda_test.csv'); D = pd.read_csv(f'{res}/dataset4.csv'); D = D[D.cls.isin(['canker', 'healthy'])]
out = pd.concat([evaluate([os.path.join(base, p) for p in U.path], list(U.cls), 'uganda_test'), evaluate([os.path.join(base, p) for p in D.path], list(D.cls), 'public')])
out.to_csv(f'{res}/preds/ugandatrain__s{seed}.csv', index=False); shutil.rmtree(root)
for s, d in out.groupby('source'): print(seed, s, 'macroF1=%.3f' % f1_score(d.true, d.pred, average='macro'), 'acc=%.3f' % (d.true == d.pred).mean())
