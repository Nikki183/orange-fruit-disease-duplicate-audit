"""Train YOLO classification models on each split and save per-image test predictions.
usage: run_exp.py RESULTS_DIR WORK_DIR MODEL SPLIT_NAME [EPOCHS] [EXTRA_TEST_CSV]"""
import sys, os, json, time, shutil, pandas as pd, numpy as np, torch
from ultralytics import YOLO
res, work, model_name, split_name = sys.argv[1:5]
epochs=int(sys.argv[5]) if len(sys.argv)>5 else 30
extra=sys.argv[6] if len(sys.argv)>6 else None
df=pd.read_csv(f'{res}/dataset4.csv'); df['path']=[os.path.abspath(os.path.join(res,'..',p)) for p in df.path]; sp={s['name']:s for s in json.load(open(f'{res}/splits.json'))}[split_name]
seed=int(split_name.split('_')[-1][1:])
tag=f'{model_name}__{split_name}'; root=os.path.join(work,'ds',tag)
if os.path.exists(root): shutil.rmtree(root)
for part in ('train','val','test'):
    for i in sp[part]:
        d=os.path.join(root,part,df.cls[i]); os.makedirs(d,exist_ok=True)
        os.symlink(df.path[i], os.path.join(d,f'{i:04d}_'+os.path.basename(df.path[i]).replace(' ','_')))
t0=time.time()
m=YOLO(f'{model_name}.pt')
m.train(data=root, epochs=epochs, imgsz=224, batch=32, seed=seed, deterministic=True, device='mps',
        project=os.path.join(work,'runs'), name=tag, exist_ok=True, verbose=False, plots=False, workers=2, patience=100)
train_s=time.time()-t0
best=YOLO(os.path.join(str(m.trainer.save_dir),'weights','best.pt')); json.dump(dict(m.trainer.metrics), open(os.path.join(str(m.trainer.save_dir),'final_metrics.json'),'w'), default=float)
names=best.names
def predict(paths):
    out=[]
    for k in range(0,len(paths),64):
        for r in best.predict(paths[k:k+64], imgsz=224, device='mps', verbose=False):
            out.append(r.probs.data.cpu().numpy())
    return np.array(out)
tp=list(df.path[sp['test']]); P=predict(tp)
rows=pd.DataFrame(dict(model=model_name, split=split_name, regime=sp['regime'], idx=sp['test'], path=tp,
    true=list(df.cls[sp['test']]), pred=[names[int(a)] for a in P.argmax(1)], conf=P.max(1),
    dup_in_train=list(df.group[sp['test']].isin(set(df.group[sp['train']])))))
for j,c in names.items(): rows[f'p_{c}']=P[:,j]
rows['source']='public'
if extra:
    ex=pd.read_csv(extra); Q=predict(list(ex.path))
    er=pd.DataFrame(dict(model=model_name, split=split_name, regime=sp['regime'], idx=-1, path=ex.path, true=ex.cls,
        pred=[names[int(a)] for a in Q.argmax(1)], conf=Q.max(1), dup_in_train=False))
    for j,c in names.items(): er[f'p_{c}']=Q[:,j]
    er['source']='uganda'; rows=pd.concat([rows,er])
# latency on 50 images, batch 1
lat=[]
for p in tp[:50]:
    t=time.perf_counter(); best.predict(p, imgsz=224, device='mps', verbose=False); lat.append((time.perf_counter()-t)*1000)
os.makedirs(f'{res}/preds',exist_ok=True)
rows.to_csv(f'{res}/preds/{tag}.csv',index=False)
json.dump(dict(tag=tag, train_s=train_s, latency_ms_median=float(np.median(lat[5:])), params=sum(p.numel() for p in best.model.parameters())),
          open(f'{res}/preds/{tag}.json','w'))
shutil.rmtree(root)
print(tag,'acc',(rows[rows.source=='public'].true==rows[rows.source=='public'].pred).mean(),'train_s',round(train_s))
