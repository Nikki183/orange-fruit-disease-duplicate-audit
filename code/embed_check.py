import sys, torch, torchvision as tv, pandas as pd, numpy as np
from PIL import Image, ImageDraw
res=sys.argv[1]; df=pd.read_csv(f'{res}/dataset4.csv')
m=tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2); m.fc=torch.nn.Identity(); m.eval()
tf=tv.models.ResNet50_Weights.IMAGENET1K_V2.transforms()
E=[]
with torch.no_grad():
    for i in range(0,len(df),32):
        x=torch.stack([tf(Image.open(p).convert('RGB')) for p in df.path[i:i+32]]); E.append(m(x))
E=torch.nn.functional.normalize(torch.cat(E),dim=1).numpy(); np.save(f'{res}/emb.npy',E)
S=E@E.T; n=len(df); g=df.group.values
iu=np.triu_indices(n,1); s=S[iu]; same=g[iu[0]]==g[iu[1]]
for t in (0.90,0.93,0.95,0.97):
    sel=(s>=t)&~same
    print(t,'new pairs',sel.sum(),'cross-class',(df.cls.values[iu[0]][sel]!=df.cls.values[iu[1]][sel]).sum())
print('same-group sim quantiles',np.quantile(s[same],[0.05,0.5]))
# sheet of top non-group pairs
o=np.argsort(-np.where(same,-1,s))[:24:2]
W=160; sh=Image.new('RGB',(2*W+10,len(o)*(W+18)),'white'); d=ImageDraw.Draw(sh)
for k,ix in enumerate(o):
    a,b=iu[0][ix],iu[1][ix]
    for c,p in enumerate((df.path[a],df.path[b])): sh.paste(Image.open(p).convert('RGB').resize((W,W)),(c*(W+10),k*(W+18)+16))
    d.text((3,k*(W+18)+2),f"{s[ix]:.3f} {df.cls[a]}/{df.cls[b]}",fill='black')
sh.save(f'{res}/emb_sheet.jpg',quality=70)
