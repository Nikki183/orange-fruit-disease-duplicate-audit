import sys,json,numpy as np,os
from PIL import Image, ImageDraw
res=sys.argv[1]
D=json.load(open(os.path.join(res,'doh_audit.json'))); M=np.load(os.path.join(res,'doh_pair.npy'))
pairs=[(i,j) for i in range(len(D)) for j in range(i+1,len(D)) if M[i,j]==0 and D[i]['cls']!=D[j]['cls']]
rauf=[r for r in D if r['rauf_d']==0][::40][:8]
rows=[(D[i]['path'],D[j]['path'],f"{D[i]['cls']} vs {D[j]['cls']}") for i,j in pairs[:8]]+[(r['path'],r['rauf_match'],'doh vs rauf') for r in rauf]
W=200; sheet=Image.new('RGB',(2*W+10,len(rows)*(W+20)),'white'); d=ImageDraw.Draw(sheet)
for k,(a,b,t) in enumerate(rows):
    for c,p in enumerate((a,b)):
        sheet.paste(Image.open(p).convert('RGB').resize((W,W)),(c*(W+10),k*(W+20)+20))
    d.text((5,k*(W+20)+4),t+' | '+os.path.basename(a)+' | '+os.path.basename(b),fill='black')
sheet.save(os.path.join(res,'sheet.jpg'),quality=70); print(len(pairs)); [print(D[i]['path'].split('/')[-1],D[j]['path'].split('/')[-1]) for i,j in pairs]
