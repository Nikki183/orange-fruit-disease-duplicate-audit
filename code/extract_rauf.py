import sys, io, os, pandas as pd
from PIL import Image
pq, out = sys.argv[1], sys.argv[2]
df = pd.read_parquet(pq)
lab = ['black_spot','canker','greening','healthy','melanose','scab']
df = df[df.plant_part==0]
print(df.label.map(lambda i: lab[i]).value_counts())
for i,(img,l) in enumerate(zip(df.image, df.label)):
    d=os.path.join(out,lab[l]); os.makedirs(d,exist_ok=True)
    im=Image.open(io.BytesIO(img['bytes'])); 
    if i<3: print(im.size, im.mode, img.get('path'))
    im.convert('RGB').save(os.path.join(d,f'rauf_{lab[l]}_{i:04d}.png'))
