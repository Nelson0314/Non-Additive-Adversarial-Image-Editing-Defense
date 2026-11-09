import csv,glob,collections,re
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows={}
import os, sys
for f in [sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'judgments.csv')]:
    for r in csv.DictReader(open(f)):
        if not r.get('cell') or not r.get('condition'): continue
        s=r['sheet'].strip().replace('.jpg',''); rows[(s,r['cell'].strip())]=r
ok=lambda r: r['accessory'].strip()=='1' and r['mismatch'].strip()=='0' and r['degraded'].strip()=='0'
und={(s,r['purification']):ok(r) for (s,c),r in rows.items() if r['condition']=='Undefended'}
conds=['PhotoGuard','DCT-Shield','BlurGuard','SIFM','Mist','CAPGuard']; purs=['none','jpeg50','blur2','vae_trans']
F=collections.Counter(); N=collections.Counter()
for (s,c),r in rows.items():
    if r['condition']=='Undefended' or not und.get((s,r['purification'])): continue
    p=s.rsplit('__',1)[1]; k=(p,r['condition'],r['purification']); N[k]+=1; F[k]+= 0 if ok(r) else 1
names={'p0':'Sunglasses','p1':'Police suit','p2':'Helmet','p3':'Bowtie'}
cols=['No purification','JPEG 50','Gaussian blur σ2','VAE-Trans','Purified (3 combined)']
fig,axes=plt.subplots(4,1,figsize=(12,13.5),facecolor='white')
for ax,p in zip(axes,['p0','p1','p2','p3']):
    ax.axis('off')
    elig=[sum(1 for (s,pp),v in und.items() if s.endswith(p) and pp==q and v) for q in purs]
    tot=[sum(1 for (s,pp) in und if s.endswith(p) and pp==q) for q in purs]
    cell=[]
    for c in conds:
        line=[]
        for q in purs:
            n=N[(p,c,q)]; f=F[(p,c,q)]; line.append(f'{100*f/n:.1f}% ({f}/{n})' if n else '–')
        n=sum(N[(p,c,q)] for q in purs[1:]); f=sum(F[(p,c,q)] for q in purs[1:])
        line.append(f'{100*f/n:.1f}% ({f}/{n})' if n else '–'); cell.append(line)
    t=ax.table(cellText=cell,rowLabels=conds,colLabels=cols,loc='center',cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1,1.55)
    for j in range(len(cols)): t[0,j].set_facecolor('#e8e8e8'); t[0,j].set_text_props(weight='bold')
    for j in range(len(cols)): t[len(conds),j].set_facecolor('#fff2cc')
    t[len(conds),-1].set_facecolor('#fff2cc')
    ax.set_title(f'{names[p]}  —  undefended edit success: '+', '.join(f'{a}/{b}' for a,b in zip(elig,tot))+' (none / JPEG / blur / VAE-Trans)',fontsize=11,loc='left',pad=4)
fig.suptitle('Conditional ISR on InstructPix2Pix by instruction (100 Pexels portraits, seed 20260812)\nfailures / cells where the undefended edit succeeded under the same purification',fontsize=13)
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'isr_by_prompt.png'),dpi=150,facecolor='white')
