import csv,glob,collections,random,re
rows={}
import os, sys
for f in [sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'judgments.csv')]:
    for r in csv.DictReader(open(f)):
        if not r.get('cell') or not r.get('condition'): continue
        rows[(r['sheet'].strip(),r['cell'].strip())]=r
sheets=collections.Counter(s for s,_ in rows)
print('sheets',len(sheets),'cells',len(rows),'bad sheets',[s for s,n in sheets.items() if n!=28][:5])
ok=lambda r: r['accessory'].strip()=='1' and r['mismatch'].strip()=='0' and r['degraded'].strip()=='0'
und={}
for (s,c),r in rows.items():
    if r['condition']=='Undefended': und[(s,r['purification'])]=ok(r)
conds=['PhotoGuard','DCT-Shield','BlurGuard','SIFM','Mist','CAPGuard']; purs=['none','jpeg50','blur2','vae_trans']
cells=collections.defaultdict(list)  # (cond,pur) -> list of (image,prompt,fail)
for (s,c),r in rows.items():
    if r['condition']=='Undefended' or not und.get((s,r['purification'])): continue
    img,p=s.rsplit('__',1); p=p.replace('.jpg','')
    cells[(r['condition'],r['purification'])].append((img,p,0 if ok(r) else 1,r))
print('undefended success per purification:',{p:f"{sum(v for (s,pp),v in und.items() if pp==p)}/{sum(1 for (s,pp) in und if pp==p)}" for p in purs})
for pi in ['p0','p1','p2','p3']:
    print(' undef success',pi,f"{sum(v for (s,pp),v in und.items() if s.endswith(pi))}/{sum(1 for (s,pp) in und if s.endswith(pi))}")
random.seed(20260812)
def ci(lst):
    by=collections.defaultdict(list)
    for img,p,f,_ in lst: by[img].append(f)
    imgs=list(by); B=[]
    for _ in range(1000):
        smp=[random.choice(imgs) for _ in imgs]; n=sum(len(by[i]) for i in smp); B.append(100*sum(sum(by[i]) for i in smp)/max(n,1))
    B.sort(); return B[25],B[974]
print('\nISR% [95% CI] (fails/eligible)')
for c in conds:
    out=[]
    for p in purs+['purified']:
        L=cells[(c,p)] if p!='purified' else cells[(c,'jpeg50')]+cells[(c,'blur2')]+cells[(c,'vae_trans')]
        f=sum(x[2] for x in L); n=len(L); lo,hi=ci(L); out.append(f"{100*f/n:5.1f} [{lo:4.1f},{hi:4.1f}] ({f}/{n})")
    print(c.ljust(11),' | '.join(out))
print('\nper prompt, purified (jpeg50+blur2+vae_trans) ISR%:')
for c in conds:
    L=cells[(c,'jpeg50')]+cells[(c,'blur2')]+cells[(c,'vae_trans')]
    print(c.ljust(11),'  '.join(f"{pi}:{100*sum(x[2] for x in L if x[1]==pi)/max(1,sum(1 for x in L if x[1]==pi)):5.1f}%({sum(1 for x in L if x[1]==pi)})" for pi in ['p0','p1','p2','p3']))
print('\nper prompt, no purification ISR%:')
for c in conds:
    L=cells[(c,'none')]
    print(c.ljust(11),'  '.join(f"{pi}:{100*sum(x[2] for x in L if x[1]==pi)/max(1,sum(1 for x in L if x[1]==pi)):5.1f}%" for pi in ['p0','p1','p2','p3']))
print('\nfailure kinds among eligible (noAcc/mismatch/degraded), purified | none:')
for c in conds:
    def k(L): return '/'.join(str(sum(1 for x in L if x[3][f].strip()==v)) for f,v in [('accessory','0'),('mismatch','1'),('degraded','1')])
    print(c.ljust(11),k(cells[(c,'jpeg50')]+cells[(c,'blur2')]+cells[(c,'vae_trans')]),'|',k(cells[(c,'none')]))
# McNemar CAPGuard vs others, purified, paired on same eligible cells
print('\nMcNemar purified, CAPGuard vs X (b=CAP fail & X ok, c=CAP ok & X fail):')
from math import comb
key=lambda x:(x[0],x[1],x[3]['purification'])
cap={key(x):x[2] for p in ['jpeg50','blur2','vae_trans'] for x in cells[('CAPGuard',p)]}
for c in conds[:-1]:
    oth={key(x):x[2] for p in ['jpeg50','blur2','vae_trans'] for x in cells[(c,p)]}
    b=sum(1 for k2 in cap if k2 in oth and cap[k2]==1 and oth[k2]==0); cc=sum(1 for k2 in cap if k2 in oth and cap[k2]==0 and oth[k2]==1)
    n=b+cc; pval=min(1,2*sum(comb(n,i) for i in range(0,min(b,cc)+1))/2**n) if n else 1
    print(f'  vs {c}: b={b} c={cc} p={pval:.2g}')
