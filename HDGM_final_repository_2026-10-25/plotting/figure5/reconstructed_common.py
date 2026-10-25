"""Reconstructed 2026-10-16 from saved seed results; NOT recovered historical code.

All estimands follow the archived Figure5 numerical audit. New bootstrap draws
use seed 20261016; original unknown random streams are not claimed recovered.
"""
from pathlib import Path
import csv, hashlib, itertools, json, os
import numpy as np
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/pythonProject32/hdgm_difficulty_v2_1_frozen_outputs_confirmatory/difficulty_v2_1_multiseed_results.json'
PROVENANCE=ROOT/'data/New project/figure5_integrated_BC_v2_provenance.json'
OUT=Path(__file__).resolve().parent/'reconstructed_outputs'
ALIASES=['alias_0.000','alias_0.250','alias_0.500','alias_0.750']
NOISES=['noise_0.020','noise_0.050','noise_0.080','noise_0.120']
NAMES=ALIASES+NOISES
INK='#24324A';SLATE='#64748B';HISTORY='#7961AC';ALIAS='#D1495B';NOISE='#009E73'
SIGNS=np.array(list(itertools.product([-1,1],repeat=10)))

def load():
    d=json.loads(DATA.read_text(encoding='utf-8-sig'))
    rows=sorted(d['records'],key=lambda r:r['seed'])
    assert len(rows)==10 and len({r['seed'] for r in rows})==10
    return rows

def access(c,model='hdgm'):
    r=c['representation'][model]
    return r['reference_transfer_probe']-r['history_reset_transfer_probe']

def holm(ps):
    order=np.argsort(ps);out=np.zeros(len(ps));last=0
    for j,i in enumerate(order):
        last=max(last,min(1,float(ps[i])*(len(ps)-j)));out[i]=last
    return out

def exact(x):
    x=np.asarray(x,float)
    return float(np.mean(np.abs(SIGNS@x)>=abs(x.sum())-1e-12))

def auc_tests(rows):
    arrays={};p=[]
    for axis in ['alias','noise']:
        bymodel={m:[] for m in ['hdgm','tem']}
        for r in rows:
            cs=sorted([c for c in r['conditions'].values() if c['axis']==axis],key=lambda c:c['value'])
            x=np.array([c['value'] for c in cs])
            for m in bymodel:bymodel[m].append(float(np.trapezoid([access(c,m) for c in cs],x)/(x[-1]-x[0])))
        delta=np.array(bymodel['hdgm'])-bymodel['tem'];p.append(exact(delta));arrays[axis]={'hdgm':bymodel['hdgm'],'pam':bymodel['tem'],'difference':float(delta.mean())}
    for axis in ['alias','noise']:
        delta=[r['curves']['representation'][axis]['hdgm']['stability_auc']-r['curves']['representation'][axis]['tem']['stability_auc'] for r in rows]
        p.append(exact(delta))
    q=holm(p)
    for j,axis in enumerate(['alias','noise']):arrays[axis].update(p=p[j],holm_p=float(q[j]))
    return arrays

def centered(rows):
    X=[];Y=[];kinds=[];seeds=[]
    for r in rows:
        # noise_.05 is the shared baseline already represented by alias_0.
        cs=[c for c in r['conditions'].values() if not(c['axis']=='noise' and abs(c['value']-.05)<1e-9)]
        assert len(cs)==7
        x=np.array([access(c) for c in cs]);y=np.array([c['behavior']['hdgm'] for c in cs])
        X.append(x-x.mean());Y.append(y-y.mean());kinds.append([c['axis'] for c in cs]);seeds.append(r['seed'])
    return np.array(X),np.array(Y),np.array(kinds),seeds

def matrix(rows):
    return np.array([[[access(r['conditions'][name]),r['conditions'][name]['representation']['hdgm']['history_dependent_separation'],r['conditions'][name]['behavior']['hdgm']-.5] for name in NAMES] for r in rows])

def theme():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})

def save(fig,stem,metrics,records):
    OUT.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(stem+'.'+ext),dpi=200,facecolor='white')
    from PIL import Image
    Image.open(OUT/(stem+'.png')).convert('L').save(OUT/(stem+'_gray.png'))
    metrics.update(source_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),reconstruction='New code, not recovered historical implementation',packaging_matplotlib=matplotlib.__version__,packaging_numpy=np.__version__)
    (OUT/(stem+'_statistics.json')).write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    if records:
        with (OUT/(stem+'_data.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    plt.close(fig)
