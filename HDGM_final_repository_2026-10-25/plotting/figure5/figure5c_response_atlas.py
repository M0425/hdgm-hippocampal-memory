"""Seed-resolved behavioral response atlas; source values never interpolated.

Visual spec: user reference 6 (heatmap + aligned summaries), reference 1
(muted palette). Three models, eight conditions, ten seeds in numeric order.
178 x 122 mm. Absolute choice accuracy, common 0-1 linear color scale;
summary markers show mean and 95% Student-t CI across independent seeds.
Paired conditions remain aligned by seed; no clustering or seed selection.
"""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
from rebuild_figure5_panel_c_dotmatrix import DATA

OUT = Path(__file__).parent / 'figure5C_response_atlas_zoom'
MODELS = [('hdgm','HDGM','#0072B2','o'), ('tem','TEM','#475569','s'),
          ('stateless','stateless','#7C8796','X')]
KEYS = ['alias_0.000','alias_0.250','alias_0.500','alias_0.750',
        'noise_0.020','noise_0.050','noise_0.080','noise_0.120']
LABELS = ['0','.25','.50','.75','.02','.05','.08','.12']
X = np.array([0,1,2,3,4.6,5.6,6.6,7.6])
CMAP = LinearSegmentedColormap.from_list('choice_accuracy',
    [(0,'#B66699'),(.25,'#DEC1CE'),(.5,'#F5F4F6'),(.75,'#B7A6C7'),(1,'#584577')])

def main():
    records = sorted(json.loads(DATA.read_text(encoding='utf-8'))['records'],key=lambda r:r['seed'])
    assert len(records)==10 and len(set(r['seed'] for r in records))==10
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.6,
        'text.color':'#202532','axes.labelcolor':'#202532'})
    fig=plt.figure(figsize=(178/25.4,122/25.4),facecolor='white')
    fig.text(.025,.96,'C',fontsize=13,weight='bold',va='top')
    fig.text(.105,.96,'Seed-resolved behavioral responses',fontsize=10,weight='bold',va='top')
    fig.text(.105,.91,'Same seed order across models and perturbations',fontsize=8,color='#59616C')
    axes=[]; export=[]; summaries=[]
    for m,(model,title,color,marker) in enumerate(MODELS):
        left=.105+m*.297; w=.265
        hm=fig.add_axes([left,.415,w,.36]); axes.append(hm)
        a=np.array([[r['conditions'][k]['behavior'][model] for k in KEYS] for r in records])
        assert np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
        hm.pcolormesh(np.arange(-.5,4.5),np.arange(-.5,10.5),a[:,:4],
            cmap=CMAP,norm=Normalize(0,1),edgecolors='white',linewidth=.5)
        mesh=hm.pcolormesh(np.arange(4.1,9.1),np.arange(-.5,10.5),a[:,4:],
            cmap=CMAP,norm=Normalize(0,1),edgecolors='white',linewidth=.5)
        hm.set(xlim=(-.5,8.1),ylim=(9.5,-.5),xticks=X,xticklabels=LABELS)
        hm.set_yticks(range(10))
        hm.set_yticklabels([str(r['seed']) for r in records] if m==0 else [])
        if m==0: hm.set_ylabel('Seed',labelpad=2)
        hm.tick_params(length=0,pad=3)
        for s in hm.spines.values():s.set_visible(False)
        fig.text(left+w/2,.851,title,color=color,fontsize=10,weight='bold',ha='center')
        hm.text(1.5,1.055,r'Alias $\alpha$',transform=hm.get_xaxis_transform(),ha='center',color='#AD5062')
        hm.text(6.1,1.055,r'Noise $\sigma$',transform=hm.get_xaxis_transform(),ha='center',color='#267A68')
        # Baseline plots retain a common zoomed scale and room for y tick labels.
        ax=fig.add_axes([left + (.014 if m else 0),.14,w - (.014 if m else 0),.17]); axes.append(ax)
        mu=a.mean(axis=0); ci=2.2621571627409915*a.std(axis=0,ddof=1)/np.sqrt(10)
        ax.axhline(.5,ls=(0,(3,3)),lw=.8,color='#9199A3',zorder=0)
        ax.errorbar(X,mu,yerr=ci,fmt=marker,ls='none',ms=4,mfc='white',mec=color,
            mew=1.1,ecolor=color,elinewidth=.9,capsize=2,zorder=3)
        limits=(.40,.82) if m==0 else (.47,.56)
        ticks=[.4,.5,.6,.8] if m==0 else [.48,.50,.52,.54,.56]
        ax.set(xlim=(-.5,8.1),ylim=limits,xticks=X,xticklabels=LABELS,yticks=ticks)
        assert np.all(mu-ci>=limits[0]) and np.all(mu+ci<=limits[1]), 'CI exceeds axis limits'
        if m:
            ax.set_yticklabels([f'{v:.2f}' for v in ticks])
            ax.set_title('Zoom: 0.47–0.56',fontsize=8,color='#59616C',pad=6)
        ax.tick_params(length=2,pad=3)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#B5BAC1')
        if m==0: ax.set_ylabel('Mean accuracy',labelpad=2)
        for j,k in enumerate(KEYS):
            summaries.append(dict(model=model,condition=k,n=10,mean=float(mu[j]),
                ci95_low=float(mu[j]-ci[j]),ci95_high=float(mu[j]+ci[j])))
            for i,r in enumerate(records):
                export.append(dict(model=model,seed=r['seed'],condition=k,choice_accuracy=float(a[i,j])))
        if m==0: print(title, np.round(mu,6).tolist())
    fig.text(.105,.355,'Across-seed mean and 95% CI',fontsize=8,weight='bold')
    fig.text(.978,.355,'Dashed line: chance (0.50)',ha='right',fontsize=8,color='#59616C')
    cax=fig.add_axes([.30,.053,.38,.018])
    cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[0,.5,1])
    cb.outline.set_visible(False);cb.ax.tick_params(length=2,pad=2,labelsize=8)
    fig.text(.105,.057,'Choice accuracy',fontsize=8,va='center')
    fig.text(.715,.057,'0.50 = chance',fontsize=8,va='center',color='#59616C')
    fig.canvas.draw()
    for suffix in ['png','svg','pdf']:
        fig.savefig(str(OUT)+'.'+suffix,dpi=360,facecolor='white')
    # Renderer-level grayscale QA; independent of image editing workflows.
    original=[collection.get_cmap() for ax in axes for collection in ax.collections]
    graycolors=[]
    for rgba in CMAP(np.linspace(0,1,256)):
        l=float(np.dot(rgba[:3],[.2126,.7152,.0722]));graycolors.append((l,l,l,1))
    gray=LinearSegmentedColormap.from_list('gray_accuracy',graycolors)
    for ax in axes:
        for collection in ax.collections:
            if collection.get_array() is not None:collection.set_cmap(gray)
    fig.savefig(str(OUT)+'_qa_gray.png',dpi=170)
    for name,rows in [('source_values',export),('summary',summaries)]:
        with open(str(OUT)+'_'+name+'.csv','w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta={'source':str(DATA),'sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
        'shape':[3,10,8],'seed_order':[r['seed'] for r in records],
        'scale':'Absolute accuracy, linear 0-1, identical for every heatmap',
        'CI':'Two-sided Student t 95% CI of the mean over 10 seeds, df=9; unadjusted',
        'summary_y_limits':{'hdgm':[.40,.82],'tem':[.47,.56],'stateless':[.47,.56]},
        'baseline':'alias=0 and noise=.05 shown in both scans; not pooled as independent observations',
        'reference':'User supplied image 6: heatmaps with aligned summaries; image 1: muted colors',
        'dimensions_mm':[178,122]}
    Path(str(OUT)+'_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Exported 240 source observations and 24 means/intervals:',OUT)

if __name__=='__main__':main()
