"""Reconstruct panel d: seed history accessibility and normalized access-AUC tests."""
from reconstructed_common import *

def draw(fig,rows):
    axes=fig.subplots(1,2);tests=auc_tests(rows);exports=[]
    rng=np.random.default_rng(20261016)
    for ax,axis,names,col,title in zip(axes,['alias','noise'],[ALIASES,NOISES],[ALIAS,NOISE],['Alias corruption','Observation noise']):
        for j,name in enumerate(names):
            h=np.array([access(r['conditions'][name]) for r in rows]);p=np.array([access(r['conditions'][name],'tem') for r in rows])
            ci=np.percentile(h[rng.integers(0,10,(16000,10))].mean(1),[2.5,97.5]);y=3-j
            # Gaussian display density only: all seed marks remain visible.
            xx=np.linspace(-.07,.55,400);bw=.02
            kde=np.exp(-.5*((xx[:,None]-h)/bw)**2).mean(1);kde=.65*kde/max(kde.max(),1e-12)
            ax.fill_between(xx,y,y+kde,color=HISTORY,alpha=.30);ax.plot(xx,y+kde,color=HISTORY,lw=.7)
            jitter=np.linspace(-.13,-.05,10)
            ax.scatter(h,y+jitter,s=15,facecolor='white',edgecolor=HISTORY,lw=.7,zorder=3)
            ax.scatter(p,y+jitter-.12,s=13,marker='s',facecolor='white',edgecolor=SLATE,lw=.7,zorder=3)
            ax.plot(ci,[y+.24]*2,color='#493A75',lw=1.5)
            ax.scatter(h.mean(),y+.24,s=28,marker='D',color='#493A75',edgecolor='white',lw=.5,zorder=4)
            for k,r in enumerate(rows):exports.append({'seed':r['seed'],'condition':name,'hdgm_access':float(h[k]),'pam_access':float(p[k]),'hdgm_mean':float(h.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])})
        ax.axvline(0,color=SLATE,lw=.6,ls='--');ax.set(xlim=(-.06,.55),ylim=(-.4,3.9),yticks=[3,2,1,0],yticklabels=['0','.25','.50','.75'] if axis=='alias' else ['.02','.05','.08','.12'],xlabel='History accessibility (accuracy difference)')
        ax.tick_params(axis='y',labelleft=True);ax.set_title(title,loc='left',color=col,fontsize=10,pad=22)
        ax.text(1,1.015,f"HDGM-PAM access AUC; Holm p={tests[axis]['holm_p']:.4f}",ha='right',transform=ax.transAxes,color=col,fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.80,wspace=.20)
    fig.text(.09,.065,'Circles: HDGM seeds; squares: PAM seeds; diamonds: HDGM mean. Intervals: 95% bootstrap CI (16,000 draws).',fontsize=8)
    fig.text(.02,.95,'d',fontsize=14,weight='bold')
    return {'auc':tests,'bootstrap_draws':16000,'bootstrap_seed':20261016,'n_seeds':10,'interval_note':'New deterministic bootstrap stream; original stream was not recorded.'},exports

if __name__=='__main__':
    theme();fig=plt.figure(figsize=(7.4,3.8));m,r=draw(fig,load());save(fig,'figure5d_reconstructed',m,r)
