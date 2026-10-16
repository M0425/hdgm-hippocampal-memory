"""Reconstruct panel e; cluster resampling keeps all seven conditions per seed."""
from reconstructed_common import *

def draw(fig,rows):
    X,Y,kinds,seeds=centered(rows);x=X.ravel();y=Y.ravel();r=float(np.corrcoef(x,y)[0,1])
    rng=np.random.default_rng(20261016);idx=rng.integers(0,10,(12000,10))
    xb=X[idx].reshape(12000,-1);yb=Y[idx].reshape(12000,-1)
    # Every resampled block is a centered seed: no independent-trial resampling.
    rr=(xb*yb).sum(1)/np.sqrt((xb*xb).sum(1)*(yb*yb).sum(1))
    slope=(xb*yb).sum(1)/(xb*xb).sum(1);ci=np.percentile(rr,[2.5,97.5])
    stored=json.loads(PROVENANCE.read_text(encoding='utf-8'));assert abs(r-stored['coupling_r'])<1e-12
    ax=fig.add_axes([.13,.21,.83,.65]);grid=np.linspace(-.32,.29,150)
    band=np.percentile(slope[:,None]*grid,[2.5,97.5],axis=0)
    ax.fill_between(grid,band[0],band[1],color='#0077A6',alpha=.15)
    ax.plot(grid,(x@y)/(x@x)*grid,color='#0077A6',lw=1.7)
    for axis,col,marker in [('alias',ALIAS,'o'),('noise',NOISE,'D')]:
        mask=kinds.ravel()==axis;ax.scatter(x[mask],y[mask],s=22,color=col,alpha=.65,marker=marker,edgecolor='white',lw=.5,label=axis)
    ax.axhline(0,color=SLATE,ls='--',lw=.6);ax.axvline(0,color=SLATE,ls='--',lw=.6)
    ax.set(xlim=(-.32,.29),ylim=(-.32,.25),xlabel='Seed-centered history accessibility',ylabel='Seed-centered choice accuracy')
    ax.grid(color='#E4E8EC',lw=.5);ax.set_axisbelow(True);ax.legend(frameon=False,loc='lower right')
    archived_ci=stored['coupling_r_ci']
    ax.text(.02,1.04,f"Within-seed r = {r:.2f}; archived 95% CI [{archived_ci[0]:.2f}, {archived_ci[1]:.2f}]",transform=ax.transAxes,fontsize=9,color='#0077A6')
    fig.text(.02,.94,'e',fontsize=14,weight='bold');fig.text(.13,.055,'10 seed clusters; 7 conditions/seed. Archived interval shown; band uses reconstructed bootstrap.',fontsize=8)
    exports=[{'seed':s,'axis':kinds[j,k],'centered_access':float(X[j,k]),'centered_choice':float(Y[j,k])} for j,s in enumerate(seeds) for k in range(7)]
    return {'r':r,'reconstructed_ci95':ci.tolist(),'archived_ci95':stored['coupling_r_ci'],'ci_note':'Original random stream unavailable; archived interval retained separately, not substituted for recomputation.','bootstrap_seed':20261016,'bootstrap_draws':12000,'unique_conditions_per_seed':7,'n_seeds':10},exports

if __name__=='__main__':
    theme();fig=plt.figure(figsize=(7,4.1));m,r=draw(fig,load());save(fig,'figure5e_reconstructed',m,r)
