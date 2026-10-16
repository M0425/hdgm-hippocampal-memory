"""Reconstruct panel f; 24 exact sign-flip tests form one Holm family."""
from reconstructed_common import *
from matplotlib.patches import FancyBboxPatch

def draw(fig,rows):
    A=matrix(rows);dz=(A.mean(0)/A.std(0,ddof=1)).T
    ps=np.array([exact(A[:,j,k]) for j in range(8) for k in range(3)])
    q=holm(ps).reshape(8,3).T
    old=json.loads(PROVENANCE.read_text(encoding='utf-8'));assert np.allclose(dz,old['F_dz'],atol=1e-12);assert np.allclose(q,old['F_p_holm'],atol=1e-12)
    ax=fig.add_axes([.22,.20,.75,.64]);ax.set(xlim=(-.5,8.5),ylim=(-.7,2.7));ax.axis('off')
    labels=['History retained','Branch formed','Decision successful'];colors=[HISTORY,NOISE,'#E6A400'];exports=[]
    for group,x0,col,title in [(0,0,ALIAS,'ALIAS CORRUPTION'),(1,4.8,NOISE,'OBSERVATION NOISE')]:
        ax.text(x0+1.5,2.8,title,color=col,ha='center',weight='bold',fontsize=9)
        for j in range(4):
            k=4*group+j;x=x0+j
            for row in range(3):
                y=2-row;c=colors[row] if dz[row,k]>=0 else ALIAS
                import matplotlib.colors as mc
                rgb=np.array(mc.to_rgb(c));strength=min(.70,.17+.36*abs(dz[row,k]));fill=rgb*strength+np.ones(3)*(1-strength)
                ax.add_patch(FancyBboxPatch((x-.43,y-.33),.86,.66,boxstyle='round,pad=.015,rounding_size=.04',fc=fill,ec=INK if q[row,k]<.05 else 'white',lw=1.3 if q[row,k]<.05 else .6))
                ax.text(x,y+.04,f'{dz[row,k]:+.2f}',ha='center',va='center',weight='bold',fontsize=9,color=INK)
                if q[row,k]<.05:ax.text(x+.30,y+.20,'*',ha='center',fontsize=10)
                # Seed dots show effect sign/magnitude; all ten observations retained in CSV.
                normalized=np.clip(A[:,k,row]/max(np.max(np.abs(A[:,k,row])),1e-12),-1,1)
                ax.scatter(x+np.linspace(-.32,.32,10),y-.21+.08*normalized,s=4,color=c,alpha=.8)
                for i,r in enumerate(rows):exports.append({'seed':r['seed'],'condition':NAMES[k],'metric':labels[row],'seed_value':float(A[i,k,row]),'dz':float(dz[row,k]),'holm_p':float(q[row,k])})
            ax.text(x,-.52,(['0','.25','.50','.75'] if group==0 else ['.02','.05','.08','.12'])[j],ha='center',fontsize=8)
    for j,label in enumerate(labels):ax.text(-.65,2-j,label.replace(' ','\n',1),ha='right',va='center',color=colors[j],fontsize=9,weight='bold')
    fig.text(.02,.94,'f',fontsize=14,weight='bold')
    fig.text(.22,.06,'Values: paired effect size dz. Outlined cells: Holm p < .05 across 24 exact tests.',fontsize=8)
    return {'dz':dz.tolist(),'holm_p':q.tolist(),'n_seeds':10,'tests':24,'exact_sign_flips':1024,'archived_values_match':True,'baselines':['history-reset decoding','reset separation ratio','choice chance = 0.5']},exports

if __name__=='__main__':
    theme();fig=plt.figure(figsize=(8,3.9));m,r=draw(fig,load());save(fig,'figure5f_reconstructed',m,r)
