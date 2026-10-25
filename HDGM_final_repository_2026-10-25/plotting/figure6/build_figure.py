import os, json, csv, hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path('work/mplcache').resolve())
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from PIL import Image

OUT=Path(__file__).resolve().parent
SOURCE=OUT.parents[1]/'pilot/checkpoint_diagnostics_v1/summary.json'
records=json.loads(SOURCE.read_text(encoding='utf8'))['records']
variants=['full','fixed_gates','no_memory','neither']
names=['Full','Fixed gates','No assoc. memory','Both removed']
seeds=[11,22,33]
keys=[(v,s) for v in variants for s in seeds]
lookup={(r['variant'],r['seed'],r['endpoint']):r['conditions'] for r in records}
assert len(lookup)==24
spec={'claim':'Exploratory structural controls distinguish pooled history decoding, position-conditioned decoding and behavioral use; alternative successful solutions remain visible.', 'visual_form':'four-panel paired dot plots and annotated decoding matrices','source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'metric':'test accuracy','unit':'percent','direction':'higher accuracy','order':keys,'dimensions_mm':[178,190],'uncertainty':'All three seeds individually; no inferential tests or estimated intervals','endpoints':'150 original representation endpoint; 300 post hoc decoding','references':'Astra sparse comparison/matrix grammar; global reference ranks 4,5; form references 10,13,19','outputs':['pdf','svg','png','csv']}
(OUT/'visual_spec.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
rows=[]
for e in [150,300]:
 for v,s in keys:
  for condition in ['full','reset']:
   c=lookup[v,s,e][condition]
   row=dict(variant=v,seed=s,update=e,condition=condition,pooled=100*c['history_decode'],choice_original=100*c['behavior_original']['accuracy'],choice_fresh=100*c['behavior_fresh']['accuracy'])
   row.update({f'position_{p}':100*c['by_position'][str(p)]['test_accuracy'] for p in range(6,12)})
   rows.append(row)
with (OUT/'plotted_data.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6,'text.color':'#171717','axes.labelcolor':'#171717'})
teal='#1F6F5F'; gray='#666666'
fig=plt.figure(figsize=(178/25.4,190/25.4))
axs=[fig.add_axes(rect) for rect in [[.215,.58,.275,.33],[.695,.58,.275,.33],[.215,.13,.275,.32],[.695,.13,.275,.32]]]
y=np.arange(12)
labels=[f'{names[i//3]}  {s}' if i%3==0 else str(s) for i,(v,s) in enumerate(keys)]
def base(ax,letter,title):
 ax.set_yticks(y,labels);ax.set_ylim(11.65,-.65)
 ax.tick_params(axis='y',length=0,pad=5)
 for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
 for h in [2.5,5.5,8.5]:ax.axhline(h,color='#D8D8D4',lw=.6,zorder=0)
 ax.text(-.57,1.085,letter,transform=ax.transAxes,fontsize=11,fontweight='bold')
 ax.set_title(title,loc='left',pad=22)
 ax.text(-.035,1.015,'Variant / seed',transform=ax.transAxes,ha='right',fontsize=8)
def dots(ax,letter,title,metric):
 base(ax,letter,title);ax.set_xlim(0,104);ax.set_xticks([0,50,100]);ax.axvline(50,color='#999999',ls='--',lw=.7,zorder=0)
 for i,(v,s) in enumerate(keys):
  c=lookup[v,s,300]
  if metric=='choice':
   vals=[c['full']['behavior_original']['accuracy'],c['full']['behavior_fresh']['accuracy'],c['reset']['behavior_fresh']['accuracy']]
   for val,off,marker,color,face in zip(vals,[-.20,0,.20],['s','o','x'],[gray,teal,gray],['white',teal,gray]):
    ax.plot(val*100,i+off,marker=marker,color=color,markerfacecolor=face,ms=4,mew=.8,ls='none')
  else:
   a,b=100*c['full']['history_decode'],100*c['reset']['history_decode']
   ax.plot([b,a],[i,i],color='#B6B6B0',lw=.8,zorder=1)
   ax.plot(a,i,'o',color=teal,ms=4,zorder=3);ax.plot(b,i,'x',color=gray,ms=4,zorder=3)
 ax.set_xlabel('Choice accuracy (%)' if metric=='choice' else 'History decoding (%)')
dots(axs[0],'A','Behavior at update 300','choice')
dots(axs[1],'B','Pooled decoding at update 300','decode')
handles=[Line2D([],[],marker='s',mfc='white',color=gray,ls='',ms=4,label='Original'),Line2D([],[],marker='o',color=teal,ls='',ms=4,label='Fresh'),Line2D([],[],marker='x',color=gray,ls='',ms=4,label='Reset (fresh)')]
axs[0].legend(handles=handles,loc='lower left',bbox_to_anchor=(-.03,1.005),ncol=3,frameon=False,handletextpad=.25,columnspacing=.6,fontsize=8,borderaxespad=0)
axs[1].legend(handles=[Line2D([],[],marker='o',color=teal,ls='',ms=4,label='Intact'),Line2D([],[],marker='x',color=gray,ls='',ms=4,label='Reset')],loc='lower left',bbox_to_anchor=(-.03,1.005),ncol=2,frameon=False,fontsize=8,borderaxespad=0)
cmap=LinearSegmentedColormap.from_list('accuracy',['#F4F5F3','#A8C7BD','#1F6F5F'])
for ax,e,letter in zip(axs[2:],[150,300],['C','D']):
 base(ax,letter,f'Decoding at update {e}')
 vals=np.array([[lookup[v,s,e]['full']['history_decode']]+[lookup[v,s,e]['full']['by_position'][str(p)]['test_accuracy'] for p in range(6,12)] for v,s in keys])*100
 im=ax.imshow(vals,aspect='auto',cmap=cmap,vmin=0,vmax=100,interpolation='nearest',zorder=-1)
 for i in range(12):
  for j in range(7):ax.text(j,i,f'{vals[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if vals[i,j]>=80 else '#171717')
 ax.axvline(.5,color='white',lw=3)
 ax.set_xticks(range(7),['Pool','6','7','8','9','10','11']);ax.tick_params(axis='x',length=0)
 ax.set_xlabel('Pooled probe | Position-specific probes')
 ax.spines['bottom'].set_visible(False)
 for h in [2.5,5.5,8.5]:ax.axhline(h,color='white',lw=2)
 ax.text(0,1.015,'Intact history; accuracy (%)',transform=ax.transAxes,fontsize=8)
cax=fig.add_axes([.40,.055,.30,.013]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,50,100]);cb.outline.set_visible(False);cb.ax.tick_params(length=2,labelsize=8)
fig.text(.5,.016,'Three seeds per variant • Chance = 50% • Update 300 decoding is post hoc',ha='center',fontsize=8)
for ext in ['pdf','svg']:fig.savefig(OUT/f'figure_A1.{ext}')
fig.savefig(OUT/'figure_A1.png',dpi=220)
fig.savefig(OUT/'qa_final_size.png',dpi=110)
Image.open(OUT/'qa_final_size.png').convert('L').save(OUT/'qa_grayscale.png')
fig.canvas.draw()
renderer=fig.canvas.get_renderer();outside=[]
for t in fig.findobj(matplotlib.text.Text):
 if t.get_visible() and t.get_text():
  b=t.get_window_extent(renderer)
  if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:outside.append(t.get_text())
(OUT/'qa.json').write_text(json.dumps({'source_records':len(records),'export_rows':len(rows),'out_of_canvas_text':outside,'minimum_font_pt':8,'all_values_percent':True},indent=2),encoding='utf8')
print(json.dumps({'outputs':str(OUT),'outside':outside}))


