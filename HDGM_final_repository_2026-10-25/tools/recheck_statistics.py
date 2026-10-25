from pathlib import Path
import json,csv,itertools,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'data/pythonProject32'; F=ROOT/'data/New project'; O=ROOT/'verification';O.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
S=np.array(list(itertools.product([-1,1],repeat=10)))
def test(a,b=None):
 d=np.asarray(a,dtype=float)-(0 if b is None else np.asarray(b,dtype=float)); assert len(d)==10
 return {'mean':float(d.mean()),'p':float((np.abs(S@d)>=abs(d.sum())-1e-12).mean()),'dz':float(d.mean()/d.std(ddof=1)) if d.std(ddof=1)>1e-12 else 0.0}
def holm(ps):
 order=np.argsort(ps); out=np.zeros(len(ps));v=0
 for j,k in enumerate(order):v=max(v,min(1,ps[k]*(len(ps)-j)));out[k]=v
 return out.tolist()
checks=[]
def check(name,a,b,tol=1e-9):checks.append({'name':name,'recomputed':float(a),'stored':float(b),'pass':bool(abs(a-b)<tol)})
report={};main=read(P/'fourclass_rapid_acquisition_v4_primary_results.json')
report['main']={}
for metric in ['early_aulc','acc_at_100','acc_at_150']:
 a=[r['hdgm_'+metric] for r in main['records']];b=[r['tem_'+metric] for r in main['records']]
 report['main'][metric]={'hdgm':np.mean(a),'tem':np.mean(b),**test(a,b)}
delta=np.array([r['hdgm_early_aulc']-r['tem_early_aulc'] for r in main['records']]);rng=np.random.RandomState(24680)
report['main']['audit_bootstrap_auc']={'replicates':20000,'seed':24680,'interval':np.percentile(delta[rng.randint(0,10,(20000,10))].mean(1),[2.5,97.5]).tolist(),'status':'Audit calculation; original hard-coded figure interval estimator not recorded.'}
rows=list(csv.DictReader((P/'figure2_missing_outputs/figure2_delay_transfer.csv').open(encoding='utf-8-sig')))
lookup={(int(r['seed']),r['model'],int(r['inserted_steps_per_delay'])):float(r['accuracy']) for r in rows}
seeds=[r['seed'] for r in main['records']];changes={m:[lookup[s,m,64]-lookup[s,m,0] for s in seeds] for m in ['HDGM','TEM']}
report['delay']={'HDGM_change':test(changes['HDGM']),'TEM_change':test(changes['TEM']),'difference_of_changes':test(changes['HDGM'],changes['TEM'])}
q=holm([v['p'] for v in report['delay'].values()])
for v,h in zip(report['delay'].values(),q):v['holm_if_three_contrasts']=h
report['delay']['family_status']='Three-contrast family reconstructed in audit (two within-model changes plus between-model change contrast); original plotting script hard-codes .070 without a saved family definition.'
op=read(F/'figure3_refined_v2_outputs/data/operation.json');gate=read(F/'figure3_refined_v2_outputs/data/gate.json');ca=read(F/'figure3_refined_v2_outputs/data/ca1.json')
for mode,section in op['statistics'].items():
 if 'condition_tests' not in section:continue
 for metric,conditions in section['condition_tests'].items():
  computed=[];stored=[]
  for condition,t in conditions.items():
   def val(r,cond):
    z=r['conditions'][cond]
    if metric.startswith('normal_'):return z['normal'][metric[7:]]
    if metric.startswith('long_'):return z['long'][metric[5:]]
    return z['representation'][metric]
   try: result=test([val(r,'control') for r in op['records']],[val(r,condition) for r in op['records']])
   except KeyError:continue
   check(mode+'/'+metric+'/'+condition+'/mean',result['mean'],t['mean_difference']);check(mode+'/'+metric+'/'+condition+'/p',result['p'],t['p_two_sided_exact']);computed.append(result['p']);stored.append((condition,t))
  for adj,(condition,t) in zip(holm(computed),stored):
   if 'holm_p_across_9_conditions' in t:check(mode+'/'+metric+'/'+condition+'/holm',adj,t['holm_p_across_9_conditions'])
report['gate_zero_mean_clamp']={}
for m in ['ec','ca3','ca1']:
 for kind in ['zero','mean_clamp']:
  report['gate_zero_mean_clamp'][m+'/'+kind]=test([r['conditions']['control']['normal']['accuracy'] for r in gate['records']],[r['conditions'][kind+'__'+m+'__encoding']['normal']['accuracy'] for r in gate['records']])
for fam,section,key in [('corrective_output_probe','output_probe','class'),('secondary_output_geometry','geometry','output_geometry_ratio'),('secondary_behavior_normal','behavior_normal','accuracy'),('secondary_behavior_long','behavior_long','accuracy')]:
 ps=[]; entries=[]
 for control in ca['analysis_config']['conditions'][1:]:
  d=np.array([r['conditions']['full'][section][key]-r['conditions'][control][section][key] for r in ca['records']]);v=test(d);t=ca['statistics'][fam]['full_minus_'+control]
  check(fam+'/'+control+'/mean',v['mean'],t['mean_diff']);check(fam+'/'+control+'/p',v['p'],t['p_value']); ps.append(v['p']);entries.append((control,t))
  rng=np.random.RandomState(62010);ci=np.percentile(d[rng.randint(0,10,(50000,10))].mean(1),[2.5,97.5])
  check(fam+'/'+control+'/ci_low',ci[0],t['ci95_low']);check(fam+'/'+control+'/ci_high',ci[1],t['ci95_high'])
 for h,(control,t) in zip(holm(ps),entries):check(fam+'/'+control+'/holm',h,t['holm_p'])
ps=[]
for src,t in ca['statistics']['secondary_CA1_output_integration'].items():
 name=src.removeprefix('output_minus_');v=test([r['full_representations']['output_state']['class'] for r in ca['records']],[r['full_representations'][name]['class'] for r in ca['records']]);check(src+'/mean',v['mean'],t['mean_diff']);check(src+'/p',v['p'],t['p_value']);ps.append(v['p'])
for h,(src,t) in zip(holm(ps),ca['statistics']['secondary_CA1_output_integration'].items()):check(src+'/holm',h,t['holm_p'])
report['CA1_actual_checkpoints']=sorted(set(r['source_checkpoint_audit']['checkpoint_update'] for r in ca['records']))
report['CA1_source_run_max_updates']=sorted(set(r['source_checkpoint_audit']['source_run_max_updates'] for r in ca['records']))
place=read(F/'figure4_refined_outputs/data/position.json');branch=read(F/'figure4_refined_outputs/data/branch.json')
report['place']={}
for name,sub,metric,null in [('stability','sensory_residual_population_geometry','population_map_stability','sensory_residual_shuffled_population_geometry'),('geometry','sensory_residual_alias_population','alias_centroid_to_within_ratio','sensory_residual_alias_shuffled')]:
 a=[r['primary_probes']['hdgm'][sub][metric] for r in place['records']];b=[r['primary_probes']['hdgm'][null][metric] for r in place['records']];report['place'][name]={'hdgm':np.mean(a),'null':np.mean(b),**test(a,b)}
report['branch_behavior']={m:float(np.mean([r['choice_test'][m] for r in branch['records']])) for m in ['hdgm','tem','stateless']}
report['branch_stored_tests']=branch['summary']['representation_primary_statistics_holm'];report['branch_behavior_tests']=branch['summary']['behavior_primary_statistics_holm']
diff=read(P/'hdgm_difficulty_v2_1_frozen_outputs_confirmatory/difficulty_v2_1_multiseed_results.json')
report['difficulty']={};accessps=[]
for axis in ['alias','noise']:
 vals={m:[] for m in ['hdgm','tem']}
 for r in diff['records']:
  cs=sorted([c for c in r['conditions'].values() if c['axis']==axis],key=lambda c:c['value']);x=[c['value'] for c in cs]
  for m in vals:
   y=[c['representation'][m]['reference_transfer_probe']-c['representation'][m]['history_reset_transfer_probe'] for c in cs];vals[m].append(float(np.trapezoid(y,x)/(x[-1]-x[0])))
 v=test(vals['hdgm'],vals['tem']);report['difficulty'][axis+'_access_auc']={'hdgm':np.mean(vals['hdgm']),'tem':np.mean(vals['tem']),**v};accessps.append(v['p'])
stabilityps=[]
for axis in ['alias','noise']:
 a=[r['curves']['representation'][axis]['hdgm']['stability_auc'] for r in diff['records']];b=[r['curves']['representation'][axis]['tem']['stability_auc'] for r in diff['records']];v=test(a,b);stabilityps.append(v['p'])
for axis,q in zip(['alias','noise'],holm(accessps+stabilityps)[:2]):report['difficulty'][axis+'_access_auc']['holm_with_two_stability_contrasts']=q
report['difficulty']['access_auc_status']='Access AUC differs from source primary full-history transfer AUC. Recomputed as a post hoc four-contrast family retaining both stability contrasts; p values coincide, estimands do not.'
# Recompute panel E descriptive within-seed association and all 24 panel F cells.
X=[];Y=[];Fdiff=[]
for r in diff['records']:
 cs=[c for c in r['conditions'].values() if not(c['axis']=='noise' and abs(c['value']-.05)<1e-9)]
 x=np.array([c['representation']['hdgm']['reference_transfer_probe']-c['representation']['hdgm']['history_reset_transfer_probe'] for c in cs]);y=np.array([c['behavior']['hdgm'] for c in cs]);X.append(x-x.mean());Y.append(y-y.mean())
 cells=[]
 for axis in ['alias','noise']:
  for c in sorted([c for c in r['conditions'].values() if c['axis']==axis],key=lambda c:c['value']):
   h=c['representation']['hdgm'];cells.append([h['reference_transfer_probe']-h['history_reset_transfer_probe'],h['history_dependent_separation'],c['behavior']['hdgm']-.5])
 Fdiff.append(cells)
X=np.array(X);Y=np.array(Y);report['difficulty']['coupling_r']=float(np.corrcoef(X.ravel(),Y.ravel())[0,1]);A=np.array(Fdiff); dz=np.mean(A,0)/np.std(A,axis=0,ddof=1);qs=np.array(holm([test(A[:,j,k])['p'] for j in range(8) for k in range(3)])).reshape(8,3)
prov=read(F/'figure5_integrated_BC_v2_provenance.json');report['figure5_provenance_r']=prov['coupling_r'];report['figure5_provenance_ci']=prov['coupling_r_ci'];report['F_recomputed_dz']=dz.T.tolist();report['F_provenance_dz']=prov['F_dz'];report['F_recomputed_q']=qs.T.tolist();report['F_provenance_q']=prov['F_p_holm']
report['checks']=checks; report['check_count']=len(checks);report['failed']=[x for x in checks if not x['pass']]
(O/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['checks','branch_stored_tests','branch_behavior_tests','gate_zero_mean_clamp','F_provenance_dz','F_provenance_q']},indent=2))

