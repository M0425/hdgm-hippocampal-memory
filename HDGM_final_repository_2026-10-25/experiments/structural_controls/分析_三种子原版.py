# -*- coding: utf-8 -*-
"""HDGM 2x2 structure pilot. Default: seeds 11,22,33; --debug: engineering only.
Run using the project .venv-hdgm interpreter. No new packages required.
"""
import argparse, copy, csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
from dataclasses import asdict
from typing import Dict
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl_cache'))
import torch
import numpy as np
spec=importlib.util.spec_from_file_location('hdgm_original_splitter',ROOT/'分叉.py')
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)

nn=b.nn
Optional=b.Optional
SplitterV7Config=b.SplitterV7Config
apply_row_reset=b.apply_row_reset

class FactorialCore(nn.Module):
    def __init__(self, cfg: SplitterV7Config, memory=True, gates=True):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.memory_enabled = memory
        self.gates_enabled = gates
        self.fc_x = nn.Linear(cfg.obs_dim, h)
        self.action_emb = nn.Embedding(7, cfg.action_emb_dim)
        self.fc_a = nn.Linear(cfg.action_emb_dim, h)
        self.fc_g = nn.Linear(h, h)
        self.decoder = nn.Linear(h, cfg.obs_dim)
        self.gate_ec = nn.Linear(cfg.obs_dim + h + h, cfg.obs_dim)
        self.gate_ca3 = nn.Linear(h * 2, h)
        self.gate_ca1 = nn.Linear(h * 3, h)
        self.fb_proj = nn.Linear(h, h)
        self.pos_head = nn.Linear(h, cfg.corridor_len)
        self.phase_head = nn.Linear(h, 6)

    def forward(self, x: torch.Tensor, action: torch.Tensor,
                reset_mask: Optional[torch.Tensor] = None,
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        b, tmax, _ = x.shape
        hdim = self.cfg.hidden_dim
        g = x.new_zeros(b, hdim)
        M = x.new_zeros(b, hdim, hdim)
        v = x.new_zeros(b, hdim)
        pred_loss = x.new_zeros(())
        mem_hist, pred_hist, attr_hist, read_hist, fb_hist = [], [], [], [], []
        ec_hist, ca3_hist, ca1_hist, write_hist = [], [], [], []
        xhat_hist, g_hist, pos_logits_hist, phase_logits_hist = [], [], [], []
        for t in range(tmax - 1):
            r = None if reset_mask is None else reset_mask[:, t]
            g, M, v = apply_row_reset(g, r), apply_row_reset(M, r), apply_row_reset(v, r)
            ec = torch.sigmoid(self.gate_ec(torch.cat([x[:, t], g, v], 1)))
            if not self.gates_enabled:
                ec = torch.ones_like(ec)
            x_emb = torch.tanh(self.fc_x(ec * x[:, t]))
            a_emb = self.fc_a(self.action_emb(action[:, t]))
            g_pred = torch.tanh(self.fc_g(g) + a_emb)
            hstate = x_emb
            ca3_steps = []
            for _ in range(self.cfg.ca3_steps):
                ca3 = torch.sigmoid(self.gate_ca3(torch.cat([hstate, g], 1)))
                if not self.gates_enabled:
                    ca3 = torch.ones_like(ca3)
                recurrent = torch.tanh(
                    0.5 * hstate + torch.bmm(M, hstate.unsqueeze(2)).squeeze(2)
                )
                hstate = (1.0 - ca3) * hstate + ca3 * recurrent
                ca3_steps.append(ca3)
            p_t = hstate
            g = g_pred + self.cfg.alpha * p_t
            p_hat = torch.tanh(0.5 * g + torch.bmm(M, g.unsqueeze(2)).squeeze(2))
            ca1 = torch.sigmoid(self.gate_ca1(torch.cat([g, p_hat, v], 1)))
            if not self.gates_enabled:
                ca1 = torch.ones_like(ca1)
            p_out = torch.relu(p_hat * (self.cfg.c1 + self.cfg.c2 * ca1) - self.cfg.beta)
            x_hat = self.decoder(p_out)
            pos_logits = self.pos_head(p_out)
            phase_logits = self.phase_head(p_out)
            pred_loss = pred_loss + (x_hat - x[:, t + 1]).square().mean()
            v = self.cfg.rho * v + self.fb_proj(p_out)
            write = ec.mean(1, keepdim=True).unsqueeze(-1)
            M = self.cfg.lambda_m * M + write * torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1)) if self.memory_enabled else torch.zeros_like(M)
            mem_hist.append(p_out)
            pred_hist.append(g_pred)
            attr_hist.append(p_t)
            read_hist.append(p_hat)
            fb_hist.append(v)
            ec_hist.append(ec)
            ca3_hist.append(torch.stack(ca3_steps, 1).mean(1))
            ca1_hist.append(ca1)
            write_hist.append(write.squeeze(-1).squeeze(-1))
            xhat_hist.append(x_hat)
            g_hist.append(g)
            pos_logits_hist.append(pos_logits)
            phase_logits_hist.append(phase_logits)
        out = {
            "pred_loss": pred_loss / max(1, tmax - 1),
            "mem": torch.stack(mem_hist, 1),
            "predictive_state": torch.stack(pred_hist, 1),
            "attractor_state": torch.stack(attr_hist, 1),
            "associative_read": torch.stack(read_hist, 1),
            "feedback_state": torch.stack(fb_hist, 1),
            "ec_gate_pattern": torch.stack(ec_hist, 1),
            "ca3_gate_pattern": torch.stack(ca3_hist, 1),
            "ca1_gate_pattern": torch.stack(ca1_hist, 1),
            # Canonical paper-model aliases retained verbatim in spirit.
            "xhat": torch.stack(xhat_hist, 1),
            "g_hist": torch.stack(g_hist, 1),
            "ec_hist": torch.stack(ec_hist, 1),
            "ca3_hist": torch.stack(ca3_hist, 1),
            "ca1_hist": torch.stack(ca1_hist, 1),
            "write_hist": torch.stack(write_hist, 1),
            "v_hist": torch.stack(fb_hist, 1),
            "pos_logits_hist": torch.stack(pos_logits_hist, 1),
            "phase_logits_hist": torch.stack(phase_logits_hist, 1),
        }
        return out



VARIANTS=[('full',True,True),('fixed_gates',True,False),('no_memory',False,True),('neither',False,False)]
def save_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(path)
def checkpoint(path,data):
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)
def probes(model,banks,cfg,seed):
    sets={name:{s:b.collect_alias_dataset(model,banks['probe_'+s],'mem',reset) for s in ('train','val','test')} for name,reset in [('full',False),('reset',True)]}
    result={}
    for name,condition,feature in [('full','full','X'),('reset','reset','X'),('observation','full','obs'),('timing','full','timing')]:
        data=sets[condition]
        def x(split):return b.timing_features(data[split],cfg) if feature=='timing' else data[split][feature]
        result[name]=b.fit_probe(x('train'),data['train']['y'],x('val'),data['val']['y'],x('test'),data['test']['y'],cfg,9000+seed)
    result['access_difference']=result['full']-result['reset']
    for condition in sets:result[condition+'_geometry']=b.geometry_metrics(sets[condition]['test'])
    return result
def preflight(cfg,env,initial):
    data=b.to_device(env.clone(66601).sample_batch(cfg.batch_size,balanced=True))
    audit=b.task_integrity_audit(data,cfg)
    original=b.SequenceChoiceModel(b.HDGMCore(cfg),cfg.hidden_dim).to(b.DEVICE)
    candidate=b.SequenceChoiceModel(FactorialCore(cfg),cfg.hidden_dim).to(b.DEVICE)
    original.load_state_dict(initial);candidate.load_state_dict(initial)
    with torch.no_grad():
        a=original(data['x'],data['action']);c=candidate(data['x'],data['action'])
        delta=max(float((a[k]-c[k]).abs().max()) for k in a)
    assert delta==0, 'Full variant differs from original'
    audit['full_original_maxdiff']=delta
    return audit
def run_seed(seed,cfg,args,output):
    folder=output/f'seed_{seed}';folder.mkdir(exist_ok=True)
    b.set_global_seed(seed)
    env=b.CounterbalancedBifurcationEnv(cfg,seed+17);banks=b.make_banks(env,cfg,seed)
    initial=copy.deepcopy(b.SequenceChoiceModel(FactorialCore(cfg),cfg.hidden_dim).state_dict())
    save_json(folder/'preflight.json',preflight(cfg,env,initial))
    results={}
    for name,memory,gates in VARIANTS:
        resultpath=folder/(name+'.json')
        if resultpath.exists():results[name]=json.loads(resultpath.read_text(encoding='utf8'));continue
        model=b.SequenceChoiceModel(FactorialCore(cfg,memory,gates),cfg.hidden_dim).to(b.DEVICE);model.load_state_dict(initial)
        opt=b.build_optimizer('hdgm',model,cfg);history=[];representation=None;start=0
        cp=folder/(name+'_latest.pt')
        if cp.exists():
            state=torch.load(cp,map_location=b.DEVICE,weights_only=False)
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['update'];history=state['history'];representation=state['representation']
            print(f'RESUME seed={seed} {name} update={start}',flush=True)
        train_env=env.clone(90000+seed)
        for _ in range(start):train_env.sample_batch(cfg.batch_size,balanced=True)
        for update in range(start+1,cfg.behavior_update+1):
            model.train();batch=b.to_device(train_env.sample_batch(cfg.batch_size,balanced=True));opt.zero_grad(set_to_none=True)
            t=time.perf_counter();loss,states=b.batch_loss(model,batch,cfg)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {seed}, {name}, {update}')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip,error_if_nonfinite=True)
            if not gates:assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for n,p in model.core.named_parameters() if n.startswith('gate_'))
            opt.step();torch.cuda.synchronize()
            row={'update':update,'loss':float(loss.detach()),'gradient_norm':float(norm),'seconds':time.perf_counter()-t};history.append(row)
            del loss,states,batch
            if update in cfg.checkpoints:row['test_accuracy']=b.evaluate_choice(model,banks['test'],'hdgm')
            if update==cfg.representation_update:
                checkpoint(folder/f'{name}_representation.pt',b.state_dict_cpu(model))
                representation=probes(model,banks,cfg,seed)
                print(f'PROBES seed={seed} {name}: {representation}',flush=True)
            if update%25==0 or update in cfg.checkpoints:
                checkpoint(cp,{'model':b.state_dict_cpu(model),'optimizer':opt.state_dict(),'update':update,'history':history,'representation':representation})
                print(f'seed={seed} {name} {update}/{cfg.behavior_update} loss={row["loss"]:.5f} test={row.get("test_accuracy","-")}',flush=True)
                if args.debug_stop_after and update>=args.debug_stop_after:return None
        results[name]={'seed':seed,'variant':name,'representation':representation,'behavior':history[-1]['test_accuracy'],'history':history}
        save_json(resultpath,results[name]);del model,opt;torch.cuda.empty_cache()
    save_json(folder/'results.json',results);return results
def summarize(records,output):
    rows=[];interactions=[]
    for seed,models in records.items():
        for name,v in models.items():
            r=v['representation'];rows.append({'seed':seed,'model':name,'history_decode':r['full'],'reset_decode':r['reset'],'access_difference':r['access_difference'],'observation_decode':r['observation'],'timing_decode':r['timing'],'behavior':v['behavior'],'separation_full':r['full_geometry']['separation_ratio'],'separation_reset':r['reset_geometry']['separation_ratio']})
        d={r['model']:r for r in rows if r['seed']==seed}
        interactions.append({'seed':seed,**{k:d['full'][k]-d['fixed_gates'][k]-d['no_memory'][k]+d['neither'][k] for k in ['history_decode','access_difference','behavior']}})
    for filename,data in [('metrics.csv',rows),('interactions.csv',interactions)]:
        with (output/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save_json(output/'summary.json',{'records':records,'interactions':interactions,'interpretation':'Development pilot; no significance tests. Three seeds do not establish a reliable interaction.'})
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug',action='store_true',help='one seed, two updates per model, tiny probe banks; never scientific results')
    parser.add_argument('--debug-stop-after',type=int,default=0,help='debug-only simulated interruption for resume testing')
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.debug_stop_after and not args.debug:parser.error('--debug-stop-after requires --debug')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable; select the configured .venv-hdgm interpreter.')
    cfg=b.SplitterV7Config();seeds=[11] if args.debug else args.seeds
    if len(set(seeds))!=len(seeds):parser.error('Duplicate seeds')
    if args.debug:
        cfg.batch_size=2;cfg.bank_batch_size=2;cfg.val_batches=1;cfg.test_batches=1
        cfg.probe_train_batches=cfg.probe_val_batches=cfg.probe_test_batches=1
        cfg.probe_epochs=2;cfg.probe_patience=2;cfg.checkpoints=(1,2);cfg.representation_update=1;cfg.behavior_update=2
    output=args.output or ROOT/('factorial_debug' if args.debug else 'factorial_3seed_outputs')
    output.mkdir(parents=True,exist_ok=True)
    protocol={'debug':args.debug,'seeds':seeds,'config':asdict(cfg),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['分析.py','分叉.py']},'development_only':True,'definition':'memory off keeps M=0; gates off fixes all three gates to1; predictive recurrence retained','endpoints':'full and independently fitted reset history probes at representation_update; behavior at behavior_update; same probe seed across variants','no_inferential_tests':True}
    protocol=json.loads(json.dumps(protocol))
    pp=output/'protocol.json'
    if pp.exists() and json.loads(pp.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Protocol/source changed. Use a new --output directory; prior results were preserved.')
    save_json(pp,protocol)
    print(f'START debug={args.debug}, seeds={seeds}, GPU={protocol["gpu"]}, output={output}',flush=True)
    records={}
    for seed in seeds:
        record=run_seed(seed,cfg,args,output)
        if record is None:print('DEBUG interruption saved; rerun without --debug-stop-after',flush=True);return
        records[str(seed)]=record
    summarize(records,output);print('COMPLETE',flush=True)
if __name__=='__main__':main()
