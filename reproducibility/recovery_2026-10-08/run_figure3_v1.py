"""Prepare a recovered evaluation command. Executes only with --execute."""
from pathlib import Path
import argparse,json,subprocess,sys,shlex
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint-dir',type=Path)
 p.add_argument('--output-dir',type=Path,default=HERE/'fresh_v1_evaluation')
 p.add_argument('--device',default='cpu')
 p.add_argument('--quick',action='store_true')
 p.add_argument('--execute',action='store_true')
 a=p.parse_args()
 checkpoint_dir=a.checkpoint_dir
 if checkpoint_dir is None and (HERE/"weights/v4").is_dir():checkpoint_dir=HERE/"weights/v4"
 if checkpoint_dir is None:
  records=json.loads((HERE/'SELECTED_WEIGHT_METADATA.json').read_text(encoding='utf8'))
  checkpoint_dir=Path(next(x['file'] for x in records if x['group']=='paper_rapid_v4_outputs')).parent
 if not checkpoint_dir.is_dir():raise SystemExit('Checkpoint directory unavailable; supply --checkpoint-dir.')
 # Preserve the previously accepted JSON/CSV/NPZ results.
 if a.execute and a.output_dir.exists() and any(a.output_dir.iterdir()):raise SystemExit('Choose an empty output directory to preserve previous runs.')
 command=[sys.executable,str(HERE/'code/HDGM_v4_stage_specific_perturbation_v1.py'),'--main-code',str(HERE/'code/HDGM_v4_rapid_acquisition_embedded.py'),'--checkpoint-dir',str(checkpoint_dir.resolve()),'--output-dir',str(a.output_dir.resolve()),'--device',a.device,'--modes','zero,mean_clamp']
 if a.quick:command.append('--quick')
 print(subprocess.list2cmdline(command))
 if a.execute:subprocess.run(command,check=True)
 else:print('Preview only. No evaluation, fitting, or training was executed.')
if __name__=='__main__':main()
