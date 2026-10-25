"""Regenerate saved-data plots; never train a model. Run from any directory."""
from pathlib import Path
import argparse, os, re, runpy, sys, tempfile
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('figure',choices=['2','3','4','5b','5c','5d','5e','5f','6']);a=p.parse_args()
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'generated/.mplcache'))
import matplotlib.figure
original=matplotlib.figure.Figure.savefig
def save(self,*args,**kw):
    from matplotlib.text import Text
    for text in self.findobj(match=Text):
        value=text.get_text()
        value=re.sub(r'\bTEM\b','PAM',value)
        if value in list('ABCDEF'):value=value.lower()
        text.set_text(value)
    return original(self,*args,**kw)
matplotlib.figure.Figure.savefig=save
if a.figure in ['2','3','4']:
    folder=ROOT/'plotting'/('figure'+a.figure);script=folder/'plot.py'
elif a.figure=='6':
    folder=ROOT/'plotting/figure6';script=folder/'build_figure.py'
elif a.figure in ['5d','5e','5f']:
    folder=ROOT/'plotting/figure5'
    script=folder/('rebuild_figure5_panel_'+a.figure[-1]+'.py')
else:
    folder=ROOT/'plotting/figure5'
    script=folder/('rebuild_figure5_panel_b_bifurcation.py' if a.figure=='5b' else 'figure5c_response_atlas.py')
sys.path.insert(0,str(folder));os.chdir(folder)
sys.argv=[str(script)]
if a.figure=='2':sys.argv+=['--data-root',str(folder/'data'),'--out-dir',str(folder)]
if a.figure in ['5b','5c']:
    data=ROOT/'data/pythonProject32/hdgm_difficulty_v2_1_frozen_outputs_confirmatory/difficulty_v2_1_multiseed_results.json'
    if a.figure=='5c':
        import rebuild_figure5_panel_c_dotmatrix as dependency
        dependency.DATA=data
    code=script.read_text(encoding='utf-8-sig')
    if a.figure=='5b':
        lines=code.splitlines();code='\n'.join('DATA = Path('+repr(str(data))+')' if l.startswith('DATA = Path(') else l for l in lines)
    exec(compile(code,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
else:runpy.run_path(str(script),run_name='__main__')
