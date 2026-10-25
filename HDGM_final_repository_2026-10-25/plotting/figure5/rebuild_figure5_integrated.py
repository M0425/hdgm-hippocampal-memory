"""Rebuild the full Figure 5 using original A/D/E/F and approved B/C.

No raster panels are pasted: every panel is rendered into a Matplotlib
SubFigure, retaining vector paths and editable SVG text. Source main()
bodies are adapted in memory only; original scripts are never overwritten.
"""
from pathlib import Path
import ast
import csv
import hashlib
import importlib.util
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path(r'C:\Users\马明哲\Documents\Downloads\rebuild_figure5_capacity_boundary.py')
OUT = ROOT / 'figure5_integrated_BC_v2'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_existing(module, subfigure):
    """Use the original plotting statements, but skip standalone exports."""
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8-sig'))
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    body = []
    for stmt in original.body:
        source = ast.unparse(stmt)
        if source.startswith('fig.savefig(') or source.startswith('fig.canvas.draw('):
            break
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'fig' for t in stmt.targets):
            continue
        body.append(stmt)
    body.append(ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[])))
    fn = ast.FunctionDef(name='_draw_integrated', args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg='fig')], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body, decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = dict(module.__dict__)
    exec(compile(code, str(module.__file__), 'exec'), scope)
    return scope['_draw_integrated'](subfigure)


def header(sub, letter, title):
    sub.text(.025, .965, letter, fontsize=18, weight='bold', va='top', color='#1E293B')
    sub.text(.105, .958, title, fontsize=11, weight='bold', va='top', color='#1E293B')


def main():
    old = load_module('fig5_original', ORIGINAL)
    b = load_module('fig5_b', ROOT/'rebuild_figure5_panel_b_bifurcation.py')
    c = load_module('fig5_c', ROOT/'figure5c_response_atlas.py')
    old.DATA_JSON = b.DATA
    old.OUT_CSV = ROOT/'figure5_integrated_BC_source_data.csv'
    records, summary = old.load_data()
    assert b.DATA.resolve() == c.DATA.resolve()
    source_hash = hashlib.sha256(b.DATA.read_bytes()).hexdigest()
    old.export_seed_csv(records)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.linewidth':.8,
        'figure.facecolor':'white', 'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(17, 14.5), facecolor='white')
    gs = fig.add_gridspec(3, 24, left=.008, right=.992, top=.992, bottom=.012,
        height_ratios=[1.03, 1.05, .78], hspace=.12, wspace=.1)
    panels = {
        'A':fig.add_subfigure(gs[0,:9]), 'B':fig.add_subfigure(gs[0,9:]),
        'C':fig.add_subfigure(gs[1,:12]), 'D':fig.add_subfigure(gs[1,12:]),
        'E':fig.add_subfigure(gs[2,:12]), 'F':fig.add_subfigure(gs[2,12:])}
    a = panels['A'].add_axes([.045,.09,.93,.78])
    from figure5_panel_a_cartoon import draw as draw_a_cartoon
    draw_a_cartoon(a,old)
    header(panels['A'], 'A', 'Two perturbations of a frozen memory system')

    bs = render_existing(b, panels['B'])
    # Redistribute physical space only: unchanged limits and empirical marks.
    bs['ax'].set_position([.085,.180,.755,.650])
    bs['ax_bottom'].set_position([.085,.098,.755,.040])
    bs['ax_right'].set_position([.862,.180,.045,.650])
    # Unified external header; preserve all data-coordinate marks and limits.
    for t in bs['ax'].texts:
        if t.get_text() in ['B','Hippocampal population bifurcation portrait']:
            t.set_visible(False)
        if t.get_text() == 'noise branch →\nstable memory manifold':
            t.set_position((.700,.594))
    header(panels['B'], 'B', 'Hippocampal population bifurcation portrait')
    for t in panels['B'].texts:
        if t.get_text() == 'History decoding (frozen transfer probe)':
            t.set_position((.4625,.063))
        if t.get_text().startswith('Packets:'):
            t.set_text('Packets: 10 seed marks; center: mean; envelope: relative 95% mean uncertainty.\n'
                       'Filaments: observed seed trajectories; trunk width: split-half stability. Main axes are zoomed.')
            t.set_position((.085,.006))
            t.set_va('bottom')
    cs = render_existing(c, panels['C'])
    for t in panels['C'].texts:
        if t.get_text() == 'C': t.set_fontsize(18)
        if t.get_text() == 'Seed-resolved behavioral responses': t.set_fontsize(11)
    # Exact mean and interval equality with the accepted standalone C.
    with (ROOT/'figure5C_response_atlas_zoom_summary.csv').open(encoding='utf-8') as f:
        reference = list(csv.DictReader(f))
    for actual, expected in zip(cs['summaries'], reference):
        assert actual['model']==expected['model'] and actual['condition']==expected['condition']
        for field in ['mean','ci95_low','ci95_high']:
            assert actual[field] == float(expected[field])
    assert len(cs['summaries'])==len(reference)==24

    d = panels['D'].add_axes([.08,.13,.89,.70])
    old.draw_panel_d(d, records, summary)
    header(panels['D'], 'D', 'History access across perturbations')
    for child, label, color in zip(d.child_axes,
            ['Alias corruption', 'Observation noise'], [old.ALIAS_DARK,old.NOISE_DARK]):
        child.set_title(label,loc='left',fontsize=9,color=color,pad=26,weight='bold')
    panels['D'].text(.10,.09,'Open circles: HDGM seeds   Squares: TEM seeds\n'
        'Diamonds and intervals: HDGM mean and 95% bootstrap CI',fontsize=8,color=old.SLATE)

    e = panels['E'].add_axes([.105,.20,.85,.65])
    r, rci = old.draw_panel_e(e, records)
    header(panels['E'], 'E', 'Within-seed access–behavior coupling')
    f = panels['F'].add_axes([.025,.04,.95,.82])
    dz, q = old.draw_panel_f(f, records)
    header(panels['F'], 'F', 'Readout effects across perturbations')
    f.text(1.5,2.69,'ALIAS CORRUPTION',ha='center',fontsize=8,color=old.ALIAS_DARK,weight='bold')
    f.text(6.5,2.69,'OBSERVATION NOISE',ha='center',fontsize=8,color=old.NOISE_DARK,weight='bold')

    # Enforce one readable physical font floor without changing data geometry.
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text(): t.set_fontsize(max(8, t.get_fontsize()))
    fig.canvas.draw()
    for ext in ['png','svg','pdf']:
        fig.savefig(str(OUT)+'.'+ext,dpi=360,facecolor='white')
    fig.savefig(str(OUT)+'_preview.png',dpi=150,facecolor='white')
    for letter in ['A','B','C']:
        extent=panels[letter].bbox.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(str(OUT)+'_panel_'+letter+'.png',dpi=170,bbox_inches=extent,facecolor='white')
    # Grayscale diagnostic derived from the plot renderer, not a replacement asset.
    from PIL import Image
    Image.open(str(OUT)+'_preview.png').convert('L').save(str(OUT)+'_qa_gray.png')
    meta = {
        'data_source':str(b.DATA), 'data_sha256':source_hash,
        'scripts':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [ORIGINAL,Path(b.__file__),Path(c.__file__),Path(__file__),ROOT/'figure5_panel_a_cartoon.py']},
        'dimensions_mm':[431.8,368.3], 'minimum_font_pt':8,
        'panels':'A: enlarged conceptual vector cue cartoons; D/E/F: original calculations; B: unchanged data with tighter right margin; C: accepted zoomed atlas',
        'C_exact_comparison':'24 means and 48 CI endpoints identical to accepted C',
        'C_CI':'Student t df=9, unadjusted 95%; TEM/stateless limits 0.47–0.56',
        'D_CI':'16000 seed bootstrap resamples, percentile 95%',
        'E_CI':'12000 seed-cluster bootstrap resamples, percentile 95%',
        'coupling_r':r,'coupling_r_ci':list(rci),
        'F_dz':dz.tolist(),'F_p_holm':q.tolist(),
        'reference':'User supplied original layout and approved B/C, superseding default style examples'}
    Path(str(OUT)+'_provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved', OUT)
    print('Coupling:',r,rci,'; F significant cells:',int((q<.05).sum()))
    print('C source/summary integrity checks: PASS')


if __name__=='__main__':main()
