from pathlib import Path
import ctypes, json, hashlib, re
import xml.etree.ElementTree as ET
import pypdfium2 as pdfium
import pypdfium2.raw as raw

SRC = Path('C:/Users/LENOVO/Desktop/New project')
OUT = Path('outputs/figure5_clean')
OUT.mkdir(parents=True, exist_ok=True)
titles = {
'Two perturbations of a frozen memory system',
'Hippocampal population bifurcation portrait',
'Seed-resolved behavioral responses',
'History access across perturbations',
'Within-seed access–behavior coupling',
'Readout effects across perturbations'}
notes = {
'Distinct inputs converge','Structure survives perturbation',
'One factor varied at a time · cue cartoons are illustrative',
'chance / collapsed','basin','accessible-memory basin','interpretive regions;',
'not fitted potentials','shared memory','state','alias branch →','chance attractor',
'noise branch →','stable memory manifold',
'Packets: 10 seed marks; center: mean; envelope: relative 95% mean uncertainty.',
'Filaments: observed seed trajectories; trunk width: split-half stability. Main axes are zoomed.',
'Same seed order across models and perturbations','Across-seed mean and 95% CI',
'Dashed line: chance (0.50)','0.50 = chance',
'Open circles: HDGM seeds   Squares: TEM seeds',
'Diamonds and intervals: HDGM mean and 95% bootstrap CI',
'70 seed x condition observations; duplicated baseline removed',
'vertical links organize readouts; they do not assert mediation'}
norm = lambda s: ' '.join(s.split())
remove = {norm(s) for s in titles | notes}
d = pdfium.PdfDocument(SRC/'figure5_integrated_BC_v2.pdf')
p = d[0]
tp = p.get_textpage()
objects = list(p.get_objects())
removed=[]
for obj in objects:
    if obj.type != raw.FPDF_PAGEOBJ_TEXT: continue
    n = raw.FPDFTextObj_GetText(obj.raw, tp.raw, None, 0)
    buf = (ctypes.c_ushort * ((n+1)//2))()
    raw.FPDFTextObj_GetText(obj.raw, tp.raw, buf, n)
    s = bytes(buf).decode('utf-16-le').rstrip('\x00')
    x0,y0,x1,y1 = obj.get_bounds()
    # Two mathtext explanations in A and the mathtext production note in F.
    math_note = (x1 < 447 and 790 < y0 < 803) or (x0 > 700 and 10 < y0 < 25)
    if norm(s) in remove or math_note:
        removed.append({'text':s,'bounds':[x0,y0,x1,y1]})
        p.remove_obj(obj)
tp.close()
p.gen_content()
target=OUT/'figure5_no_titles.pdf'
d.save(target)
check=pdfium.PdfDocument(target)
page=check[0]
text=page.get_textpage().get_text_range()
assert all(t not in text for t in titles)
page.render(scale=1.5).to_pil().save(OUT/'figure5_no_titles_preview.png')
page.render(scale=3).to_pil().save(OUT/'figure5_no_titles.png')
page.render(scale=1).to_pil().convert('L').save(OUT/'qa_gray.png')
assert sum(o.type != raw.FPDF_PAGEOBJ_TEXT for o in page.get_objects()) == sum(o.type != raw.FPDF_PAGEOBJ_TEXT for o in objects)

# Preserve a text-editable vector counterpart with the same removals.
ET.register_namespace('', 'http://www.w3.org/2000/svg')
tree=ET.parse(SRC/'figure5_integrated_BC_v2.svg')
root=tree.getroot()
for parent in list(root.iter()):
    for node in list(parent):
        if node.tag.endswith('}text'):
            s=norm(''.join(node.itertext()))
            if s in remove or 'erases discriminative structure' in s or 'adds stochastic variation' in s or s.startswith('node = paired'):
                parent.remove(node)
tree.write(OUT/'figure5_no_titles.svg',encoding='utf-8',xml_declaration=True)

caption=(SRC/'figure5_integrated_BC_caption.md').read_text(encoding='utf-8')
caption=caption.split('## Production notes')[0]
caption=caption.replace('The accepted population-bifurcation portrait, plotted in history', 'Population responses plotted in history')
caption=caption.replace('Basins, attractor and manifold labels are interpretive,', 'The shaded regions and branch geometry are interpretive,')
caption += '\n## Explanatory text transferred from the figure\n\n'
caption += '**A.** Distinct inputs converge under structural aliasing; increasing alpha erases discriminative structure. Observation noise adds stochastic variation while preserving the underlying cue structure. One factor is varied at a time; cue cartoons are illustrative.\n\n'
caption += '**B.** The original accessible-memory basin, collapsed basin, attractor and manifold terminology described conceptual interpretations only. These labels have been removed from the plot; no formal dynamical bifurcation or fitted potential is established. The shared baseline is alias alpha = 0 and noise sigma = .05.\n\n'
caption += '**C–F.** Panel titles and descriptive footnotes have been removed; their sample, interval, symbol and interpretation definitions are given in the corresponding panel descriptions above. All numerical annotations, plotted observations, scales and uncertainty graphics are retained.\n'
(OUT/'figure5_caption.md').write_text(caption,encoding='utf-8')
(OUT/'edit_manifest.json').write_text(json.dumps({'source':str(SRC/'figure5_integrated_BC_v2.pdf'),'source_sha256':hashlib.sha256((SRC/'figure5_integrated_BC_v2.pdf').read_bytes()).hexdigest(),'removed_text_objects':removed,'non_text_object_count_unchanged':True,'scope':'Text-only cleanup; retain original geometry, data, statistical annotations, panel letters and dimensions.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(target.resolve())
print('Removed text objects:',len(removed))
for obj in objects:
    obj.close()
p.close()
d.close()
page.close()
check.close()
