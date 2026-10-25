# Figure 1 and Figure 2 revision

## Output and scope

- Figure 1: `figure1_textured_v5_brain_corrected.png`. Mouse head removed from panel A. Brain remains a conceptual functional illustration, not a validated anatomical reconstruction.
- Figure 2: `figure2_hdgm_refined_colour_v2.png` and matching native PDF. Panel A is generated illustration; no empirical chart was passed through a generative image model. B-D and E paired statistics preserve native source vector coordinates and labels. E circular summary and six hold-condition branches use the original delay-transfer CSV. Branch curves are schematic condition connectors, not neural state trajectories.
- Source: `C:/Users/马明哲/Documents/Downloads/figure2_complete (2).pdf`. The compact source PNG has a later decorative E layout. This revision recreates its circular/branch organization from CSV instead of altering screenshot pixels.
- Fixed model palette: HDGM berry #A64D79; TEM blue-green #328B91. Cue/stage accents indigo #6C73AE, ochre #D6A04A, sage #71997F. Richer semantic color follows the user's instruction in place of the visualization skill's default single-accent palette.
- Native heatmap indexed color lookup tables are changed monotonically; all image indices and scientific geometry remain unchanged. The original normalization is preserved. Colour keys show only lower-to-higher ordering because the exact source normalization endpoints were not available for this composite.
- Protocol distinction: frozen-model information ablations and separate observation/timing-only linear shortcut classifiers must not be conflated. Panel A is illustrative; the separate-classifier caveat is placed beneath it.
- Labels in source B-D retained. Transfer accuracy is event-aligned model decision accuracy, not autonomous online animal behavior.
- For final journal submission, set dimensions and typography to the venue specification. This export is a high-resolution review composition, not a guarantee of legibility at arbitrary downscaling.

## Reproducibility and checks

`figure2_restyle_vector.py` preserves all non-colour PDF operators and checks original heatmap index hashes. `figure2_restyle_integrity.json` records source hash, palette, and the CSV-derived means. Visual review includes full-colour and grayscale PDF renders.

## Generation mode

Built-in image generation was used for the Figure 1 local correction and Figure 2 panel A only. Statistical panels use a separate native-PDF/CSV workflow.

### Figure 1 final prompt

Precise local correction to the supplied COMPLETE Figure 1. In the top-left panel A, the anatomical brain has an erroneous mouse HEAD growing out of its lower-left edge (face, black eye, snout, ears, whiskers). REMOVE THE ENTIRE MOUSE HEAD and all external animal features. Replace that local area with white background and a clean natural contour of an isolated smooth rodent-brain sagittal schematic. Keep only the standalone brain/hippocampal/entorhinal schematic, NO animal head, NO eyes, NO snout, NO ears, NO whiskers, NO fur or body. Keep the teal highlighted hippocampal region and blue entorhinal region as conceptual anatomy. Do NOT create a human head or skull. Preserve every other panel A–E, all module drawings, shading, text, equations, paths and page layout exactly. No other edits. This is a head-removal anatomical illustration correction.

### Figure 2 panel A final prompt

Use case: scientific-educational. Create ONLY panel A of a computational neuroscience manuscript, a complete elegant experimental schematic, landscape aspect ratio 1.65:1. White background, generously spaced, exceptionally crisp legible English text. Premium neuroscience editorial illustration with subtle translucent 2.5D layered objects, fine outlines, soft restrained texture, not generic boxes and arrows. No panel letter.
Title top left: "Controlled history evaluation".
Upper half: two horizontal parallel episodic sequences. Label left "History 1" in dusty indigo #6C73AE and "History 2" in amber #D6A04A. Each row shows two small distinct patterned cue tiles and tiny memory beads, then an identical current observation tile: exact same neutral gray split-lane geometric tile in both rows. Both current tiles are vertically aligned and grouped with a fine brace labeled "Same observation". On right row 1 finishes with a plum output choice token labeled "Choice 1"; row 2 with a teal output choice token labeled "Choice 2". Tiny subtitle under these two rows: "Different history, different target".
Lower half: compact structured evaluation protocol, four equally sized illustrated mini-lanes in a 2 by 2 arrangement on left, feeding a common frozen computational network icon on right. EXACT labels: "Full history", "History reset", "Observation-only", "Timing-only". Full history icon: 3 tiny linked beads and memory loop. History reset icon: same beads with interrupted loop and reset arrow. Observation-only icon: one copy of the neutral observation tile. Timing-only icon: a small clock. Each has its own clear tiny arrow towards a shared junction, no tangles. Network is a tastefully rendered translucent layered neural-network block with discrete nodes, not biological tissue, small padlock icon above it, label "Fixed evaluation". Output on far right a simple four-choice readout labelled "Prediction". Footer beneath network: "Frozen model / shortcut controls".
Important scientific constraint: this is a model-evaluation schematic, NOT an animal experiment. No mice, brains, electrodes, surgery, realistic neurons, data charts, numerical empirical results, or extra invented protocol. Four paths are distinct evaluation controls; do not claim shortcut classifiers share neural weights. No glowing neon, dark backgrounds, big shadows, gradients across background, watermark, or long paragraphs. Color richness must be in coherent semantic cues: dusty indigo, amber, plum #A64D79, teal #328B91, sage, with slate ink. Balance vivid domain illustration with spacious rigor. All arrows clear and all text fully separated from shapes.
