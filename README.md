# Zero-Shot Super-Class Discovery on Places365

A PyTorch implementation of DN4 (Image-to-Class few-shot learning, CVPR
2019) extended with a zero-shot **super-class** evaluation path: given two
base classes (e.g. `gazebo-exterior`, `picnic_area`), score whether a query
image belongs to a synthetic super-class (e.g. `park`) built by pooling the
two base classes' local-descriptor banks — without ever training on the
super-class itself.

The six triplets bundled with the demo weren't hand-picked. They were found
by a combinatorial screen over every pair of Places365's 365 categories
(66,430 pairs), looking for cases where some third category's photos get
disproportionately pulled toward the pooled pair — then the classifier's
spread-penalty coefficient was swept and tuned against them. See
"Discovering New Super-Classes" and "Tuning the Spread Penalty" below for
how both were done.

## Requirements

Requires [Miniconda/Anaconda](https://docs.conda.io/en/latest/miniconda.html)
on PATH. Build the environment with the cross-platform setup script (works
the same way on Linux, macOS, and Windows — pure Python + `conda`, no shell
script):

```bash
python setup_env.py
conda activate super_class
```

`--dry-run` prints the commands without running them; `--name`/`--python`
override the environment name (default `super_class`) and Python version
(default `3.10.18`). Re-running it is safe — it reuses the environment if
one by that name already exists rather than recreating it.

Or set it up manually:

```bash
conda create -n super_class python=3.10.18
conda activate super_class
pip install -r requirements.txt
```

## Quick Start

```bash
python demo_places365.py
```

Runs the 6 discovered super-class triplets in
`dataset/Places365/discovered_supers.csv` at a tuned `super_alpha=0.15`
(where super-class recall roughly matches or beats constituent recall). No
external data download needed — the repo ships with a small demo subset
(10 images x 24 classes, ~3.8MB). Each episode's support set is 2
constituent classes + 3 random distractors (the super class is never in
support); queries are drawn separately from each constituent and from the
super class itself. Uses the `DINOv2_Local` backbone (frozen `dinov2_vits14`)
and the `DN4_SuperClass` classifier.

Point at a different triplet CSV / larger data with:

```bash
python Test_Super_Places.py --places_dir <dir> --csv_path <triplets.csv> \
    --encoder_model DINOv2_Local --imageSize 224 --super_alpha 0.15
```

or override paths via environment variables:

```bash
export PLACES365_DIR=/path/to/Places365
export PLACES365_CSV=/path/to/triplets.csv
python demo_places365.py
```

To build a larger subset (more images/class, or your own triplets) from a
full Places365 download:

```bash
python prepare_places365_subset.py \
  --source_dir /path/to/full/Places365 \
  --output_dir ./dataset/Places365 \
  --mode copy
```

`--mode symlink` links instead of copying; `--images_per_class` changes the
sample size. The triplet CSV must have a header of `base1,base2,super`,
each row a triplet of class folder names.

## Discovering New Super-Classes

For every pair of Places365 categories, checks whether some third
category's photos get pulled into the super-class column at an unusually
high rate — the same signal the known triplets already encode (e.g. lake +
cottage pulling toward boathouse). A pair/candidate combination with a high
"attraction rate" is a candidate new super-class relationship.

- `research/screen_curated.py` — screens all C(21,2)=210 pairs within a
  curated Places365 class set.
- `research/screen_allpairs.py` — screens all C(365,2)=66,430 pairs across
  the full Places365 category list (base classes can be any Places365
  category; the candidate query pool is kept to a fixed subset for
  tractability). Supports sharding (`--shard_id`/`--num_shards`) to run
  multiple worker processes in parallel against one GPU.
- `reports/gen_places_screen_pdf.py` / `reports/gen_places_allpairs_pdf.py` /
  `reports/gen_places_allpairs_combined_pdf.py` — turn the JSON/JSONL screening
  output into a PDF report (validation against known triplets, ranked list
  of new discoveries, confidence intervals).

## Tuning the Spread Penalty (`super_alpha`)

`ImgtoSuperClass_Metric`'s super-class score is `super_raw - super_alpha *
spread(constituent_scores)` (see `models/classifier.py`). The class default
(`super_alpha=1.0`, full penalty) favors constituent accuracy over
super-class recall. `research/sweep_alpha.py` sweeps this coefficient (reusing
one loaded model across all values, with fixed episode sampling so every
value is compared on identical data) and `reports/gen_alpha_sweep_pdf.py` plots the
constituent/super-class accuracy tradeoff curve. The crossover point is
dataset-dependent — around 0.10-0.20 across the triplet sets tried so far —
with `super_alpha=0.15` used as the demo default.

## Visualizing the Local Descriptor Feature Space

`research/umap_featurespace.py` extracts the frozen DINOv2 backbone's raw
local (patch-token) descriptors for a triplet's support and query images —
same episode construction as the accuracy eval, just visualizing the
features directly instead of classifying them — and
`reports/gen_umap_featurespace_pdf.py` projects them with UMAP (cosine metric) to
show whether a super-class query's descriptors sit between its two
constituents' clouds in feature space, or somewhere else entirely.
Independent of `super_alpha`, which only affects the classifier's decision,
not the backbone features.

## Project Layout

- `setup_env.py` — cross-platform conda environment builder.
- `demo_places365.py` — the demo entrypoint.
- `Test_Super_Places.py` — zero-shot Places365 super-class evaluator, run on
  any CSV of triplets. Its parser is built inside `main()` so the module can
  be imported (by `demo_places365.py`) without argparse consuming `sys.argv`.
- `research/screen_curated.py` / `research/screen_allpairs.py` — discover
  new candidate super-class triplets (see above).
- `research/sweep_alpha.py` — tunes the spread-penalty coefficient.
- `research/umap_featurespace.py` — visualizes the local descriptor
  feature space.
- `reports/gen_*.py` — turn each `research/` script's JSON/JSONL/NPZ output
  into a PDF report.
- `prepare_places365_subset.py` — builds a Places365 image subset from a
  full download.
- `dataset/Places365/` — the bundled demo image subset + triplet CSV.
- `models/` — `DINOv2_Local` backbone and `ImgtoSuperClass_Metric`
  (`DN4_SuperClass`) classifier head, plus the original DN4/ProtoNet
  scaffolding they build on (unused by anything currently in the repo, kept
  for reference — see `models/network.py`'s `encoder_dict`/`classifier_dict`
  for what's actually wired up).

## Notes

- The code uses CUDA automatically when available; force CPU with `--cpu`.
- The repository no longer assumes a local `.venv` folder.

## Citation

If you use this code, please cite the original DN4 paper:

```bibtex
@inproceedings{DN4_CVPR_2019,
  author       = {Wenbin Li and Lei Wang and Jinglin Xu and Jing Huo and Yang Gao and Jiebo Luo},
  title        = {Revisiting Local Descriptor Based Image-To-Class Measure for Few-Shot Learning},
  booktitle    = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages        = {7260--7268},
  year         = {2019}
}
```
