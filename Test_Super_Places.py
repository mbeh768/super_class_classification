#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot super-class evaluation on Places365.

Reads (base1, base2, super) triplets from a CSV and, for each triplet,
runs episodic evaluation where:

  Support : base1 (col 0) + base2 (col 1) + (way_num - 2) random distractor
            classes from Places365.  The SUPER CLASS IS NEVER IN SUPPORT.
  Queries : `query_num` images each from base1, base2, and super, with
            ground-truth columns 0, 1, and way_num respectively.

The model uses `DN4_SuperClass` (ImgtoSuperClass_Metric) which adds an
extra column at index way_num scored by pooling the descriptor banks of
the first `base_per_super` (=2) support classes.

Reports per-triplet accuracy on each query type and an aggregate summary.

Example:
    python Test_Super_Places.py \
        --places_dir /data/places365 \
        --csv_path  ./dataset/Places365/potential_supers.csv \
        --encoder_model DINOv2_Local --imageSize 224 \
        --super_alpha 0.0 \
        --resume ./results/DINOv2_Places_Super/
"""

from __future__ import print_function
import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image, ImageFile

sys.dont_write_bytecode = True
import models.network as FewShotNet
import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--places_dir', default=os.environ.get('PLACES365_DIR', './dataset/Places365'),
        help='root of extracted Places365 (per-class folders)')
    parser.add_argument('--csv_path', default=os.environ.get('PLACES365_CSV', './dataset/Places365/potential_supers.csv'),
        help='CSV with header: base1,base2,super')
    parser.add_argument('--triplet_filter', default=None,
        help='only run triplets whose super class matches this name')
    parser.add_argument('--resume', default=os.environ.get('PLACES365_OUT', './results/DINOv2_Places_Super'),
        help='output directory for the results log (created if needed)')
    parser.add_argument('--encoder_model', default='DINOv2_Local')
    parser.add_argument('--classifier_model', default='DN4_SuperClass')
    parser.add_argument('--imageSize', type=int, default=224)
    parser.add_argument('--way_num', type=int, default=5)
    parser.add_argument('--shot_num', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=15,
        help='queries per type (base1 / base2 / super)')
    parser.add_argument('--neighbor_k', type=int, default=3)
    parser.add_argument('--super_alpha', type=float, default=1.0,
        help='spread-penalty coefficient (0=no penalty, 1=full)')
    parser.add_argument('--episode_test_num', type=int, default=50,
        help='number of episodes per triplet')
    parser.add_argument('--max_triplets', type=int, default=None,
        help='optional limit for quick demo runs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', default=False,
        help='force CPU even when CUDA is available')
    return parser


# ============================ Data utilities ============================ #

def build_transform(image_size):
    """Match the (0.5, 0.5, 0.5) normalisation used by the project's existing
    dataloaders. DINOv2_Local internally re-normalises to ImageNet stats."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def list_class_images(class_dir):
    """Return sorted .jpg paths for one class folder."""
    return sorted(p for p in class_dir.iterdir()
                  if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})


def load_triplets(csv_path, triplet_filter=None):
    """Yield (base1, base2, super) tuples from the CSV, skipping header
    and any rows whose first cell starts with '#'."""
    triplets = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            base1, base2, super_cls = (c.strip() for c in row[:3])
            # skip header and comments
            if base1.lower() in {'base1'} or base1.startswith('#'):
                continue
            if triplet_filter and super_cls != triplet_filter:
                continue
            triplets.append((base1, base2, super_cls))
    return triplets


def discover_classes(places_dir):
    """All immediate subdirectories of places_dir that look like class folders."""
    return sorted(d.name for d in Path(places_dir).iterdir()
                  if d.is_dir() and not d.name.startswith('_'))


# ============================ Episode runner ============================ #

def run_triplet(model, triplet, places_dir, distractor_pool, transform, opt, device, F_txt):
    """Run opt.episode_test_num episodes for a single triplet.
    Returns dict mapping target_col -> (hits, total, dist_counts)."""
    base1, base2, super_cls = triplet
    super_col = opt.way_num

    # Pre-collect image paths for the three target classes
    paths_by_class = {}
    for cname in (base1, base2, super_cls):
        cdir = Path(places_dir) / cname
        if not cdir.is_dir():
            print(f'  ! class folder missing: {cdir}', file=F_txt)
            return None
        paths_by_class[cname] = list_class_images(cdir)
        if len(paths_by_class[cname]) < opt.shot_num + opt.query_num:
            print(f'  ! {cname}: only {len(paths_by_class[cname])} images, '
                  f'need {opt.shot_num + opt.query_num}', file=F_txt)

    rng = random.Random((opt.seed, base1, base2, super_cls).__hash__())

    # Per-target accumulators
    hits = defaultdict(int)
    totals = defaultdict(int)
    col_counts = defaultdict(lambda: [0] * (opt.way_num + 1))

    n_distractors = opt.way_num - 2
    if n_distractors < 0:
        raise ValueError('way_num must be >= 2')

    for ep in range(opt.episode_test_num):
        # 1. Pick distractor classes for this episode
        distractors = rng.sample(distractor_pool, n_distractors)
        episode_classes = [base1, base2] + distractors   # cols 0..way_num-1

        # 2. Build support set (shot_num per class)
        support_imgs = []
        used_paths = set()
        ok = True
        for cname in episode_classes:
            cdir = Path(places_dir) / cname
            paths = list_class_images(cdir)
            if len(paths) < opt.shot_num:
                ok = False
                break
            sampled = rng.sample(paths, opt.shot_num)
            for p in sampled:
                used_paths.add(p)
                support_imgs.append(transform(Image.open(p).convert('RGB')))
        if not ok:
            continue
        support_tensor = torch.stack(support_imgs).to(device)

        # 3. Build queries: query_num each from base1 (col 0), base2 (col 1),
        #    super (col way_num).  Super is NEVER in support_paths above.
        query_imgs = []
        query_targets = []
        for col, cname in [(0, base1), (1, base2), (super_col, super_cls)]:
            avail = [p for p in paths_by_class[cname] if p not in used_paths]
            n = min(opt.query_num, len(avail))
            sampled = rng.sample(avail, n)
            for p in sampled:
                query_imgs.append(transform(Image.open(p).convert('RGB')))
                query_targets.append(col)
        query_tensor = torch.stack(query_imgs).to(device)

        # 4. Forward
        with torch.no_grad():
            out = model(query_tensor, support_tensor)     # [Q, way_num+1]
        pred = out.argmax(dim=1).cpu().tolist()

        for q_idx, gt in enumerate(query_targets):
            p = pred[q_idx]
            totals[gt] += 1
            if p == gt:
                hits[gt] += 1
            col_counts[gt][p] += 1

        if ep == 0 or (ep + 1) % 10 == 0:
            msg = ('    ep {0:3d}/{1}  '
                   '{2}={3:5.1f}%  {4}={5:5.1f}%  super={6:5.1f}%'.format(
                       ep + 1, opt.episode_test_num,
                       base1, 100.0 * hits[0] / max(totals[0], 1),
                       base2, 100.0 * hits[1] / max(totals[1], 1),
                       100.0 * hits[super_col] / max(totals[super_col], 1)))
            print(msg)
            print(msg, file=F_txt)

    return {gt: (hits[gt], totals[gt], col_counts[gt])
            for gt in totals}


# ============================ Main ============================ #

def main(argv=None):
    parser = build_parser()
    opt = parser.parse_args(argv)
    opt.cuda = bool(opt.cuda and not opt.cpu and torch.cuda.is_available())
    device = torch.device('cuda' if opt.cuda else 'cpu')
    cudnn.benchmark = opt.cuda

    if not os.path.isdir(opt.places_dir):
        raise FileNotFoundError(f'Places365 directory not found: {opt.places_dir}')
    if not os.path.isfile(opt.csv_path):
        raise FileNotFoundError(f'CSV file not found: {opt.csv_path}')

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # Output dir + log file
    opt.outf = opt.resume if str(opt.resume).endswith('/') else opt.resume + '/'
    os.makedirs(opt.outf, exist_ok=True)
    F_txt = open(os.path.join(opt.outf, 'Test_Super_Places_results.txt'), 'a+')
    print('==================== Super-Class Places Test ====================')
    print('==================== Super-Class Places Test ====================', file=F_txt)
    print(opt)
    print(opt, file=F_txt)

    # Build model (DINOv2 loads its own pretrained weights)
    model = FewShotNet.define_model(
        encoder_model=opt.encoder_model,
        classifier_model=opt.classifier_model,
        norm='batch',
        way_num=opt.way_num,
        shot_num=opt.shot_num,
        neighbor_k=opt.neighbor_k,
        init_type='normal',
        use_gpu=opt.cuda,
    )
    model.classifier.base_per_super = 2
    model.classifier.super_alpha = opt.super_alpha
    model.eval()

    # Optional: load a checkpoint if one happens to be in --resume.
    best_path = os.path.join(opt.resume, 'model_best.pth.tar')
    if os.path.isfile(best_path):
        ckpt = utils.get_resume_file(best_path, F_txt)
        if ckpt is not None:
            missing, unexpected = model.load_state_dict(
                {k: v.to(device) for k, v in ckpt['model'].items()}, strict=False)
            print('loaded checkpoint:', best_path)
            print('  missing   :', missing, file=F_txt)
            print('  unexpected:', unexpected, file=F_txt)
    else:
        print('no checkpoint at {} (expected for DINOv2 zero-shot run)'.format(best_path))

    # Load triplets and discover the distractor pool
    triplets = load_triplets(opt.csv_path, opt.triplet_filter)
    if not triplets:
        raise RuntimeError(f'No triplets found in {opt.csv_path}')
    print(f'\n{len(triplets)} triplet(s) to evaluate from {opt.csv_path}')
    print(f'{len(triplets)} triplet(s) to evaluate', file=F_txt)

    all_classes = discover_classes(opt.places_dir)
    transform = build_transform(opt.imageSize)

    # Aggregate results across triplets
    summary_rows = []
    grand = defaultdict(lambda: {'hits': 0, 'total': 0})

    for t_idx, triplet in enumerate(triplets):
        if opt.max_triplets is not None and t_idx >= opt.max_triplets:
            break
        base1, base2, super_cls = triplet
        # Distractor pool: all Places classes minus the three triplet members
        distractor_pool = [c for c in all_classes
                           if c not in {base1, base2, super_cls}]
        if len(distractor_pool) < (opt.way_num - 2):
            print(f'  ! not enough distractors for {triplet}', file=F_txt)
            continue
        # Also skip if any triplet folder is missing
        missing = [c for c in (base1, base2, super_cls)
                   if not (Path(opt.places_dir) / c).is_dir()]
        if missing:
            print(f'  ! skipping {triplet}: missing folders {missing}')
            print(f'  ! skipping {triplet}: missing folders {missing}', file=F_txt)
            continue

        header = '\n[{0}/{1}]  base1={2}  base2={3}  super={4}'.format(
            t_idx + 1, len(triplets), base1, base2, super_cls)
        print(header)
        print(header, file=F_txt)

        t0 = time.time()
        result = run_triplet(model, triplet, opt.places_dir,
                             distractor_pool, transform, opt, device, F_txt)
        if result is None:
            continue

        # Per-triplet summary — keep names and accuracies in distinct keys
        super_col = opt.way_num
        row = {'base1': base1, 'base2': base2, 'super': super_cls,
               'b1_acc': float('nan'), 'b2_acc': float('nan'), 'super_acc': float('nan')}
        triplet_iter = [
            (0,         base1,   'b1_acc',    base1),
            (1,         base2,   'b2_acc',    base2),
            (super_col, 'super', 'super_acc', super_cls),
        ]
        for gt_col, grand_key, row_key, display_label in triplet_iter:
            h, n, dist = result.get(gt_col, (0, 0, [0] * (opt.way_num + 1)))
            acc = 100.0 * h / n if n > 0 else float('nan')
            row[row_key] = acc
            grand[grand_key]['hits'] += h
            grand[grand_key]['total'] += n
            dist_pct = ['{0:.1f}'.format(100.0 * c / max(n, 1)) for c in dist]
            line = '    {0:<25s} n={1:4d} acc={2:6.2f}%  dist=[{3}]'.format(
                display_label, n, acc, ', '.join(dist_pct))
            print(line)
            print(line, file=F_txt)
        row['elapsed'] = time.time() - t0
        summary_rows.append(row)

    # Final aggregate
    print('\n==================== Aggregate ====================')
    print('\n==================== Aggregate ====================', file=F_txt)
    print('per-triplet:')
    print('per-triplet:', file=F_txt)
    for r in summary_rows:
        line = ('  {0:<22s} + {1:<22s} -> {2:<22s}  '
                'b1={3:5.1f}  b2={4:5.1f}  super={5:5.1f}  ({6:.1f}s)').format(
                    r['base1'], r['base2'], r['super'],
                    r['b1_acc'], r['b2_acc'], r['super_acc'], r['elapsed'])
        print(line)
        print(line, file=F_txt)

    print('\noverall (micro-averaged across all queries of each type):')
    print('\noverall (micro-averaged across all queries of each type):', file=F_txt)
    for label, agg in grand.items():
        if agg['total'] == 0:
            continue
        acc = 100.0 * agg['hits'] / agg['total']
        line = '  {0:<25s}  n={1:5d}  acc={2:6.2f}%'.format(label, agg['total'], acc)
        print(line)
        print(line, file=F_txt)

    F_txt.close()


if __name__ == '__main__':
    main()
