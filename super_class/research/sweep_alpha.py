#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep super_alpha (the spread-penalty coefficient in ImgtoSuperClass_Metric)
across a range of values, reusing one loaded model/backbone, and report
constituent vs. super-class accuracy at each value -- for both the original
curated triplets and the newly discovered ones.

super_scores = super_raw - super_alpha * spread(constituent_scores)
Lower alpha = less penalty = super column easier to win (higher super
recall, but more risk of constituent queries being misclassified as super).

Usage:
  python research/sweep_alpha.py --places_dir /data/places365 \
      --csv_path ./dataset/Places365/discovered_supers.csv \
      --alphas 0.0 0.25 0.5 0.75 1.0 \
      --out_json ./results/alpha_sweep_discovered.json
"""
from __future__ import print_function
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

# Repo root is one directory up from research/ -- add it so
# `import Test_Super_Places` / `import models.network` resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.backends.cudnn as cudnn
from PIL import ImageFile

import Test_Super_Places as tsp
import models.network as FewShotNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--places_dir', required=True)
    p.add_argument('--csv_path', required=True)
    p.add_argument('--out_json', required=True)
    p.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--encoder_model', default='DINOv2_Local')
    p.add_argument('--imageSize', type=int, default=224)
    p.add_argument('--way_num', type=int, default=5)
    p.add_argument('--shot_num', type=int, default=1)
    p.add_argument('--query_num', type=int, default=10)
    p.add_argument('--neighbor_k', type=int, default=3)
    p.add_argument('--episode_test_num', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    p.add_argument('--cpu', action='store_true', default=False)
    return p


def main(argv=None):
    opt = build_parser().parse_args(argv)
    opt.cuda = bool(opt.cuda and not opt.cpu and torch.cuda.is_available())
    device = torch.device('cuda' if opt.cuda else 'cpu')
    cudnn.benchmark = opt.cuda

    places_dir = os.path.expanduser(opt.places_dir)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model = FewShotNet.define_model(
        encoder_model=opt.encoder_model, classifier_model='DN4_SuperClass',
        norm='batch', way_num=opt.way_num, shot_num=opt.shot_num,
        neighbor_k=opt.neighbor_k, init_type='normal', use_gpu=opt.cuda,
    )
    model.classifier.base_per_super = 2
    model.eval()

    triplets = tsp.load_triplets(opt.csv_path)
    all_classes = tsp.discover_classes(places_dir)
    transform = tsp.build_transform(opt.imageSize)

    # Fix the distractor draws ONCE (independent of alpha) so every alpha in
    # the sweep is scored on the exact same episodes -- alpha only rescales
    # the classifier's decision boundary, not the sampled data.
    sweep = {'meta': vars(opt).copy(), 'alphas': {}}
    sweep['meta']['device'] = str(device)

    for alpha in opt.alphas:
        model.classifier.super_alpha = alpha
        t0 = time.time()
        grand = {'base1': [0, 0], 'base2': [0, 0], 'super': [0, 0]}
        per_triplet = []
        for base1, base2, super_cls in triplets:
            distractor_pool = [c for c in all_classes if c not in {base1, base2, super_cls}]
            result = tsp.run_triplet(model, (base1, base2, super_cls), places_dir,
                                      distractor_pool, transform, opt, device, open(os.devnull, 'w'))
            super_col = opt.way_num
            row = {'base1': base1, 'base2': base2, 'super': super_cls}
            for gt_col, key in [(0, 'base1'), (1, 'base2'), (super_col, 'super')]:
                h, n, _ = result.get(gt_col, (0, 0, None))
                row[f'{key}_acc'] = 100.0 * h / n if n > 0 else None
                grand[key][0] += h
                grand[key][1] += n
            per_triplet.append(row)

        agg = {k: {'hits': h, 'total': n, 'acc': 100.0 * h / n if n > 0 else None}
               for k, (h, n) in grand.items()}
        c_hits = grand['base1'][0] + grand['base2'][0]
        c_total = grand['base1'][1] + grand['base2'][1]
        agg['constituent_combined'] = {'hits': c_hits, 'total': c_total,
                                        'acc': 100.0 * c_hits / c_total if c_total > 0 else None}
        elapsed = time.time() - t0
        print(f'alpha={alpha:.2f}  base1={agg["base1"]["acc"]:.1f}%  base2={agg["base2"]["acc"]:.1f}%  '
              f'super={agg["super"]["acc"]:.1f}%  constituent_combined={agg["constituent_combined"]["acc"]:.1f}%  '
              f'({elapsed:.0f}s)', flush=True)
        sweep['alphas'][str(alpha)] = {'aggregate': agg, 'per_triplet': per_triplet, 'elapsed_sec': elapsed}

    with open(opt.out_json, 'w') as f:
        json.dump(sweep, f, indent=2)
    print(f'Wrote {opt.out_json}')


if __name__ == '__main__':
    main()
