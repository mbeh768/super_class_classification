#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run every (base1, base2, super) triplet in the Places365 CSV through the
same zero-shot super-class evaluation as Test_Super_Places.py, and dump a
single JSON file with per-triplet accuracies + confusion columns instead of
(only) a text log. Meant to be run once against an abridged local Places365
copy (e.g. 10 images/class) and the resulting JSON turned into a PDF report
elsewhere.

Usage:
  python research/report_superclass.py \
      --places_dir ~/places365-dataset-abridged \
      --csv_path ./dataset/Places365/potential_supers.csv \
      --out_json ./results/places_superclass_report.json
"""

from __future__ import print_function
import argparse
import json
import os
import random
import sys
import time
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
import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--places_dir', required=True,
                    help='root of extracted Places365 (per-class folders)')
    p.add_argument('--csv_path', default='./dataset/Places365/potential_supers.csv',
                    help='CSV with header: base1,base2,super')
    p.add_argument('--out_json', default='./results/places_superclass_report.json')
    p.add_argument('--encoder_model', default='DINOv2_Local')
    p.add_argument('--classifier_model', default='DN4_SuperClass')
    p.add_argument('--imageSize', type=int, default=224)
    p.add_argument('--way_num', type=int, default=5,
                    help='5 = 2 constituents + 3 distractors in support; '
                         'the super class is queried but never in support, '
                         'giving 6 columns total (way_num+1)')
    p.add_argument('--shot_num', type=int, default=1)
    p.add_argument('--query_num', type=int, default=5,
                    help='queries per type (base1 / base2 / super); kept '
                         'small since the abridged set has 10 imgs/class')
    p.add_argument('--neighbor_k', type=int, default=3)
    p.add_argument('--super_alpha', type=float, default=1.0)
    p.add_argument('--episode_test_num', type=int, default=30,
                    help='episodes per triplet; distractor sampling varies '
                         'per episode so this also averages over distractors')
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
    if not os.path.isdir(places_dir):
        raise FileNotFoundError(f'Places365 directory not found: {places_dir}')
    if not os.path.isfile(opt.csv_path):
        raise FileNotFoundError(f'CSV file not found: {opt.csv_path}')

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    os.makedirs(os.path.dirname(opt.out_json) or '.', exist_ok=True)
    F_txt = open(opt.out_json + '.log', 'a+')

    print('==================== Super-Class Places Report ====================')
    print(opt)
    print(opt, file=F_txt)

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

    triplets = tsp.load_triplets(opt.csv_path)
    if not triplets:
        raise RuntimeError(f'No triplets found in {opt.csv_path}')
    print(f'{len(triplets)} triplet(s) to evaluate from {opt.csv_path}')

    all_classes = tsp.discover_classes(places_dir)
    print(f'{len(all_classes)} class folders found under {places_dir}: {all_classes}')
    transform = tsp.build_transform(opt.imageSize)

    report = {
        'meta': {
            'places_dir': places_dir,
            'csv_path': opt.csv_path,
            'encoder_model': opt.encoder_model,
            'classifier_model': opt.classifier_model,
            'way_num': opt.way_num,
            'shot_num': opt.shot_num,
            'query_num': opt.query_num,
            'neighbor_k': opt.neighbor_k,
            'super_alpha': opt.super_alpha,
            'episode_test_num': opt.episode_test_num,
            'seed': opt.seed,
            'n_classes_available': len(all_classes),
            'classes_available': all_classes,
            'device': str(device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'triplets': [],
    }

    super_col = opt.way_num
    for t_idx, (base1, base2, super_cls) in enumerate(triplets):
        distractor_pool = [c for c in all_classes if c not in {base1, base2, super_cls}]
        missing = [c for c in (base1, base2, super_cls)
                   if not (Path(places_dir) / c).is_dir()]
        header = f'\n[{t_idx + 1}/{len(triplets)}] base1={base1} base2={base2} super={super_cls}'
        print(header)
        print(header, file=F_txt)

        if missing:
            msg = f'  ! skipping: missing folders {missing}'
            print(msg)
            print(msg, file=F_txt)
            report['triplets'].append({
                'base1_name': base1, 'base2_name': base2, 'super_name': super_cls,
                'skipped': True, 'reason': f'missing folders: {missing}',
            })
            continue
        if len(distractor_pool) < (opt.way_num - 2):
            msg = f'  ! skipping: not enough distractors ({len(distractor_pool)} available)'
            print(msg)
            print(msg, file=F_txt)
            report['triplets'].append({
                'base1_name': base1, 'base2_name': base2, 'super_name': super_cls,
                'skipped': True, 'reason': 'not enough distractors',
            })
            continue

        t0 = time.time()
        result = tsp.run_triplet(model, (base1, base2, super_cls), places_dir,
                                  distractor_pool, transform, opt, device, F_txt)
        elapsed = time.time() - t0
        if result is None:
            report['triplets'].append({
                'base1_name': base1, 'base2_name': base2, 'super_name': super_cls,
                'skipped': True, 'reason': 'run_triplet returned None (image count issue)',
            })
            continue

        entry = {
            'base1_name': base1, 'base2_name': base2, 'super_name': super_cls,
            'skipped': False, 'elapsed_sec': elapsed,
            'columns': [base1, base2] + [None] * (opt.way_num - 2) + [f'super({super_cls})'],
        }
        for gt_col, key, label in [(0, 'base1', base1), (1, 'base2', base2),
                                    (super_col, 'super', super_cls)]:
            h, n, dist = result.get(gt_col, (0, 0, [0] * (opt.way_num + 1)))
            acc = 100.0 * h / n if n > 0 else None
            entry[key] = {'label': label, 'hits': h, 'total': n, 'acc': acc,
                           'argmax_dist': dist}
            line = '    {0:<25s} n={1:4d} acc={2}  dist={3}'.format(
                label, n, ('%.2f%%' % acc) if acc is not None else 'n/a', dist)
            print(line)
            print(line, file=F_txt)
        report['triplets'].append(entry)

    # Micro-averaged aggregates across all evaluated (non-skipped) triplets
    grand = {'base1': [0, 0], 'base2': [0, 0], 'super': [0, 0]}  # [hits, total]
    for t in report['triplets']:
        if t.get('skipped'):
            continue
        for key in ('base1', 'base2', 'super'):
            grand[key][0] += t[key]['hits']
            grand[key][1] += t[key]['total']
    report['aggregate'] = {
        key: {'hits': h, 'total': n, 'acc': (100.0 * h / n if n > 0 else None)}
        for key, (h, n) in grand.items()
    }
    # constituents combined (base1+base2 together) vs super, since that's the
    # headline "constituent accuracy vs super-class accuracy" comparison
    c_hits = grand['base1'][0] + grand['base2'][0]
    c_total = grand['base1'][1] + grand['base2'][1]
    report['aggregate']['constituent_combined'] = {
        'hits': c_hits, 'total': c_total,
        'acc': (100.0 * c_hits / c_total if c_total > 0 else None),
    }

    with open(opt.out_json, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\nWrote report JSON to {opt.out_json}')
    print(f'Wrote report JSON to {opt.out_json}', file=F_txt)
    F_txt.close()


if __name__ == '__main__':
    main()
