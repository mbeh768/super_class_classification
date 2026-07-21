#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combinatorial screening for NEW candidate super-classes across every pair of
Places365 classes on disk (C(21,2) = 210 pairs), instead of the 10 hand-
curated triplets in potential_supers.csv.

There are no real photos for most hypothetical hybrids (e.g. no folder for
"hospital_room + mountain"), so accuracy can't be scored for them the way
Test_Super_Places.py scores the 7 known supers. Instead, for each pair
(base1, base2) we build the standard support set (base1 + base2 + 3 random
distractors, way_num=5, super column at index 5) and, in the SAME forward
pass, query every one of the other classes' real photos (not just base1/
base2/distractors). A candidate class whose images get pulled into the
super column at a high rate -- more than they get pulled into base1/base2/
distractor columns -- is a signal that (base1, base2) "want" that class as
their natural hybrid. This is exactly the relationship the 7 known triplets
in potential_supers.csv already encode, so we validate the method by
checking whether it recovers those 7 pairs' true super class as a top
candidate before trusting it on the other ~200 unlabeled pairs.

Usage:
  python research/screen_curated.py \
      --places_dir /data/places365 \
      --out_json ./results/places_superclass_screen.json
"""

from __future__ import print_function
import argparse
import itertools
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


KNOWN_SUPERS = {
    frozenset(['lake-natural', 'cottage']): 'boathouse',
    frozenset(['lake-natural', 'house']): 'boathouse',
    frozenset(['ocean', 'cottage']): 'boathouse',
    frozenset(['ocean', 'house']): 'boathouse',
    frozenset(['snowfield', 'mountain']): 'mountain_snowy',
    frozenset(['snowfield', 'canyon']): 'crevasse',
    frozenset(['assembly_line', 'car_interior']): 'auto_factory',
    frozenset(['mountain', 'forest_path']): 'mountain_path',
    frozenset(['building_facade', 'forest_road']): 'street',
    frozenset(['hospital_room', 'clean_room']): 'operating_room',
}
SUPER_CLASS_NAMES = set(KNOWN_SUPERS.values())


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--places_dir', required=True)
    p.add_argument('--out_json', default='./results/places_superclass_screen.json')
    p.add_argument('--encoder_model', default='DINOv2_Local')
    p.add_argument('--classifier_model', default='DN4_SuperClass')
    p.add_argument('--imageSize', type=int, default=224)
    p.add_argument('--way_num', type=int, default=5)
    p.add_argument('--shot_num', type=int, default=1)
    p.add_argument('--query_num', type=int, default=3,
                    help='queries per candidate class per episode')
    p.add_argument('--neighbor_k', type=int, default=3)
    p.add_argument('--super_alpha', type=float, default=1.0)
    p.add_argument('--episode_test_num', type=int, default=15,
                    help='episodes per pair; distractor sampling varies per episode')
    p.add_argument('--top_k', type=int, default=5,
                    help='how many top candidate classes to keep per pair in the report')
    p.add_argument('--max_pairs', type=int, default=None,
                    help='debug: only run the first N pairs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    p.add_argument('--cpu', action='store_true', default=False)
    return p


def run_pair(model, base1, base2, all_classes, paths_by_class, transform, opt, device):
    """Run opt.episode_test_num episodes for one (base1, base2) pair.
    Returns (support_hits, support_totals, attraction_hits, attraction_totals)
    where support_* are keyed by 'base1'/'base2' and attraction_* are keyed
    by candidate class name (any class not used as support this episode)."""
    other_classes = [c for c in all_classes if c not in (base1, base2)]
    rng = random.Random((opt.seed, base1, base2).__hash__())
    super_col = opt.way_num

    support_hits = {'base1': 0, 'base2': 0}
    support_totals = {'base1': 0, 'base2': 0}
    attraction_hits = defaultdict(int)
    attraction_totals = defaultdict(int)

    for ep in range(opt.episode_test_num):
        distractors = rng.sample(other_classes, opt.way_num - 2)
        support_classes = [base1, base2] + distractors  # cols 0..way_num-1

        support_imgs = []
        used_paths = defaultdict(set)
        ok = True
        for cname in support_classes:
            paths = paths_by_class[cname]
            if len(paths) < opt.shot_num:
                ok = False
                break
            sampled = rng.sample(paths, opt.shot_num)
            used_paths[cname].update(sampled)
            for p in sampled:
                support_imgs.append(transform(tsp.Image.open(p).convert('RGB')))
        if not ok:
            continue
        support_tensor = torch.stack(support_imgs).to(device)

        query_imgs = []
        query_meta = []  # (kind, key) kind='support' -> key in {0,1} for base1/base2 col; kind='candidate' -> key=class name
        for col, cname in enumerate(support_classes[:2]):
            avail = [p for p in paths_by_class[cname] if p not in used_paths[cname]]
            n = min(opt.query_num, len(avail))
            for p in rng.sample(avail, n):
                query_imgs.append(transform(tsp.Image.open(p).convert('RGB')))
                query_meta.append(('support', 'base1' if col == 0 else 'base2'))

        candidate_classes = [c for c in other_classes if c not in distractors]
        for cname in candidate_classes:
            avail = paths_by_class[cname]  # never used as support this episode
            n = min(opt.query_num, len(avail))
            for p in rng.sample(avail, n):
                query_imgs.append(transform(tsp.Image.open(p).convert('RGB')))
                query_meta.append(('candidate', cname))

        if not query_imgs:
            continue
        query_tensor = torch.stack(query_imgs).to(device)

        with torch.no_grad():
            out = model(query_tensor, support_tensor)   # [Q, way_num+1]
        pred = out.argmax(dim=1).cpu().tolist()

        for (kind, key), p in zip(query_meta, pred):
            if kind == 'support':
                support_totals[key] += 1
                gt_col = 0 if key == 'base1' else 1
                if p == gt_col:
                    support_hits[key] += 1
            else:
                attraction_totals[key] += 1
                if p == super_col:
                    attraction_hits[key] += 1

    return support_hits, support_totals, dict(attraction_hits), dict(attraction_totals)


def main(argv=None):
    opt = build_parser().parse_args(argv)
    opt.cuda = bool(opt.cuda and not opt.cpu and torch.cuda.is_available())
    device = torch.device('cuda' if opt.cuda else 'cpu')
    cudnn.benchmark = opt.cuda

    places_dir = os.path.expanduser(opt.places_dir)
    if not os.path.isdir(places_dir):
        raise FileNotFoundError(f'Places365 directory not found: {places_dir}')

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    os.makedirs(os.path.dirname(opt.out_json) or '.', exist_ok=True)

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

    all_classes = tsp.discover_classes(places_dir)
    print(f'{len(all_classes)} classes found: {all_classes}')
    transform = tsp.build_transform(opt.imageSize)

    paths_by_class = {c: tsp.list_class_images(Path(places_dir) / c) for c in all_classes}

    pairs = list(itertools.combinations(sorted(all_classes), 2))
    if opt.max_pairs is not None:
        pairs = pairs[:opt.max_pairs]
    print(f'{len(pairs)} pairs to screen (C({len(all_classes)},2))')

    report = {
        'meta': {
            'places_dir': places_dir,
            'encoder_model': opt.encoder_model,
            'classifier_model': opt.classifier_model,
            'way_num': opt.way_num,
            'shot_num': opt.shot_num,
            'query_num': opt.query_num,
            'neighbor_k': opt.neighbor_k,
            'super_alpha': opt.super_alpha,
            'episode_test_num': opt.episode_test_num,
            'seed': opt.seed,
            'n_classes': len(all_classes),
            'n_pairs': len(pairs),
            'device': str(device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'pairs': [],
    }

    t_start = time.time()
    for idx, (base1, base2) in enumerate(pairs):
        t0 = time.time()
        s_hits, s_totals, a_hits, a_totals = run_pair(
            model, base1, base2, all_classes, paths_by_class, transform, opt, device)

        rates = []
        for cname, total in a_totals.items():
            hits = a_hits.get(cname, 0)
            rates.append({'candidate': cname, 'hits': hits, 'total': total,
                           'attraction_rate': 100.0 * hits / total if total > 0 else None})
        rates.sort(key=lambda r: (r['attraction_rate'] if r['attraction_rate'] is not None else -1),
                   reverse=True)

        known_super = KNOWN_SUPERS.get(frozenset([base1, base2]))
        known_rank = None
        if known_super is not None:
            for rank, r in enumerate(rates, start=1):
                if r['candidate'] == known_super:
                    known_rank = rank
                    break

        entry = {
            'base1': base1, 'base2': base2,
            'base1_acc': (100.0 * s_hits['base1'] / s_totals['base1']) if s_totals['base1'] > 0 else None,
            'base2_acc': (100.0 * s_hits['base2'] / s_totals['base2']) if s_totals['base2'] > 0 else None,
            'known_super': known_super,
            'known_super_rank': known_rank,
            'top_candidates': rates[:opt.top_k],
            'elapsed_sec': time.time() - t0,
        }
        report['pairs'].append(entry)

        if (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(pairs) - idx - 1) / rate if rate > 0 else float('nan')
            top1 = rates[0]['candidate'] if rates else 'n/a'
            print(f'[{idx+1}/{len(pairs)}] {base1} + {base2} -> top candidate: {top1} '
                  f'({rates[0]["attraction_rate"]:.1f}% )  known={known_super} rank={known_rank}  '
                  f'elapsed={elapsed:.0f}s eta={eta:.0f}s')

    with open(opt.out_json, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nWrote {opt.out_json}  (total time {time.time()-t_start:.0f}s)')


if __name__ == '__main__':
    main()
