#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combinatorial super-class screen over ALL Places365 categories (base1/base2
drawn from the full ~365-class set), but with the per-episode candidate
query pool restricted to the 21 originally-curated classes -- otherwise the
"which class gets attracted into the super column" query set would explode
to ~360 classes/episode and make the whole run computationally infeasible
(see research/screen_curated.py's docstring for the base design this
extends).

Answers: "does ANY pair of Places365 categories attract toward one of our 7
known super-classes (or any of the 14 known constituents)?" -- a much wider
search for alternate constituent pairs of the relationships we already know
about, rather than the C(21,2) search for brand-new super-class concepts.

Supports sharding (--shard_id/--num_shards) so multiple worker processes can
run concurrently against the same GPU -- the previous single-process run
left the GPU at ~0% utilization (bottlenecked on CPU image decode + small
batch sizes), so parallel workers should give a large wall-clock speedup.

Output is JSON Lines (one JSON object per pair, flushed immediately) rather
than a single JSON blob, since this run is long and unattended -- partial
results stay readable even if a worker is killed mid-run.

Usage (single shard, for smoke testing):
  python research/screen_allpairs.py --places_dir /data/places365 \
      --out_jsonl ./results/allpairs_shard0.jsonl --shard_id 0 --num_shards 4 --max_pairs 20
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

# Each shard is one process among --num_shards run concurrently on the same
# machine; without this, PyTorch/BLAS default to an intra-op thread pool
# sized to ALL visible cores *per process*, so N shards means N-way thread
# oversubscription. Harmless-ish on bare-metal Linux; under WSL2 the extra
# context-switch cost made a 4-shard run ~4.6x slower than the equivalent
# bare-metal run (0.47 vs 2.18 pairs/s) despite a faster GPU. Must be set
# before importing torch/numpy so the native thread pools pick it up at init.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import torch
import torch.backends.cudnn as cudnn

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from PIL import ImageFile

import Test_Super_Places as tsp
import models.network as FewShotNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


CURATED_CLASSES = [
    'assembly_line', 'auto_factory', 'boathouse', 'building_facade', 'canyon',
    'car_interior', 'clean_room', 'cottage', 'crevasse', 'forest_path',
    'forest_road', 'hospital_room', 'house', 'lake-natural', 'mountain',
    'mountain_path', 'mountain_snowy', 'ocean', 'operating_room', 'snowfield',
    'street',
]

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


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--places_dir', required=True,
                    help='root containing ALL Places365 class folders (base1/base2/distractor pool)')
    p.add_argument('--out_jsonl', required=True)
    p.add_argument('--encoder_model', default='DINOv2_Local')
    p.add_argument('--classifier_model', default='DN4_SuperClass')
    p.add_argument('--imageSize', type=int, default=224)
    p.add_argument('--way_num', type=int, default=5)
    p.add_argument('--shot_num', type=int, default=1)
    p.add_argument('--query_num', type=int, default=2)
    p.add_argument('--neighbor_k', type=int, default=3)
    p.add_argument('--super_alpha', type=float, default=1.0)
    p.add_argument('--episode_test_num', type=int, default=3)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--max_pairs', type=int, default=None, help='debug: cap pairs processed by this shard')
    p.add_argument('--candidate_classes', type=str, default=None,
                    help='comma-separated override for the candidate query pool '
                         '(default: the 21 originally-curated classes, CURATED_CLASSES)')
    p.add_argument('--run_tag', type=str, default='',
                    help='free-text tag echoed into each output row, for distinguishing '
                         'multiple candidate-pool runs merged later')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    p.add_argument('--cpu', action='store_true', default=False)
    return p


def run_pair(model, base1, base2, all_classes, candidate_set, paths_by_class, transform, opt, device):
    other_classes = [c for c in all_classes if c not in (base1, base2)]
    candidates_fixed = [c for c in candidate_set if c not in (base1, base2)]
    rng = random.Random((opt.seed, base1, base2).__hash__())
    super_col = opt.way_num

    support_hits = {'base1': 0, 'base2': 0}
    support_totals = {'base1': 0, 'base2': 0}
    attraction_hits = defaultdict(int)
    attraction_totals = defaultdict(int)

    for ep in range(opt.episode_test_num):
        distractors = rng.sample(other_classes, opt.way_num - 2)
        support_classes = [base1, base2] + distractors

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
        query_meta = []
        for col, cname in enumerate(support_classes[:2]):
            avail = [p for p in paths_by_class[cname] if p not in used_paths[cname]]
            n = min(opt.query_num, len(avail))
            for p in rng.sample(avail, n):
                query_imgs.append(transform(tsp.Image.open(p).convert('RGB')))
                query_meta.append(('support', 'base1' if col == 0 else 'base2'))

        candidates_ep = [c for c in candidates_fixed if c not in distractors]
        for cname in candidates_ep:
            avail = paths_by_class[cname]
            n = min(opt.query_num, len(avail))
            for p in rng.sample(avail, n):
                query_imgs.append(transform(tsp.Image.open(p).convert('RGB')))
                query_meta.append(('candidate', cname))

        if not query_imgs:
            continue
        query_tensor = torch.stack(query_imgs).to(device)

        with torch.no_grad():
            out = model(query_tensor, support_tensor)
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
        raise FileNotFoundError(f'Places dir not found: {places_dir}')

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    os.makedirs(os.path.dirname(opt.out_jsonl) or '.', exist_ok=True)

    model = FewShotNet.define_model(
        encoder_model=opt.encoder_model, classifier_model=opt.classifier_model,
        norm='batch', way_num=opt.way_num, shot_num=opt.shot_num,
        neighbor_k=opt.neighbor_k, init_type='normal', use_gpu=opt.cuda,
    )
    model.classifier.base_per_super = 2
    model.classifier.super_alpha = opt.super_alpha
    model.eval()

    all_classes = tsp.discover_classes(places_dir)
    if opt.candidate_classes:
        requested = [c.strip() for c in opt.candidate_classes.split(',') if c.strip()]
        candidate_set = [c for c in requested if c in all_classes]
        missing = set(requested) - set(candidate_set)
        if missing:
            print(f'[shard {opt.shard_id}] WARNING: candidate classes not found on disk: {missing}', flush=True)
    else:
        candidate_set = [c for c in CURATED_CLASSES if c in all_classes]
    print(f'[shard {opt.shard_id}/{opt.num_shards}] {len(all_classes)} total classes, '
          f'{len(candidate_set)} candidates (tag={opt.run_tag!r})', flush=True)
    transform = tsp.build_transform(opt.imageSize)
    paths_by_class = {c: tsp.list_class_images(Path(places_dir) / c) for c in all_classes}

    all_pairs = list(itertools.combinations(sorted(all_classes), 2))
    my_pairs = all_pairs[opt.shard_id::opt.num_shards]
    if opt.max_pairs is not None:
        my_pairs = my_pairs[:opt.max_pairs]
    print(f'[shard {opt.shard_id}] {len(my_pairs)} pairs assigned (of {len(all_pairs)} total)', flush=True)

    t_start = time.time()
    with open(opt.out_jsonl, 'w') as f:
        for idx, (base1, base2) in enumerate(my_pairs):
            s_hits, s_totals, a_hits, a_totals = run_pair(
                model, base1, base2, all_classes, candidate_set, paths_by_class, transform, opt, device)

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
                'run_tag': opt.run_tag,
            }
            f.write(json.dumps(entry) + '\n')
            f.flush()

            if (idx + 1) % 100 == 0 or idx == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(my_pairs) - idx - 1) / rate if rate > 0 else float('nan')
                print(f'[shard {opt.shard_id}] [{idx+1}/{len(my_pairs)}] '
                      f'{base1} + {base2}  elapsed={elapsed:.0f}s eta={eta:.0f}s rate={rate:.2f}pairs/s',
                      flush=True)

    print(f'[shard {opt.shard_id}] done: {len(my_pairs)} pairs in {time.time()-t_start:.0f}s -> {opt.out_jsonl}',
          flush=True)


if __name__ == '__main__':
    main()
