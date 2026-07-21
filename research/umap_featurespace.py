#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the DINOv2 local-descriptor feature space for the discovered
super-class triplets, using the SAME constituent/super/distractor episode
regime as research/report_superclass.py / Test_Super_Places.py:

  Support: base1 (shot_num) + base2 (shot_num) + 3 random distractors.
  Queries: query_num images each of base1, base2, and the super class
           (super is NEVER in support).

Instead of running them through the DN4_SuperClass classifier, this pulls
the raw local patch descriptors straight out of the backbone (before the
classifier's k-NN pooling), subsamples a fixed number of patches per image,
projects the pooled descriptor cloud for each triplet with UMAP, and plots
it colored by role (base1 support/query, base2 support/query, distractor,
super query). The question this answers visually: do a super-class query's
local descriptors sit "between" its two constituents' descriptor clouds, or
somewhere else entirely?

Usage:
  python research/umap_featurespace.py \
      --places_dir /data/places365 \
      --csv_path ./dataset/Places365/discovered_supers.csv \
      --out_npz ./results/umap_featurespace.npz
"""
from __future__ import print_function
import argparse
import os
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# Repo root is one directory up from research/ -- add it so
# `import Test_Super_Places` / `import models.network` resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import ImageFile

import Test_Super_Places as tsp
import models.network as FewShotNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--places_dir', required=True)
    p.add_argument('--csv_path', required=True)
    p.add_argument('--out_npz', required=True)
    p.add_argument('--encoder_model', default='DINOv2_Local')
    p.add_argument('--imageSize', type=int, default=224)
    p.add_argument('--way_num', type=int, default=5)
    p.add_argument('--shot_num', type=int, default=1)
    p.add_argument('--query_num', type=int, default=10)
    p.add_argument('--patches_per_image', type=int, default=32,
                    help='random subsample of the 256 (16x16) local descriptors per image')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    p.add_argument('--cpu', action='store_true', default=False)
    return p


def extract_patches(model, images, device, patches_per_image, rng):
    """images: list of PIL-transformed tensors. Returns [N*patches_per_image, D]."""
    if not images:
        return np.zeros((0, 1))
    batch = torch.stack(images).to(device)
    with torch.no_grad():
        feat = model.features(batch)          # [N, D, H, W]
    N, D, H, W = feat.shape
    feat = feat.permute(0, 2, 3, 1).reshape(N, H * W, D).cpu().numpy()
    n_patches = min(patches_per_image, H * W)
    out = np.empty((N * n_patches, D), dtype=np.float32)
    for i in range(N):
        idx = rng.sample(range(H * W), n_patches)
        out[i * n_patches:(i + 1) * n_patches] = feat[i, idx]
    return out


def build_triplet_descriptors(model, base1, base2, super_cls, places_dir, all_classes,
                               transform, opt, device):
    other_classes = [c for c in all_classes if c not in (base1, base2, super_cls)]
    rng_py = random.Random((opt.seed, base1, base2, super_cls).__hash__())
    distractors = rng_py.sample(other_classes, opt.way_num - 2)

    paths = {c: tsp.list_class_images(Path(places_dir) / c)
             for c in [base1, base2, super_cls] + distractors}

    used = {base1: set(), base2: set()}
    b1_support_paths = rng_py.sample(paths[base1], opt.shot_num)
    b2_support_paths = rng_py.sample(paths[base2], opt.shot_num)
    used[base1].update(b1_support_paths)
    used[base2].update(b2_support_paths)

    distractor_paths = []
    for d in distractors:
        distractor_paths += rng_py.sample(paths[d], opt.shot_num)

    b1_query_avail = [p for p in paths[base1] if p not in used[base1]]
    b2_query_avail = [p for p in paths[base2] if p not in used[base2]]
    b1_query_paths = rng_py.sample(b1_query_avail, min(opt.query_num, len(b1_query_avail)))
    b2_query_paths = rng_py.sample(b2_query_avail, min(opt.query_num, len(b2_query_avail)))
    super_query_paths = rng_py.sample(paths[super_cls], min(opt.query_num, len(paths[super_cls])))

    def load(paths_list):
        return [transform(tsp.Image.open(p).convert('RGB')) for p in paths_list]

    patch_rng = random.Random((opt.seed, 'patches', base1, base2, super_cls).__hash__())
    roles = {}
    roles['base1_support'] = extract_patches(model, load(b1_support_paths), device, opt.patches_per_image, patch_rng)
    roles['base2_support'] = extract_patches(model, load(b2_support_paths), device, opt.patches_per_image, patch_rng)
    roles['distractor_support'] = extract_patches(model, load(distractor_paths), device, opt.patches_per_image, patch_rng)
    roles['base1_query'] = extract_patches(model, load(b1_query_paths), device, opt.patches_per_image, patch_rng)
    roles['base2_query'] = extract_patches(model, load(b2_query_paths), device, opt.patches_per_image, patch_rng)
    roles['super_query'] = extract_patches(model, load(super_query_paths), device, opt.patches_per_image, patch_rng)
    return roles, distractors


def main(argv=None):
    opt = build_parser().parse_args(argv)
    opt.cuda = bool(opt.cuda and not opt.cpu and torch.cuda.is_available())
    device = torch.device('cuda' if opt.cuda else 'cpu')

    places_dir = os.path.expanduser(opt.places_dir)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    model = FewShotNet.define_model(
        encoder_model=opt.encoder_model, classifier_model='DN4_SuperClass',
        norm='batch', way_num=opt.way_num, shot_num=opt.shot_num,
        neighbor_k=3, init_type='normal', use_gpu=opt.cuda,
    )
    model.eval()

    triplets = tsp.load_triplets(opt.csv_path)
    all_classes = tsp.discover_classes(places_dir)
    transform = tsp.build_transform(opt.imageSize)

    results = {}
    for base1, base2, super_cls in triplets:
        print(f'{base1} + {base2} -> {super_cls}')
        roles, distractors = build_triplet_descriptors(
            model, base1, base2, super_cls, places_dir, all_classes, transform, opt, device)
        key = f'{base1}__{base2}__{super_cls}'
        for role, arr in roles.items():
            results[f'{key}::{role}'] = arr
        results[f'{key}::distractors'] = np.array(distractors)

    np.savez_compressed(opt.out_npz, **results)
    print(f'Wrote {opt.out_npz}')


if __name__ == '__main__':
    main()
