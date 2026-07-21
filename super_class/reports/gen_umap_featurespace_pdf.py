#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a PDF of UMAP projections of the local-descriptor feature space
extracted by research/umap_featurespace.py, one plot per discovered
triplet plus an overview grid, colored by role (constituent support/query,
distractor, super-class query).

Usage:
  python reports/gen_umap_featurespace_pdf.py --npz results/umap_featurespace.npz \
      --out results/umap_featurespace.pdf
"""
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import umap


ROLE_STYLE = {
    'distractor_support': dict(color='#B0B0B0', marker='.', s=10, alpha=0.35, label='distractors (support)', zorder=1),
    'base1_support':      dict(color='#1f3d7a', marker='^', s=45, alpha=0.9,  label='base1 (support)', zorder=4),
    'base1_query':        dict(color='#4C72B0', marker='o', s=14, alpha=0.55, label='base1 (query)', zorder=2),
    'base2_support':      dict(color='#2d5a2d', marker='^', s=45, alpha=0.9,  label='base2 (support)', zorder=4),
    'base2_query':        dict(color='#55A868', marker='o', s=14, alpha=0.55, label='base2 (query)', zorder=2),
    'super_query':        dict(color='#DD8452', marker='*', s=36, alpha=0.85, label='super (query)', zorder=3),
}
ROLE_ORDER = ['distractor_support', 'base1_query', 'base2_query', 'super_query', 'base1_support', 'base2_support']


def load_triplets(npz):
    triplets = defaultdict(dict)
    for key in npz.files:
        triplet_key, role = key.split('::')
        triplets[triplet_key][role] = npz[key]
    return triplets


def fit_umap(roles, seed=0):
    order = [r for r in ROLE_ORDER if r in roles and roles[r].dtype != object and len(roles[r])]
    X = np.concatenate([roles[r] for r in order], axis=0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.15, metric='cosine', random_state=seed)
    emb = reducer.fit_transform(X)
    out = {}
    i = 0
    for r in order:
        n = len(roles[r])
        out[r] = emb[i:i + n]
        i += n
    return out


def plot_triplet(ax, triplet_key, emb, title_fontsize=11):
    base1, base2, super_cls = triplet_key.split('__')
    for role in ROLE_ORDER:
        if role not in emb:
            continue
        pts = emb[role]
        style = dict(ROLE_STYLE[role])
        label = style.pop('label')
        # substitute the real class names into the legend label
        label = (label.replace('base1', base1).replace('base2', base2).replace('super', super_cls))
        ax.scatter(pts[:, 0], pts[:, 1], label=label, **style)
    ax.set_title(f'{base1} + {base2} -> {super_cls}', fontsize=title_fontsize, weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    triplets = load_triplets(npz)
    print(f'{len(triplets)} triplets found')

    embeddings = {}
    for key, roles in triplets.items():
        print(f'  fitting UMAP for {key} ...')
        embeddings[key] = fit_umap(roles, seed=args.seed)

    with PdfPages(args.out) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.72, 'Local Descriptor Feature Space (UMAP)', ha='center', fontsize=22, weight='bold')
        fig.text(0.5, 0.65, 'DINOv2 patch descriptors, discovered super-class triplets', ha='center',
                  fontsize=13, color='#555555')
        fig.text(0.5, 0.45,
                  'For each triplet, support = base1 + base2 + 3 random distractors (never the super\n'
                  'class); queries are drawn separately from base1, base2, and the super class itself --\n'
                  'the exact same episode construction used for accuracy evaluation, just visualized\n'
                  'instead of classified. Each point is one local (16x16 patch grid) descriptor from the\n'
                  'frozen DINOv2 backbone, projected to 2D with UMAP (cosine metric, matching the\n'
                  'DN4 classifier\'s own similarity measure).\n\n'
                  'Question: do a super-class query\'s descriptors sit between its two constituents\'\n'
                  'clouds, or somewhere else in feature space entirely?',
                  ha='center', va='center', fontsize=10.5, color='#333333', linespacing=1.8)
        pdf.savefig(fig)
        plt.close(fig)

        # Overview grid (2x3)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()
        for ax, (key, emb) in zip(axes, embeddings.items()):
            plot_triplet(ax, key, emb, title_fontsize=10)
        for ax in axes[len(embeddings):]:
            ax.axis('off')
        # shared legend from the first subplot's handles, generic role names
        handles = []
        labels = []
        for role in ROLE_ORDER:
            style = dict(ROLE_STYLE[role])
            generic_label = style.pop('label').replace('base1', 'base1').replace('base2', 'base2').replace('super', 'super')
            style.pop('zorder', None)
            h = plt.Line2D([0], [0], marker=style.get('marker', 'o'), color='w',
                            markerfacecolor=style.get('color'), markersize=8, alpha=style.get('alpha', 1))
            handles.append(h)
            labels.append(generic_label)
        fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=8, frameon=False)
        fig.suptitle('Overview: All 6 Discovered Triplets', fontsize=14, weight='bold')
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # Full-page detail per triplet
        for key, emb in embeddings.items():
            fig, ax = plt.subplots(figsize=(10, 8.5))
            plot_triplet(ax, key, emb, title_fontsize=15)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
