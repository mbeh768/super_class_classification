#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the super_alpha sweep (constituent vs. super-class accuracy tradeoff)
for one or more triplet sets, produced by research/sweep_alpha.py.

Usage:
  python reports/gen_alpha_sweep_pdf.py \
      --sweep curated=results/alpha_sweep_curated.json \
      --sweep discovered=results/alpha_sweep_discovered.json \
      --out results/alpha_sweep.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


COLORS = {'curated': '#4C72B0', 'discovered': '#DD8452'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='append', required=True, help='name=path.json, repeatable')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sweeps = {}
    for item in args.sweep:
        name, path = item.split('=', 1)
        sweeps[name] = json.load(open(path))

    with PdfPages(args.out) as pdf:
        fig, ax = plt.subplots(figsize=(9, 7))
        for name, data in sweeps.items():
            alphas = sorted(float(a) for a in data['alphas'].keys())
            constituent = []
            super_acc = []
            for a in alphas:
                key = None
                for k in data['alphas'].keys():
                    if float(k) == a:
                        key = k
                        break
                agg = data['alphas'][key]['aggregate']
                constituent.append(agg['constituent_combined']['acc'])
                super_acc.append(agg['super']['acc'])
            color = COLORS.get(name, None)
            ax.plot(alphas, constituent, marker='o', linestyle='-', color=color,
                    label=f'{name}: constituent')
            ax.plot(alphas, super_acc, marker='s', linestyle='--', color=color,
                    label=f'{name}: super-class', alpha=0.8)

        chance = 100.0 / 6
        ax.axhline(chance, color='#999999', linestyle=':', linewidth=1)
        ax.text(1.0, chance + 1.5, f'chance = {chance:.1f}%', ha='right', fontsize=8.5, color='#777777')
        ax.set_xlabel('super_alpha (spread-penalty coefficient)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('super_alpha Sweep: Constituent vs. Super-Class Accuracy', fontsize=13, weight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, color='#DDDDDD', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='center right', fontsize=9)
        fig.text(0.5, 0.01,
                  'alpha=0 removes the penalty entirely (super wins almost every query, '
                  'constituent accuracy collapses); alpha=1 is full penalty (current default).',
                  ha='center', fontsize=8.5, color='#777777', style='italic')
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        pdf.savefig(fig)
        plt.close(fig)

        # Table page
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.axis('off')
        ax.set_title('Sweep Detail', fontsize=13, weight='bold', pad=20)
        col_labels = ['set', 'alpha', 'base1', 'base2', 'constituent', 'super']
        rows = []
        for name, data in sweeps.items():
            for k in sorted(data['alphas'].keys(), key=float):
                agg = data['alphas'][k]['aggregate']
                rows.append([name, k, f"{agg['base1']['acc']:.1f}%", f"{agg['base2']['acc']:.1f}%",
                             f"{agg['constituent_combined']['acc']:.1f}%", f"{agg['super']['acc']:.1f}%"])
        ax_t = fig.add_axes([0.1, 0.1, 0.8, 0.75])
        ax_t.axis('off')
        table = ax_t.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#333333')
            table[0, j].set_text_props(color='white', weight='bold')
        pdf.savefig(fig)
        plt.close(fig)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
