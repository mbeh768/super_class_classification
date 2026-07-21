#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a PDF report from the JSON produced by research/report_superclass.py.

Usage:
  python reports/gen_places_superclass_pdf.py --json ./results/places_superclass_report.json \
      --out ./results/places_superclass_report.pdf
"""
import argparse
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


CONSTITUENT_COLOR = '#4C72B0'
SUPER_COLOR = '#DD8452'
GRID_COLOR = '#B0B0B0'
CI_Z = 1.96  # 95% confidence


def acc_or_nan(x):
    return x if x is not None else float('nan')


def wilson_ci(hits, total, z=CI_Z):
    """95% Wilson score interval for a binomial proportion, as percentages.
    More reliable than the normal approximation at the sample sizes here
    (some cells have n<40). Returns (low_pct, high_pct); nan if total==0."""
    if not total:
        return float('nan'), float('nan')
    phat = hits / total
    denom = 1 + z**2 / total
    center = phat + z**2 / (2 * total)
    adj = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2))
    low = (center - adj) / denom
    high = (center + adj) / denom
    return 100.0 * max(low, 0.0), 100.0 * min(high, 1.0)


def ci_err(hits, total, acc_pct):
    """(lower_err, upper_err) around acc_pct for matplotlib's yerr/xerr, in pct points."""
    low, high = wilson_ci(hits, total)
    if np.isnan(low):
        return 0.0, 0.0
    return max(acc_pct - low, 0.0), max(high - acc_pct, 0.0)


def make_title_page(pdf, meta):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.75, 'Zero-Shot Super-Class Evaluation', ha='center', fontsize=24, weight='bold')
    fig.text(0.5, 0.69, 'Places365 (abridged, 10 imgs/class)', ha='center', fontsize=15, color='#555555')

    lines = [
        f"Encoder: {meta['encoder_model']}    Classifier: {meta['classifier_model']}",
        f"Paradigm: way_num={meta['way_num']} (2 constituents + {meta['way_num']-2} distractors "
        f"in support, super-class column added -> {meta['way_num']+1} total columns)",
        f"shot_num={meta['shot_num']}   query_num={meta['query_num']}/type   "
        f"neighbor_k={meta['neighbor_k']}   super_alpha={meta['super_alpha']}",
        f"episode_test_num={meta['episode_test_num']} per triplet   seed={meta['seed']}",
        f"Classes available: {meta['n_classes_available']}   Device: {meta['device']}",
        f"Data source: {meta['places_dir']}",
        f"Generated: {meta['timestamp']}",
    ]
    y = 0.56
    for line in lines:
        fig.text(0.5, y, line, ha='center', fontsize=11)
        y -= 0.045

    fig.text(0.5, 0.08,
              'Support = 2 constituent classes + 3 random distractors (never includes the super class).\n'
              'Queries drawn separately from each constituent and from the super class itself;\n'
              'a query is "correct" only if the model\'s argmax lands on its true column.',
              ha='center', fontsize=9.5, color='#555555', linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def make_aggregate_page(pdf, report):
    agg = report['aggregate']
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title('Aggregate Accuracy (micro-averaged across all triplets)', fontsize=16, weight='bold', pad=20)

    labels = ['Constituent 1\n(base1)', 'Constituent 2\n(base2)', 'Both Constituents\n(combined)', 'Super-Class']
    keys = ['base1', 'base2', 'constituent_combined', 'super']
    accs = [acc_or_nan(agg[k]['acc']) for k in keys]
    totals = [agg[k]['total'] for k in keys]
    hits = [agg[k]['hits'] for k in keys]
    colors = [CONSTITUENT_COLOR, CONSTITUENT_COLOR, CONSTITUENT_COLOR, SUPER_COLOR]
    errs = np.array([ci_err(h, n, a) for h, n, a in zip(hits, totals, accs)]).T  # shape (2, 4)

    ax_bar = fig.add_axes([0.12, 0.35, 0.76, 0.45])
    bars = ax_bar.bar(labels, accs, yerr=errs, capsize=5, color=colors, width=0.55, zorder=3,
                       error_kw={'linewidth': 1.3, 'ecolor': '#333333'})
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel('Accuracy (%)')
    ax_bar.grid(axis='y', color=GRID_COLOR, linewidth=0.6, alpha=0.5, zorder=0)
    ax_bar.spines[['top', 'right']].set_visible(False)
    chance = 100.0 / (report['meta']['way_num'] + 1)
    ax_bar.axhline(chance, color='#999999', linestyle='--', linewidth=1, zorder=2)
    ax_bar.text(len(labels) - 0.4, chance + 2, f'chance = {chance:.1f}%', fontsize=8.5, color='#777777')
    for bar, acc, n, h, err_hi in zip(bars, accs, totals, hits, errs[1]):
        label = (f'{acc:.1f}%\n(n={n})' if not np.isnan(acc) else 'n/a')
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err_hi + 2.5, label,
                    ha='center', fontsize=10, weight='bold')
    fig.text(0.5, 0.255, 'Error bars: 95% Wilson score confidence interval', ha='center',
              fontsize=8.5, color='#777777', style='italic')

    gap = accs[2] - accs[3] if not (np.isnan(accs[2]) or np.isnan(accs[3])) else float('nan')
    summary = (f"Combined constituent accuracy: {accs[2]:.1f}%   |   "
               f"Super-class accuracy: {accs[3]:.1f}%   |   Gap: {gap:+.1f} pts")
    fig.text(0.5, 0.20, summary, ha='center', fontsize=11.5, weight='bold')
    fig.text(0.5, 0.15,
              'A positive gap means the model recognizes constituent (real, familiar) classes\n'
              'more reliably than it recognizes the synthetic super-class relationship.',
              ha='center', fontsize=9.5, color='#555555', linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def make_per_triplet_page(pdf, report):
    triplets = [t for t in report['triplets'] if not t.get('skipped')]
    n = len(triplets)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_title('Per-Triplet Accuracy: Constituents vs. Super-Class', fontsize=15, weight='bold', pad=16)

    x = np.arange(n)
    width = 0.25
    b1 = [acc_or_nan(t['base1']['acc']) for t in triplets]
    b2 = [acc_or_nan(t['base2']['acc']) for t in triplets]
    sup = [acc_or_nan(t['super']['acc']) for t in triplets]
    b1_err = np.array([ci_err(t['base1']['hits'], t['base1']['total'], acc_or_nan(t['base1']['acc'])) for t in triplets]).T
    b2_err = np.array([ci_err(t['base2']['hits'], t['base2']['total'], acc_or_nan(t['base2']['acc'])) for t in triplets]).T
    sup_err = np.array([ci_err(t['super']['hits'], t['super']['total'], acc_or_nan(t['super']['acc'])) for t in triplets]).T

    err_kw = {'capsize': 2.5, 'elinewidth': 0.9, 'ecolor': '#333333'}
    ax.bar(x - width, b1, width, yerr=b1_err, label='base1', color='#4C72B0', zorder=3, error_kw=err_kw)
    ax.bar(x, b2, width, yerr=b2_err, label='base2', color='#55A868', zorder=3, error_kw=err_kw)
    ax.bar(x + width, sup, width, yerr=sup_err, label='super', color=SUPER_COLOR, zorder=3, error_kw=err_kw)

    xlabels = [f"{t['base1_name']}\n+ {t['base2_name']}\n-> {t['super_name']}" for t in triplets]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.6, alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', fontsize=9, ncol=3)
    fig.text(0.5, 0.965, 'Error bars: 95% Wilson score confidence interval', ha='center',
              fontsize=8, color='#777777', style='italic')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


def make_table_page(pdf, report):
    triplets = report['triplets']
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title('Per-Triplet Detail  (accuracy ± 95% Wilson CI)', fontsize=15, weight='bold', pad=10)

    col_labels = ['base1', 'base2', 'super', 'base1 acc', 'base2 acc', 'super acc', 'n/type', 'episodes']
    rows = []
    for t in triplets:
        if t.get('skipped'):
            rows.append([t['base1_name'], t['base2_name'], t['super_name'], '-', '-', '-', '-', t.get('reason', 'skipped')])
            continue
        n_per_type = t['base1']['total']

        def fmt(key):
            h, n, a = t[key]['hits'], t[key]['total'], t[key]['acc']
            lo, hi = wilson_ci(h, n)
            return f"{a:.1f}% [{lo:.0f}-{hi:.0f}]"

        rows.append([
            t['base1_name'], t['base2_name'], t['super_name'],
            fmt('base1'), fmt('base2'), fmt('super'),
            str(n_per_type), f"{t['elapsed_sec']:.1f}s",
        ])

    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', weight='bold')
    for i, row in enumerate(rows, start=1):
        for j in range(len(col_labels)):
            if j == 5 and row[5] != '-':
                table[i, j].set_facecolor('#FBE5D6')
            elif j in (3, 4) and row[j] != '-':
                table[i, j].set_facecolor('#DCE6F1')

    pdf.savefig(fig)
    plt.close(fig)


def make_confusion_pages(pdf, report):
    triplets = [t for t in report['triplets'] if not t.get('skipped')]
    way_num = report['meta']['way_num']
    per_page = 4
    for start in range(0, len(triplets), per_page):
        chunk = triplets[start:start + per_page]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        axes = axes.flatten()
        for ax, t in zip(axes, chunk):
            col_names = [t['base1_name'], t['base2_name'], 'd1', 'd2', 'd3', 'super'][:way_num + 1]
            mat = np.array([
                t['base1']['argmax_dist'],
                t['base2']['argmax_dist'],
                t['super']['argmax_dist'],
            ], dtype=float)
            row_totals = mat.sum(axis=1, keepdims=True)
            row_totals[row_totals == 0] = 1
            pct = 100.0 * mat / row_totals

            im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
            ax.set_xticks(range(way_num + 1))
            ax.set_xticklabels(col_names, fontsize=7, rotation=45, ha='right')
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"query: {t['base1_name']}", f"query: {t['base2_name']}", f"query: {t['super_name']} (super)"],
                                fontsize=7.5)
            ax.set_title(f"{t['base1_name']} + {t['base2_name']} -> {t['super_name']}", fontsize=9.5, weight='bold')
            for r in range(3):
                for c in range(way_num + 1):
                    ax.text(c, r, f'{pct[r, c]:.0f}', ha='center', va='center', fontsize=7,
                             color='white' if pct[r, c] > 50 else '#333333')
        for ax in axes[len(chunk):]:
            ax.axis('off')
        fig.suptitle('Argmax Distribution (% of queries predicted into each column)', fontsize=12, weight='bold')
        fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.12, wspace=0.4, hspace=0.6)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.json) as f:
        report = json.load(f)

    with PdfPages(args.out) as pdf:
        make_title_page(pdf, report['meta'])
        make_aggregate_page(pdf, report)
        make_per_triplet_page(pdf, report)
        make_table_page(pdf, report)
        make_confusion_pages(pdf, report)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
