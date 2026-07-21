#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a PDF report from the JSON produced by research/screen_curated.py:
a combinatorial screen over all C(21,2)=210 class pairs looking for which
third class's photos get pulled into the DN4 super-class column most
strongly. Includes a validation section (does the method recover the 7
known super-classes from potential_supers.csv as a top candidate?) before
presenting the full ranked discovery table.

Usage:
  python reports/gen_places_screen_pdf.py --json ./results/places_superclass_screen.json \
      --out ./results/places_superclass_screen.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


TOP1_COLOR = '#DD8452'
KNOWN_COLOR = '#55A868'
GRID_COLOR = '#B0B0B0'
CI_Z = 1.96  # 95% confidence


def wilson_ci(hits, total, z=CI_Z):
    """95% Wilson score interval for a binomial proportion, as percentages."""
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
    low, high = wilson_ci(hits, total)
    if np.isnan(low):
        return 0.0, 0.0
    return max(acc_pct - low, 0.0), max(high - acc_pct, 0.0)


def make_title_page(pdf, meta):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.75, 'Super-Class Screening', ha='center', fontsize=24, weight='bold')
    fig.text(0.5, 0.69, 'Combinatorial search over all class pairs, Places365 (abridged)',
              ha='center', fontsize=14, color='#555555')

    lines = [
        f"Encoder: {meta['encoder_model']}    Classifier: {meta['classifier_model']}",
        f"way_num={meta['way_num']} (2 constituents + {meta['way_num']-2} distractors, "
        f"super column at index {meta['way_num']})",
        f"shot_num={meta['shot_num']}   query_num={meta['query_num']}/candidate/episode   "
        f"neighbor_k={meta['neighbor_k']}   super_alpha={meta['super_alpha']}",
        f"episode_test_num={meta['episode_test_num']} per pair   seed={meta['seed']}",
        f"Classes: {meta['n_classes']}   Pairs screened: {meta['n_pairs']} (= C({meta['n_classes']},2))",
        f"Device: {meta['device']}   Generated: {meta['timestamp']}",
    ]
    y = 0.58
    for line in lines:
        fig.text(0.5, y, line, ha='center', fontsize=11)
        y -= 0.045

    fig.text(0.5, 0.15,
              'For every pair, support = base1 + base2 + 3 random distractors (way_num=5).\n'
              'Every OTHER class\'s real photos are queried in the same forward pass; a class\n'
              'attracted into the super column at a high rate is a candidate "natural hybrid"\n'
              'for that pair -- the same relationship the 7 curated triplets already encode.',
              ha='center', fontsize=9.5, color='#555555', linespacing=1.7)
    pdf.savefig(fig)
    plt.close(fig)


def make_validation_page(pdf, report):
    pairs = report['pairs']
    known = [p for p in pairs if p.get('known_super')]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title('Validation: Do Known Super-Classes Rank #1?', fontsize=16, weight='bold', pad=20)

    if not known:
        fig.text(0.5, 0.5, 'No known triplets present in this run\n'
                             '(use --max_pairs 0 or a full run to include them).',
                  ha='center', fontsize=12, color='#777777')
        pdf.savefig(fig)
        plt.close(fig)
        return

    n_top1 = sum(1 for p in known if p['known_super_rank'] == 1)
    n_top3 = sum(1 for p in known if p['known_super_rank'] is not None and p['known_super_rank'] <= 3)
    summary = (f"{n_top1}/{len(known)} known pairs recover their true super-class as the #1 "
               f"attraction candidate.\n{n_top3}/{len(known)} rank it in the top 3.")
    fig.text(0.5, 0.86, summary, ha='center', fontsize=12, weight='bold')

    col_labels = ['base1', 'base2', 'known super', 'rank', "#1 candidate", "#1 rate [95% CI]"]
    rows = []
    for p in known:
        top1 = p['top_candidates'][0] if p['top_candidates'] else None
        if top1 and top1['attraction_rate'] is not None:
            lo, hi = wilson_ci(top1['hits'], top1['total'])
            rate_str = f"{top1['attraction_rate']:.1f}% [{lo:.0f}-{hi:.0f}]"
        else:
            rate_str = '-'
        rows.append([
            p['base1'], p['base2'], p['known_super'],
            str(p['known_super_rank']) if p['known_super_rank'] else 'not in top_k',
            top1['candidate'] if top1 else '-',
            rate_str,
        ])

    ax_t = fig.add_axes([0.05, 0.15, 0.9, 0.6])
    ax_t.axis('off')
    table = ax_t.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', weight='bold')
    for i, p in enumerate(known, start=1):
        if p['known_super_rank'] == 1:
            table[i, 3].set_facecolor('#C6E0B4')
        elif p['known_super_rank'] and p['known_super_rank'] <= 3:
            table[i, 3].set_facecolor('#FFE699')
        elif p['known_super_rank'] is None:
            table[i, 3].set_facecolor('#F4B6B6')
        else:
            table[i, 3].set_facecolor('#FBE5D6')

    fig.text(0.5, 0.10, 'CI = 95% Wilson score confidence interval on the #1 candidate\'s attraction rate',
              ha='center', fontsize=8.5, color='#777777', style='italic')
    pdf.savefig(fig)
    plt.close(fig)


def make_top_discoveries_page(pdf, report, exclude_known=True, n=25):
    pairs = report['pairs']
    scored = []
    for p in pairs:
        if not p['top_candidates']:
            continue
        top1 = p['top_candidates'][0]
        if top1['attraction_rate'] is None:
            continue
        if exclude_known and p.get('known_super') == top1['candidate']:
            continue
        scored.append((p['base1'], p['base2'], top1['candidate'], top1['attraction_rate'],
                       p.get('known_super'), top1['hits'], top1['total']))
    scored.sort(key=lambda r: r[3], reverse=True)
    scored = scored[:n]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_title(f'Top {len(scored)} Newly-Discovered Candidate Super-Class Pairs\n'
                 '(highest super-column attraction rate, excluding already-known triplets)',
                 fontsize=13, weight='bold')
    labels = [f"{b1} + {b2} -> {cand}" for b1, b2, cand, rate, known, h, n_ in scored]
    rates = [rate for *_, rate, known, h, n_ in scored]
    errs = np.array([ci_err(h, n_, rate) for *_, rate, known, h, n_ in scored]).T
    y = np.arange(len(scored))
    colors = [KNOWN_COLOR if known else TOP1_COLOR for *_, known, h, n_ in scored]
    ax.barh(y, rates, xerr=errs, capsize=2.5, error_kw={'elinewidth': 0.9, 'ecolor': '#333333'},
            color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel('Attraction rate (%) of top candidate class into the super column  '
                   '(error bars: 95% Wilson CI)')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', color=GRID_COLOR, linewidth=0.6, alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_full_table_pages(pdf, report, per_page=28):
    pairs = sorted(report['pairs'],
                    key=lambda p: (p['top_candidates'][0]['attraction_rate']
                                    if p['top_candidates'] and p['top_candidates'][0]['attraction_rate'] is not None
                                    else -1),
                    reverse=True)
    col_labels = ['base1', 'base2', 'known super', 'known rank', 'top candidate', 'top rate', '2nd candidate', '2nd rate']
    for start in range(0, len(pairs), per_page):
        chunk = pairs[start:start + per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title(f'All Pairs Ranked by Top-Candidate Attraction Rate  '
                     f'({start+1}-{start+len(chunk)} of {len(pairs)})', fontsize=12, weight='bold')
        rows = []
        for p in chunk:
            tc = p['top_candidates']
            c1 = tc[0] if len(tc) > 0 else None
            c2 = tc[1] if len(tc) > 1 else None
            rows.append([
                p['base1'], p['base2'], p.get('known_super') or '-',
                str(p['known_super_rank']) if p.get('known_super_rank') else '-',
                c1['candidate'] if c1 else '-',
                f"{c1['attraction_rate']:.1f}%" if c1 and c1['attraction_rate'] is not None else '-',
                c2['candidate'] if c2 else '-',
                f"{c2['attraction_rate']:.1f}%" if c2 and c2['attraction_rate'] is not None else '-',
            ])
        table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1, 1.3)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#333333')
            table[0, j].set_text_props(color='white', weight='bold')
        for i, p in enumerate(chunk, start=1):
            if p.get('known_super'):
                table[i, 2].set_facecolor('#E2EFDA')
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
        make_validation_page(pdf, report)
        make_top_discoveries_page(pdf, report)
        make_full_table_pages(pdf, report)

    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
