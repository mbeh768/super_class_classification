#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the JSONL shards produced by research/screen_allpairs.py
(one line per pair, base1/base2 drawn from all ~365 Places365 categories,
candidates restricted to the 21 curated classes) into a report.

At 66,430 pairs a "print every row" table page (as the smaller C(21,2)
report does) would be ~2,400 PDF pages -- useless. Instead this writes:
  - a title/methodology page
  - a validation page (do the 10 known triplets recover their true super
    class as a top candidate, same check as the smaller screen)
  - a top-N discoveries bar chart (highest attraction rate, excluding
    already-known triplets)
  - a top-100 table (for browsing in the PDF itself)
  - a full CSV of all pairs' top candidate, written alongside the PDF, for
    sorting/filtering in a spreadsheet

Usage:
  python reports/gen_places_allpairs_pdf.py --jsonl_glob "./results/allpairs_shard*.jsonl" \
      --out_pdf ./results/places_superclass_allpairs.pdf \
      --out_csv ./results/places_superclass_allpairs.csv
"""
import argparse
import csv
import glob
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


TOP1_COLOR = '#DD8452'
KNOWN_COLOR = '#55A868'
GRID_COLOR = '#B0B0B0'
CI_Z = 1.96


def wilson_ci(hits, total, z=CI_Z):
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


def load_pairs(jsonl_glob):
    pairs = []
    files = sorted(glob.glob(jsonl_glob))
    if not files:
        raise FileNotFoundError(f'No files matched {jsonl_glob}')
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    return pairs, files


def make_title_page(pdf, n_pairs, n_files):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.75, 'Super-Class Screening: All Places365 Pairs', ha='center',
              fontsize=22, weight='bold')
    fig.text(0.5, 0.69, f'C(365,2) combinatorial search, {n_pairs:,} pairs '
              f'merged from {n_files} shard(s)', ha='center', fontsize=13, color='#555555')
    fig.text(0.5, 0.55,
              'base1/base2/distractors drawn from all Places365 categories on disk;\n'
              'candidate query pool restricted to the 21 originally-curated classes\n'
              '(otherwise the per-episode query set would explode to ~360 classes).\n\n'
              'Question answered: does ANY pair of Places365 categories attract toward\n'
              'one of our known super-classes or constituents -- a search for alternate\n'
              'constituent pairs of relationships we already know about, wider than the\n'
              'original C(21,2) search for brand-new super-class concepts.',
              ha='center', fontsize=10.5, color='#444444', linespacing=1.7)
    pdf.savefig(fig)
    plt.close(fig)


def make_validation_page(pdf, known_pairs):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.set_title('Validation: Do Known Super-Classes Still Rank #1?', fontsize=16, weight='bold', pad=20)

    if not known_pairs:
        fig.text(0.5, 0.5, 'No known triplets found in this merged data.',
                  ha='center', fontsize=12, color='#777777')
        pdf.savefig(fig)
        plt.close(fig)
        return

    if all(p['known_super_rank'] is None for p in known_pairs):
        fig.text(0.5, 0.62,
                  'Not applicable to this run.',
                  ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.48,
                  "This run's candidate pool does not include any of the 7 known super-class\n"
                  "names (boathouse, auto_factory, crevasse, mountain_path, mountain_snowy,\n"
                  "operating_room, street) by design, so the true super can never appear in\n"
                  "top_candidates for these 10 pairs -- \"not in top_k\" here is expected, not\n"
                  "a failed recovery. See the discoveries pages below for what this run's own\n"
                  "candidate pool actually surfaces.",
                  ha='center', va='center', fontsize=10, color='#555555', linespacing=1.7)
        pdf.savefig(fig)
        plt.close(fig)
        return

    n_top1 = sum(1 for p in known_pairs if p['known_super_rank'] == 1)
    n_top3 = sum(1 for p in known_pairs if p['known_super_rank'] is not None and p['known_super_rank'] <= 3)
    summary = (f"{n_top1}/{len(known_pairs)} known pairs recover their true super-class as #1.\n"
               f"{n_top3}/{len(known_pairs)} rank it in the top 3.")
    fig.text(0.5, 0.86, summary, ha='center', fontsize=12, weight='bold')

    col_labels = ['base1', 'base2', 'known super', 'rank', "#1 candidate", "#1 rate [95% CI]"]
    rows = []
    for p in known_pairs:
        top1 = p['top_candidates'][0] if p['top_candidates'] else None
        if top1 and top1['attraction_rate'] is not None:
            lo, hi = wilson_ci(top1['hits'], top1['total'])
            rate_str = f"{top1['attraction_rate']:.1f}% [{lo:.0f}-{hi:.0f}]"
        else:
            rate_str = '-'
        rows.append([p['base1'], p['base2'], p['known_super'],
                     str(p['known_super_rank']) if p['known_super_rank'] else 'not in top_k',
                     top1['candidate'] if top1 else '-', rate_str])

    ax_t = fig.add_axes([0.05, 0.15, 0.9, 0.6])
    ax_t.axis('off')
    table = ax_t.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#333333')
        table[0, j].set_text_props(color='white', weight='bold')
    for i, p in enumerate(known_pairs, start=1):
        if p['known_super_rank'] == 1:
            table[i, 3].set_facecolor('#C6E0B4')
        elif p['known_super_rank'] and p['known_super_rank'] <= 3:
            table[i, 3].set_facecolor('#FFE699')
        elif p['known_super_rank'] is None:
            table[i, 3].set_facecolor('#F4B6B6')
        else:
            table[i, 3].set_facecolor('#FBE5D6')

    pdf.savefig(fig)
    plt.close(fig)


def scored_discoveries(pairs, exclude_known=True):
    scored = []
    for p in pairs:
        if not p['top_candidates']:
            continue
        top1 = p['top_candidates'][0]
        if top1['attraction_rate'] is None:
            continue
        if exclude_known and p.get('known_super') == top1['candidate']:
            continue
        scored.append({
            'base1': p['base1'], 'base2': p['base2'], 'candidate': top1['candidate'],
            'rate': top1['attraction_rate'], 'hits': top1['hits'], 'total': top1['total'],
            'known_super': p.get('known_super'),
        })
    scored.sort(key=lambda r: r['rate'], reverse=True)
    return scored


def make_top_discoveries_page(pdf, scored, n=30):
    top = scored[:n]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_title(f'Top {len(top)} Candidate Pairs Across All of Places365\n'
                 '(highest super-column attraction rate, excluding already-known triplets)',
                 fontsize=13, weight='bold')
    labels = [f"{r['base1']} + {r['base2']} -> {r['candidate']}" for r in top]
    rates = [r['rate'] for r in top]
    errs = np.array([ci_err(r['hits'], r['total'], r['rate']) for r in top]).T
    y = np.arange(len(top))
    colors = [KNOWN_COLOR if r['known_super'] else TOP1_COLOR for r in top]
    ax.barh(y, rates, xerr=errs, capsize=2, error_kw={'elinewidth': 0.8, 'ecolor': '#333333'},
            color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Attraction rate (%) of top candidate into the super column  (error bars: 95% Wilson CI)')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', color=GRID_COLOR, linewidth=0.6, alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.5, 0.01,
              'NOTE: at episode_test_num=3, query_num=2 each rate is based on only ~6 queries per '
              'pair -- treat this as a coarse first-pass filter, not a final result.',
              ha='center', fontsize=8, color='#999999', style='italic')
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    pdf.savefig(fig)
    plt.close(fig)


def make_top_table_pages(pdf, scored, total_n, n=100, per_page=28):
    top = scored[:n]
    for start in range(0, len(top), per_page):
        chunk = top[start:start + per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title(f'Top {n} Discoveries ({start+1}-{start+len(chunk)})  '
                     f'-- full {total_n:,}-pair table in the companion CSV', fontsize=11, weight='bold')
        col_labels = ['base1', 'base2', 'candidate', 'rate', '95% CI', 'n']
        rows = []
        for r in chunk:
            lo, hi = wilson_ci(r['hits'], r['total'])
            rows.append([r['base1'], r['base2'], r['candidate'], f"{r['rate']:.1f}%",
                         f"[{lo:.0f}-{hi:.0f}]", str(r['total'])])
        table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#333333')
            table[0, j].set_text_props(color='white', weight='bold')
        pdf.savefig(fig)
        plt.close(fig)


def write_csv(path, pairs):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['base1', 'base2', 'base1_acc', 'base2_acc', 'known_super', 'known_super_rank',
                    'top1_candidate', 'top1_rate', 'top1_hits', 'top1_total',
                    'top2_candidate', 'top2_rate'])
        for p in pairs:
            tc = p['top_candidates']
            c1 = tc[0] if len(tc) > 0 else {}
            c2 = tc[1] if len(tc) > 1 else {}
            w.writerow([
                p['base1'], p['base2'], p.get('base1_acc'), p.get('base2_acc'),
                p.get('known_super') or '', p.get('known_super_rank') or '',
                c1.get('candidate', ''), c1.get('attraction_rate', ''),
                c1.get('hits', ''), c1.get('total', ''),
                c2.get('candidate', ''), c2.get('attraction_rate', ''),
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl_glob', required=True)
    ap.add_argument('--out_pdf', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--top_n', type=int, default=30, help='rows in the bar chart')
    ap.add_argument('--top_n_table', type=int, default=100, help='rows in the PDF table pages')
    args = ap.parse_args()

    pairs, files = load_pairs(args.jsonl_glob)
    known_pairs = [p for p in pairs if p.get('known_super')]
    scored = scored_discoveries(pairs)

    print(f'Loaded {len(pairs):,} pairs from {len(files)} file(s); {len(known_pairs)} known triplets found.')

    with PdfPages(args.out_pdf) as pdf:
        make_title_page(pdf, len(pairs), len(files))
        make_validation_page(pdf, known_pairs)
        make_top_discoveries_page(pdf, scored, n=args.top_n)
        make_top_table_pages(pdf, scored, len(pairs), n=args.top_n_table)

    write_csv(args.out_csv, pairs)
    print(f'Wrote {args.out_pdf}')
    print(f'Wrote {args.out_csv}')


if __name__ == '__main__':
    main()
