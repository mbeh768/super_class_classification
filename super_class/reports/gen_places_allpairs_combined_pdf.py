#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile the three overnight combinatorial screens (curated-21 known-super
pool, setB, setC -- 63 disjoint candidate classes total, same C(365,2)=66,430
base pairs each) into ONE report.

For each base pair, the three runs each tested a different, non-overlapping
21-class candidate pool. This merges their top_candidates lists per pair and
re-ranks across all 63 candidates together, so the "top discoveries" page
reflects the single best candidate found for that pair across the FULL
screened space, not just whichever pool happened to be run.

Usage:
  python reports/gen_places_allpairs_combined_pdf.py \
      --curated_glob "results/allpairs_shard*.jsonl" \
      --setb_glob "results/setB_shard*.jsonl" \
      --setc_glob "results/setC_shard*.jsonl" \
      --out_pdf results/places_superclass_allpairs_combined.pdf \
      --out_csv results/places_superclass_allpairs_combined.csv
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
POOL_COLORS = {'curated21': '#4C72B0', 'setB': '#DD8452', 'setC': '#55A868'}
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


def load_pool(jsonl_glob, default_tag):
    pairs = {}
    files = sorted(glob.glob(jsonl_glob))
    if not files:
        raise FileNotFoundError(f'No files matched {jsonl_glob}')
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d.setdefault('run_tag', default_tag)
                pairs[(d['base1'], d['base2'])] = d
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curated_glob', required=True)
    ap.add_argument('--setb_glob', required=True)
    ap.add_argument('--setc_glob', required=True)
    ap.add_argument('--out_pdf', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--top_n', type=int, default=30)
    ap.add_argument('--top_n_table', type=int, default=100)
    args = ap.parse_args()

    pools = {
        'curated21': load_pool(args.curated_glob, 'curated21'),
        'setB': load_pool(args.setb_glob, 'setB'),
        'setC': load_pool(args.setc_glob, 'setC'),
    }
    keys = set(pools['curated21']) | set(pools['setB']) | set(pools['setC'])
    print(f"curated21={len(pools['curated21']):,}  setB={len(pools['setB']):,}  "
          f"setC={len(pools['setC']):,}  union_pairs={len(keys):,}")

    merged = []
    for key in keys:
        base1, base2 = key
        combined_candidates = []
        known_super = None
        known_rank_curated = None
        per_pool_top1 = {}
        for tag, pool in pools.items():
            entry = pool.get(key)
            if entry is None:
                continue
            if entry.get('known_super'):
                known_super = entry['known_super']
            if tag == 'curated21':
                known_rank_curated = entry.get('known_super_rank')
            for c in entry['top_candidates']:
                if c['attraction_rate'] is None:
                    continue
                combined_candidates.append({**c, 'pool': tag})
            if entry['top_candidates']:
                per_pool_top1[tag] = entry['top_candidates'][0]
        combined_candidates.sort(key=lambda r: r['attraction_rate'], reverse=True)
        merged.append({
            'base1': base1, 'base2': base2,
            'known_super': known_super,
            'known_super_rank_curated21': known_rank_curated,
            'combined_top': combined_candidates[:5],
            'per_pool_top1': per_pool_top1,
        })

    known_pairs = [p for p in merged if p['known_super']]

    with PdfPages(args.out_pdf) as pdf:
        # Title / methodology page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.88, 'Super-Class Screening: Combined Report', ha='center',
                  fontsize=22, weight='bold')
        fig.text(0.5, 0.81, 'Curated-21 + setB + setC = 63 disjoint candidate classes, '
                  'C(365,2)=66,430 base pairs each', ha='center', fontsize=12.5, color='#555555')
        fig.text(0.5, 0.55,
                  'Three independent overnight runs, each screening the SAME 66,430 base pairs\n'
                  'against a DIFFERENT, non-overlapping 21-class candidate pool:\n\n'
                  '  curated21 -- the 7 known super-classes + 14 known constituents\n'
                  '  setB      -- 21 fresh classes (seed=1), run on worker A\n'
                  '  setC      -- 21 fresh classes (seed=2), run on worker B\n\n'
                  'This report merges all three pools\' top_candidates per pair and re-ranks\n'
                  'across the full 63-class space, so "top discoveries" below reflect the best\n'
                  'candidate found for each pair across everything screened -- not just one pool.',
                  ha='center', va='center', fontsize=10.5, color='#333333', linespacing=1.8)
        fig.text(0.5, 0.08,
                  'NOTE: at episode_test_num=3, query_num=2 each rate is based on only ~6 queries\n'
                  'per pair per pool -- treat this as a coarse first-pass filter, not a final result.',
                  ha='center', fontsize=9, color='#999999', style='italic', linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)

        # Validation page (curated21 only -- the only pool where it's meaningful)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Validation (curated21 pool only): Known Super-Classes', fontsize=15, weight='bold', pad=30)
        n_top1 = sum(1 for p in known_pairs if p['known_super_rank_curated21'] == 1)
        n_top3 = sum(1 for p in known_pairs if p['known_super_rank_curated21'] is not None
                     and p['known_super_rank_curated21'] <= 3)
        fig.text(0.5, 0.78, f'{n_top1}/{len(known_pairs)} known pairs recover their true super-class as #1.\n'
                  f'{n_top3}/{len(known_pairs)} rank it in the top 3.\n'
                  '(setB/setC cannot recover these -- their pools never contain the true super, by design.)',
                  ha='center', fontsize=11, weight='bold', linespacing=1.6)
        col_labels = ['base1', 'base2', 'known super', 'rank (curated21)']
        rows = [[p['base1'], p['base2'], p['known_super'],
                 str(p['known_super_rank_curated21']) if p['known_super_rank_curated21'] else 'not in top_k']
                for p in known_pairs]
        ax_t = fig.add_axes([0.15, 0.15, 0.7, 0.55])
        ax_t.axis('off')
        table = ax_t.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.9)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#333333')
            table[0, j].set_text_props(color='white', weight='bold')
        pdf.savefig(fig)
        plt.close(fig)

        # Top discoveries across all 63 candidates
        scored = []
        for p in merged:
            if not p['combined_top']:
                continue
            top1 = p['combined_top'][0]
            if p['known_super'] == top1['candidate']:
                continue
            scored.append({
                'base1': p['base1'], 'base2': p['base2'], 'candidate': top1['candidate'],
                'pool': top1['pool'], 'rate': top1['attraction_rate'],
                'hits': top1['hits'], 'total': top1['total'],
            })
        scored.sort(key=lambda r: r['rate'], reverse=True)
        top = scored[:args.top_n]

        fig, ax = plt.subplots(figsize=(11, 10))
        ax.set_title(f'Top {len(top)} Discoveries Across All 63 Candidate Classes\n'
                     '(best candidate per pair, merged across curated21/setB/setC,\n'
                     'excluding already-known triplets)', fontsize=12.5, weight='bold')
        labels = [f"{r['base1']} + {r['base2']} -> {r['candidate']}  [{r['pool']}]" for r in top]
        rates = [r['rate'] for r in top]
        errs = np.array([ci_err(r['hits'], r['total'], r['rate']) for r in top]).T
        y = np.arange(len(top))
        colors = [POOL_COLORS[r['pool']] for r in top]
        ax.barh(y, rates, xerr=errs, capsize=2, error_kw={'elinewidth': 0.8, 'ecolor': '#333333'},
                color=colors, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Attraction rate (%) into the super column  (error bars: 95% Wilson CI)')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', color=GRID_COLOR, linewidth=0.6, alpha=0.5, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in POOL_COLORS.values()]
        ax.legend(handles, POOL_COLORS.keys(), loc='lower right', fontsize=8, title='source pool')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Top table pages
        top_table = scored[:args.top_n_table]
        per_page = 24
        for start in range(0, len(top_table), per_page):
            chunk = top_table[start:start + per_page]
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            fig.text(0.5, 0.95, f'Top {args.top_n_table} Discoveries ({start+1}-{start+len(chunk)})  '
                     f'-- full {len(merged):,}-pair table in the companion CSV',
                     ha='center', fontsize=11, weight='bold')
            col_labels = ['base1', 'base2', 'candidate', 'pool', 'rate', '95% CI', 'n']
            rows = []
            for r in chunk:
                lo, hi = wilson_ci(r['hits'], r['total'])
                rows.append([r['base1'], r['base2'], r['candidate'], r['pool'],
                             f"{r['rate']:.1f}%", f"[{lo:.0f}-{hi:.0f}]", str(r['total'])])
            ax_t = fig.add_axes([0.05, 0.03, 0.9, 0.85])
            ax_t.axis('off')
            table = ax_t.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            for j in range(len(col_labels)):
                table[0, j].set_facecolor('#333333')
                table[0, j].set_text_props(color='white', weight='bold')
            pdf.savefig(fig)
            plt.close(fig)

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['base1', 'base2', 'known_super', 'known_super_rank_curated21',
                    'best_candidate', 'best_pool', 'best_rate', 'best_hits', 'best_total',
                    'curated21_top1', 'curated21_top1_rate',
                    'setB_top1', 'setB_top1_rate',
                    'setC_top1', 'setC_top1_rate'])
        for p in merged:
            best = p['combined_top'][0] if p['combined_top'] else {}
            pp = p['per_pool_top1']
            w.writerow([
                p['base1'], p['base2'], p['known_super'] or '', p['known_super_rank_curated21'] or '',
                best.get('candidate', ''), best.get('pool', ''), best.get('attraction_rate', ''),
                best.get('hits', ''), best.get('total', ''),
                pp.get('curated21', {}).get('candidate', ''), pp.get('curated21', {}).get('attraction_rate', ''),
                pp.get('setB', {}).get('candidate', ''), pp.get('setB', {}).get('attraction_rate', ''),
                pp.get('setC', {}).get('candidate', ''), pp.get('setC', {}).get('attraction_rate', ''),
            ])

    print(f'Wrote {args.out_pdf}')
    print(f'Wrote {args.out_csv}')


if __name__ == '__main__':
    main()
