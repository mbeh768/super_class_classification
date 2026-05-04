#!/usr/bin/env python3
"""
Generate two figures from the three-way liger discrimination test:
  1. Accuracy bar chart: lion / tiger / liger per-type accuracy with the
     super-class column active.
  2. Argmax distribution chart: stacked bar showing where each query type
     routes (lion col, tiger col, distractor avg, super col).

Saves to --outf (default ./figures/).

Usage:
  python gen_figure_ThreeWay.py \
      --resume ./results/<run_dir>/ \
      --dataset_dir ./dataset/miniImageNet
"""

from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import ImageFile
import sys
sys.dont_write_bytecode = True

import dataset.general_dataloader as FewShotDataloader
import models.network as FewShotNet
import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

model_trained = './results/SGD_Cosine_Lr0.05_Finetune_DN4_Conv64F_Local_Epoch_10_miniImageNet_84_84_5Way_1Shot/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./dataset/miniImageNet')
parser.add_argument('--data_name', default='miniImageNet')
parser.add_argument('--resume', default=model_trained)
parser.add_argument('--query_dir', default='./ligers/scraped')
parser.add_argument('--super_constituents', nargs='+', default=['n02129165', 'n02129604'])
parser.add_argument('--encoder_model', default='Conv64F_Local')
parser.add_argument('--classifier_model', default='DN4_SuperClass')
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--neighbor_k', type=int, default=3)
parser.add_argument('--episode_test_num', type=int, default=50)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--mode', default='test')
parser.add_argument('--super_gamma', type=float, default=0.6)
parser.add_argument('--checkpoint', default='model_best.pth.tar',
                    help='checkpoint filename inside --resume dir (e.g. epoch_0.pth.tar)')
parser.add_argument('--outf', default='./figures/')
parser.add_argument('--cuda', action='store_true', default=True)
opt = parser.parse_args()
opt.cuda = opt.cuda and torch.cuda.is_available()
device = torch.device('cuda' if opt.cuda else 'cpu')
cudnn.benchmark = opt.cuda
opt.base_per_super = len(opt.super_constituents)


if __name__ == '__main__':
    os.makedirs(opt.outf, exist_ok=True)

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
    model.classifier.base_per_super = opt.base_per_super
    if hasattr(model.classifier, 'super_gamma'):
        model.classifier.super_gamma = opt.super_gamma

    F_dummy = open(os.devnull, 'w')
    checkpoint = utils.get_resume_file(os.path.join(opt.resume, opt.checkpoint), F_dummy)
    F_dummy.close()
    if checkpoint is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in checkpoint['model'].items()}, strict=False)
    else:
        print('WARNING: no checkpoint, using random weights')
    model.eval()

    C = opt.way_num + 1
    super_col = opt.way_num
    constituents = opt.super_constituents
    type_names = {i: constituents[i] for i in range(len(constituents))}
    type_names[super_col] = 'liger'

    hits = defaultdict(int)
    totals = defaultdict(int)
    col_counts = defaultdict(lambda: [0] * C)

    test_loader = FewShotDataloader.get_ThreeWay_dataloader(opt, mode=opt.mode, query_num=opt.query_num)

    with torch.no_grad():
        for ep, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):
            input1 = torch.cat(query_images, 0).to(device)
            input2 = torch.cat(support_images, 0).squeeze(0).to(device)
            input2 = input2.contiguous().view(-1, input2.size(2), input2.size(3), input2.size(4))
            targets = torch.tensor(query_targets).long()
            out = model(input1, input2)
            pred = out.argmax(dim=1).cpu()
            for q_idx in range(pred.size(0)):
                gt = targets[q_idx].item()
                p = pred[q_idx].item()
                totals[gt] += 1
                if p == gt:
                    hits[gt] += 1
                col_counts[gt][p] += 1

    # ── labels and data ──────────────────────────────────────────────────────
    query_types = [0, 1, super_col]           # lion, tiger, liger
    display_names = ['Lion', 'Tiger', 'Liger']
    accs = [100.0 * hits[t] / totals[t] if totals[t] > 0 else 0.0 for t in query_types]

    # argmax distribution per query type: collapse distractor columns into one
    def dist_for(gt):
        n = totals[gt]
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0
        lion_pct  = 100.0 * col_counts[gt][0] / n
        tiger_pct = 100.0 * col_counts[gt][1] / n
        dist_pct  = 100.0 * sum(col_counts[gt][2:super_col]) / n
        super_pct = 100.0 * col_counts[gt][super_col] / n
        return lion_pct, tiger_pct, dist_pct, super_pct

    dists = [dist_for(t) for t in query_types]
    lion_d  = [d[0] for d in dists]
    tiger_d = [d[1] for d in dists]
    dist_d  = [d[2] for d in dists]
    super_d = [d[3] for d in dists]

    # ── Figure 1: accuracy bar chart ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    colors = ['#4878CF', '#D65F5F', '#6ACC65']
    bars = ax1.bar(display_names, accs, color=colors, alpha=0.88, edgecolor='black', linewidth=0.7)
    for bar, val in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 '%.1f%%' % val, ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax1.set_title('Three-Way Discrimination: Lion / Tiger / Liger', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax1.set_axisbelow(True)
    fig1.tight_layout()
    p1 = os.path.join(opt.outf, 'threeway_accuracy.png')
    fig1.savefig(p1, dpi=150)
    print('Saved:', p1)

    # ── Figure 2: stacked argmax distribution ────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(display_names))
    w = 0.55
    b1 = ax2.bar(x, lion_d,  w, label='→ Lion column',       color='#4878CF', alpha=0.88)
    b2 = ax2.bar(x, tiger_d, w, bottom=lion_d,                label='→ Tiger column',      color='#D65F5F', alpha=0.88,
                 bottom=np.array(lion_d))
    b3 = ax2.bar(x, dist_d,  w, label='→ Distractor columns', color='#aaaaaa', alpha=0.75,
                 bottom=np.array(lion_d) + np.array(tiger_d))
    b4 = ax2.bar(x, super_d, w, label='→ Super (liger) column', color='#6ACC65', alpha=0.88,
                 bottom=np.array(lion_d) + np.array(tiger_d) + np.array(dist_d))
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, fontsize=12)
    ax2.set_ylabel('Queries routed to column (%)', fontsize=12)
    ax2.set_title('Argmax Distribution by Query Type', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax2.set_axisbelow(True)
    fig2.tight_layout()
    p2 = os.path.join(opt.outf, 'threeway_argmax_dist.png')
    fig2.savefig(p2, dpi=150)
    print('Saved:', p2)
