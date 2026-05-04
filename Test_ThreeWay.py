#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-way discrimination test: liger (super-class) vs lion vs tiger.

Each episode has:
  Support: lion (col 0) + tiger (col 1) as constituents, 3 random distractors.
  Queries: lion images → expect col 0
           tiger images → expect col 1
           liger images → expect col way_num (super-class column)

Reports per-query-type accuracy and per-column argmax distribution so you can
see whether the super-class score selectively responds to ligers vs their
constituent species.
"""

from __future__ import print_function
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
from PIL import ImageFile
import sys
sys.dont_write_bytecode = True

import dataset.general_dataloader as FewShotDataloader
import models.network as FewShotNet
import utils


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


model_trained = './results/SGD_Cosine_Lr0.05_DN4_Conv64F_Local_Epoch_30_miniImageNet_84_84_5Way_1Shot/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='./dataset/miniImageNet')
parser.add_argument('--query_dir', default='./ligers/scraped',
                    help='directory of liger query images')
parser.add_argument('--super_constituents', nargs='+',
                    default=['n02129165', 'n02129604'],
                    help='synset IDs for constituents (lion + tiger by default)')
parser.add_argument('--mode', default='test', choices=['train', 'val', 'test'])
parser.add_argument('--resume', default=model_trained)
parser.add_argument('--encoder_model', default='Conv64F_Local')
parser.add_argument('--classifier_model', default='DN4_SuperClass')
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--episode_test_num', type=int, default=50)
parser.add_argument('--query_num', type=int, default=15,
                    help='lion/tiger query images per episode')
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--neighbor_k', type=int, default=3)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--super_alpha', type=float, default=1.0,
                    help='spread-penalty coefficient: 0=no penalty, 1=full (default 1.0); '
                         'lower values help when the hybrid is visually asymmetric')
parser.add_argument('--super_gamma', type=float, default=0.6,
                    help='legacy parameter, unused in current scoring')
opt = parser.parse_args()
opt.cuda = opt.cuda and torch.cuda.is_available()
device = torch.device('cuda' if opt.cuda else 'cpu')
cudnn.benchmark = opt.cuda

opt.base_per_super = len(opt.super_constituents)


# Column labels for printing
def col_name(i, way_num, constituents):
    if i == way_num:
        return 'super(liger)'
    if i < len(constituents):
        return constituents[i]
    return 'distractor%d' % (i - len(constituents))


def main():
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
    model.classifier.super_alpha = opt.super_alpha
    model.classifier.super_gamma = opt.super_gamma

    opt.outf = opt.resume if str(opt.resume).endswith('/') else opt.resume + '/'
    os.makedirs(opt.outf, exist_ok=True)
    F_txt = open(os.path.join(opt.outf, 'Test_ThreeWay_results.txt'), 'a+')
    print('==================== Three-Way Test ====================')
    print('==================== Three-Way Test ====================', file=F_txt)
    print(opt, file=F_txt)

    best_path = os.path.join(opt.resume, 'model_best.pth.tar')
    checkpoint = utils.get_resume_file(best_path, F_txt)
    if checkpoint is not None:
        missing, unexpected = model.load_state_dict(
            {k: v.to(device) for k, v in checkpoint['model'].items()}, strict=False)
        print('missing keys   :', missing)
        print('unexpected keys:', unexpected)
    else:
        print('WARNING: no checkpoint loaded — encoder is random.')

    model.eval()
    test_loader = FewShotDataloader.get_ThreeWay_dataloader(
        opt, mode=opt.mode, query_num=opt.query_num)

    C = opt.way_num + 1
    super_col = opt.way_num
    constituents = opt.super_constituents

    # Accumulators keyed by ground-truth query type:
    #   0..bps-1 → constituent class index, way_num → liger
    type_names = {i: constituents[i] for i in range(len(constituents))}
    type_names[super_col] = 'liger'

    hits = defaultdict(int)       # correct predictions
    totals = defaultdict(int)     # total queries of that type
    col_counts = defaultdict(lambda: [0] * C)  # argmax distribution per type

    with torch.no_grad():
        for ep, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):
            input1 = torch.cat(query_images, 0).to(device)
            input2 = torch.cat(support_images, 0).squeeze(0).to(device)
            input2 = input2.contiguous().view(-1, input2.size(2), input2.size(3), input2.size(4))
            targets = torch.tensor(query_targets).long()

            out = model(input1, input2)   # [Q, way_num+1]
            pred = out.argmax(dim=1).cpu()

            for q_idx in range(pred.size(0)):
                gt = targets[q_idx].item()
                p = pred[q_idx].item()
                totals[gt] += 1
                if p == gt:
                    hits[gt] += 1
                col_counts[gt][p] += 1

            if ep < 3 or ep % 10 == 0:
                msg = 'ep %3d  pred_hist=%s' % (ep, pred.tolist()[:20])
                print(msg)
                print(msg, file=F_txt)

    print('\n==== Three-Way Summary (%d episodes) ====' % len(test_loader))
    print('\n==== Three-Way Summary (%d episodes) ====' % len(test_loader), file=F_txt)

    for gt_col in sorted(totals.keys()):
        name = type_names.get(gt_col, 'col%d' % gt_col)
        n = totals[gt_col]
        acc = 100.0 * hits[gt_col] / n if n > 0 else float('nan')
        dist = ['%s:%.1f%%' % (col_name(i, super_col, constituents),
                               100.0 * col_counts[gt_col][i] / n)
                for i in range(C)]
        line = ('  %-20s  n=%4d  acc=%6.2f%%  argmax_dist=[%s]'
                % (name, n, acc, ', '.join(dist)))
        print(line)
        print(line, file=F_txt)

    F_txt.close()


if __name__ == '__main__':
    main()
