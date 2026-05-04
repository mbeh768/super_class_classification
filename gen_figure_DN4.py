#!/usr/bin/env python3
"""
Generate a bar chart of DN4 5-way 1-shot accuracy with 95% CI error bars,
across 5 evaluation rounds. Saves figure to --outf.

Usage:
  python gen_figure_DN4.py --resume ./results/<run_dir>/ --dataset_dir ./dataset/miniImageNet
"""

from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
parser.add_argument('--encoder_model', default='Conv64F_Local')
parser.add_argument('--classifier_model', default='DN4')
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--neighbor_k', type=int, default=3)
parser.add_argument('--episode_test_num', type=int, default=1000)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--repeat_num', type=int, default=5)
parser.add_argument('--checkpoint', default='model_best.pth.tar',
                    help='checkpoint filename inside --resume dir (e.g. epoch_0.pth.tar)')
parser.add_argument('--outf', default='./figures/')
parser.add_argument('--train_aug', action='store_true', default=False)
parser.add_argument('--test_aug', action='store_true', default=False)
parser.add_argument('--aug_shot_num', type=int, default=20)
parser.add_argument('--episodeSize', type=int, default=1)
parser.add_argument('--episode_train_num', type=int, default=200)
parser.add_argument('--episode_val_num', type=int, default=200)
parser.add_argument('--cuda', action='store_true', default=True)
opt = parser.parse_args()
opt.cuda = opt.cuda and torch.cuda.is_available()
device = torch.device('cuda' if opt.cuda else 'cpu')
cudnn.benchmark = opt.cuda


def evaluate_round(test_loader, model, criterion):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for _, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):
            input1 = torch.cat(query_images, 0).to(device)
            input2 = torch.cat(support_images, 0).squeeze(0).to(device)
            input2 = input2.contiguous().view(-1, input2.size(2), input2.size(3), input2.size(4))
            target = torch.cat(query_targets, 0).to(device)
            output = model(input1, input2)
            prec1, _ = utils.accuracy(output, target, topk=(1, 3))
            accuracies.append(prec1)
    mean, h = utils.mean_confidence_interval(accuracies)
    return mean, h


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
    criterion = nn.CrossEntropyLoss().to(device)

    best_path = os.path.join(opt.resume, 'model_best.pth.tar')
    import io
    F_dummy = open(os.devnull, 'w')
    checkpoint = utils.get_resume_file(os.path.join(opt.resume, opt.checkpoint), F_dummy)
    F_dummy.close()
    if checkpoint is not None:
        model.load_state_dict(
            {k: v.to(device) for k, v in checkpoint['model'].items()}, strict=False)
    else:
        print('WARNING: no checkpoint found, using random weights')

    means, cis = [], []
    for r in range(opt.repeat_num):
        loader = FewShotDataloader.get_Fewshot_dataloader(opt, ['train', 'val', 'test'])[2]
        m, h = evaluate_round(loader, model, criterion)
        means.append(m)
        cis.append(h)
        print('Round %d: %.2f +/- %.2f' % (r, m, h))

    means = np.array([float(m) for m in means])
    cis = np.array([float(h) for h in cis])

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 4))
    rounds = np.arange(1, opt.repeat_num + 1)
    bars = ax.bar(rounds, means, yerr=cis, capsize=6,
                  color='steelblue', alpha=0.85, ecolor='black', linewidth=0.8)

    ax.axhline(means.mean(), color='firebrick', linestyle='--', linewidth=1.2,
               label='Mean %.2f%%' % means.mean())
    ax.set_xlabel('Evaluation Round', fontsize=12)
    ax.set_ylabel('5-Way 1-Shot Accuracy (%)', fontsize=12)
    ax.set_title('DN4 Baseline — miniImageNet 5-Way 1-Shot', fontsize=13)
    ax.set_xticks(rounds)
    ax.set_ylim(max(0, means.min() - 10), min(100, means.max() + 10))
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle=':', alpha=0.6)
    ax.set_axisbelow(True)

    out_path = os.path.join(opt.outf, 'dn4_accuracy.png')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print('Saved:', out_path)
