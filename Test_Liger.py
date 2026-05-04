#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot super-class test with REAL external queries (e.g. liger images vs
lion+tiger supports plus distractors).

Lion and tiger (or whatever synsets the user passes via --super_constituents)
are pinned to episode positions 0..N-1 so `ImgtoSuperClass_Metric` pools them
into the super column. Query images come from --query_dir and have no ground
truth among the base classes; the target label is set to `way_num` (the super
column) so "super wins" is counted as success.

Reported metrics (averaged across episodes, distractor sets vary):
  * super_win_rate — % of queries whose argmax over [0..way_num] is the super col.
  * per-column distribution — fraction of argmax landing on each column.
  * super_margin    — mean(super_score - max(base_scores)). Positive = super dominates.
  * constituent_leak — % queries whose argmax is one of the constituent columns
	(i.e. model sees "lion" or "tiger" rather than "liger"). Useful diagnostic.
"""

from __future__ import print_function
import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
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
					help='directory of external query images')
parser.add_argument('--super_constituents', nargs='+',
					default=['n02129165', 'n02129604'],
					help='synset IDs pinned to positions 0..N-1 (default: lion + tiger)')
parser.add_argument('--mode', default='test', choices=['train', 'val', 'test'])
parser.add_argument('--resume', default=model_trained, type=str)
parser.add_argument('--encoder_model', default='Conv64F_Local')
parser.add_argument('--classifier_model', default='DN4_SuperClass')
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--episode_test_num', type=int, default=50,
					help='number of episodes (each redraws distractors)')
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--neighbor_k', type=int, default=3)
parser.add_argument('--super_alpha', type=float, default=1.0,
                    help='spread-penalty coefficient: 0=no penalty, 1=full (default 1.0); '
                         'lower values help when the hybrid is visually asymmetric')
parser.add_argument('--cuda', action='store_true', default=True)
opt = parser.parse_args()
opt.cuda = opt.cuda and torch.cuda.is_available()
device = torch.device('cuda' if opt.cuda else 'cpu')
cudnn.benchmark = opt.cuda

# `base_per_super` is derived from the constituents list, not a separate flag.
opt.base_per_super = len(opt.super_constituents)


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

	opt.outf = opt.resume if str(opt.resume).endswith('/') else opt.resume + '/'
	if not os.path.exists(opt.outf):
		os.makedirs(opt.outf)
	F_txt = open(os.path.join(opt.outf, 'Test_Liger_results.txt'), 'a+')
	print('==================== Liger Test ====================')
	print('==================== Liger Test ====================', file=F_txt)
	print(opt)
	print(opt, file=F_txt)

	best_path = os.path.join(opt.resume, 'model_best.pth.tar')
	checkpoint = utils.get_resume_file(best_path, F_txt)
	if checkpoint is not None:
		missing, unexpected = model.load_state_dict(
			{k: v.to(device) for k, v in checkpoint['model'].items()}, strict=False)
		print('missing keys   :', missing, file=F_txt)
		print('unexpected keys:', unexpected, file=F_txt)
	else:
		print('WARNING: no checkpoint loaded — encoder is random.', file=F_txt)

	model.eval()
	test_loader = FewShotDataloader.get_Liger_dataloader(opt, mode=opt.mode)

	C = opt.way_num + 1
	super_col = opt.way_num
	col_hits = torch.zeros(C, dtype=torch.long)
	super_margins = []
	super_wins = 0
	constituent_leaks = 0
	total_q = 0

	with torch.no_grad():
		for ep, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):
			input1 = torch.cat(query_images, 0).to(device)
			input2 = torch.cat(support_images, 0).squeeze(0).to(device)
			input2 = input2.contiguous().view(-1, input2.size(2), input2.size(3), input2.size(4))

			out = model(input1, input2)                       # [Q, way_num+1]
			pred = out.argmax(dim=1)                          # [Q]

			base_max = out[:, :opt.way_num].max(dim=1).values
			margin = out[:, super_col] - base_max
			super_margins.append(margin.detach().cpu())

			col_hits += torch.bincount(pred.cpu(), minlength=C)
			super_wins += int((pred == super_col).sum().item())
			constituent_leaks += int((pred < opt.base_per_super).sum().item())
			total_q += pred.numel()

			if ep < 3 or ep % 10 == 0:
				msg = ('ep {0:3d}  super_wins={1:3d}/{2}  pred_hist={3}  '
					   'mean_margin={4:+.3f}'.format(
						ep, int((pred == super_col).sum().item()), pred.numel(),
						torch.bincount(pred.cpu(), minlength=C).tolist(),
						margin.mean().item()))
				print(msg)
				print(msg, file=F_txt)

	super_win_rate = 100.0 * super_wins / total_q
	leak_rate = 100.0 * constituent_leaks / total_q
	col_dist = (col_hits.float() / total_q * 100.0).tolist()
	all_margins = torch.cat(super_margins)

	summary = (
		'\n==== Liger summary ({0} episodes x {1} queries = {2} total) ====\n'
		'super_constituents    : {3}\n'
		'super_win_rate        : {4:.2f}%\n'
		'constituent_leak_rate : {5:.2f}%  (argmax landed on a constituent base col)\n'
		'per-column argmax %%   : {6}\n'
		'super - max(base)     : mean={7:+.3f}  median={8:+.3f}  std={9:.3f}\n'
	).format(
		len(test_loader), int(total_q / max(len(test_loader), 1)), total_q,
		opt.super_constituents, super_win_rate, leak_rate,
		['{:.1f}'.format(x) for x in col_dist],
		all_margins.mean().item(),
		all_margins.median().item(),
		all_margins.std().item() if all_margins.numel() > 1 else 0.0)
	print(summary)
	print(summary, file=F_txt)
	F_txt.close()


if __name__ == '__main__':
	main()
