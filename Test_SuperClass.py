#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot super-class evaluation driver.

Loads a DN4 checkpoint (encoder is reused as-is; the DN4 head is replaced
by `ImgtoSuperClass_Metric` at inference) and loops over episodes from
`get_SuperClass_dataloader`. Reports:

  * base-only accuracy — argmax over columns [0..way_num), the standard DN4
    metric, unaffected by the super column.
  * full-head accuracy — argmax over all way_num+1 columns. Matches base-only
    when the super column loses; lower when super "steals" constituent queries.
  * super-steal rate — fraction of queries whose argmax is the super column.
  * geometric check — for queries whose true label is a constituent of the
    super-class (label < base_per_super), confirms super_score >= constituent_score.
    This should hold identically by construction (the super bank is a superset of
    the constituent bank; k-NN of a superset cannot be smaller than k-NN of a subset).

No training is performed.
"""

from __future__ import print_function
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
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
parser.add_argument('--dataset_dir', default='/data1/Liwenbin/Datasets/miniImageNet--ravi')
parser.add_argument('--data_name', default='miniImageNet')
parser.add_argument('--mode', default='test', choices=['train', 'val', 'test'])
parser.add_argument('--outf', default='./results/')
parser.add_argument('--resume', default=model_trained, type=str)
parser.add_argument('--encoder_model', default='Conv64F_Local')
parser.add_argument('--classifier_model', default='DN4_SuperClass')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--episodeSize', type=int, default=1)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--episode_train_num', type=int, default=10000)
parser.add_argument('--episode_val_num', type=int, default=1000)
parser.add_argument('--episode_test_num', type=int, default=1000)
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--neighbor_k', type=int, default=3)
parser.add_argument('--base_per_super', type=int, default=2,
					help='number of base classes pooled into the super-class')
parser.add_argument('--repeat_num', type=int, default=5)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--print_freq', '-p', default=100, type=int)
opt = parser.parse_args()
opt.cuda = opt.cuda and torch.cuda.is_available()
device = torch.device('cuda' if opt.cuda else 'cpu')
cudnn.benchmark = opt.cuda


# ============================ Evaluation loop ============================ #
def test_superclass(test_loader, model, criterion, F_txt):
	batch_time = utils.AverageMeter()
	losses = utils.AverageMeter()
	base_meter = utils.AverageMeter()      # argmax over base columns only
	full_meter = utils.AverageMeter()      # argmax over all way_num+1 columns
	steal_meter = utils.AverageMeter()     # % queries where super column wins
	geom_pass_total = 0                    # geometric invariant count
	geom_total = 0

	base_accs_ci = []                      # per-episode base accuracy for CI

	model.eval()
	end = time.time()
	for ep, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):

		input1 = torch.cat(query_images, 0).to(device)
		input2 = torch.cat(support_images, 0).squeeze(0).to(device)
		input2 = input2.contiguous().view(-1, input2.size(2), input2.size(3), input2.size(4))
		target = torch.cat(query_targets, 0).to(device)

		output = model(input1, input2)                      # [Q, way_num+1]
		loss = criterion(output[:, :opt.way_num], target)   # CE on base cols only

		# base-only accuracy
		base_pred = output[:, :opt.way_num].argmax(dim=1)
		base_correct = (base_pred == target).float().mean().item() * 100.0

		# full-head accuracy (target space is still base; super column = wrong)
		full_pred = output.argmax(dim=1)
		full_correct = (full_pred == target).float().mean().item() * 100.0

		# super-steal rate
		steal_rate = (full_pred == opt.way_num).float().mean().item() * 100.0

		# geometric invariant: for constituent queries, super_score >= constituent_score
		constituent_mask = target < opt.base_per_super
		if constituent_mask.any():
			super_scores = output[constituent_mask, opt.way_num]
			constituent_scores = output[constituent_mask].gather(
				1, target[constituent_mask].unsqueeze(1)).squeeze(1)
			geom_pass_total += int((super_scores >= constituent_scores - 1e-5).sum().item())
			geom_total += int(constituent_mask.sum().item())

		Q = target.size(0)
		losses.update(loss.item(), Q)
		base_meter.update(base_correct, Q)
		full_meter.update(full_correct, Q)
		steal_meter.update(steal_rate, Q)
		base_accs_ci.append(torch.tensor([base_correct]))

		batch_time.update(time.time() - end)
		end = time.time()

		if ep % opt.print_freq == 0 and ep != 0:
			msg = ('Test [{0}/{1}]  Time {bt.val:.3f} ({bt.avg:.3f})  '
				   'Loss {loss.val:.3f} ({loss.avg:.3f})  '
				   'Base {base.val:.2f} ({base.avg:.2f})  '
				   'Full {full.val:.2f} ({full.avg:.2f})  '
				   'Steal {steal.val:.2f} ({steal.avg:.2f})'.format(
					ep, len(test_loader), bt=batch_time, loss=losses,
					base=base_meter, full=full_meter, steal=steal_meter))
			print(msg)
			print(msg, file=F_txt)

	geom_pct = (100.0 * geom_pass_total / geom_total) if geom_total > 0 else float('nan')
	summary = (' * Base {base.avg:.3f}  Full {full.avg:.3f}  '
			   'Steal {steal.avg:.3f}  Geom {geom:.3f}% ({gp}/{gt})'.format(
				base=base_meter, full=full_meter, steal=steal_meter,
				geom=geom_pct, gp=geom_pass_total, gt=geom_total))
	print(summary)
	print(summary, file=F_txt)

	return base_meter.avg, full_meter.avg, steal_meter.avg, geom_pct, losses.avg, base_accs_ci


if __name__ == '__main__':

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
	# classifier_module was instantiated with default base_per_super; override here.
	model.classifier.base_per_super = opt.base_per_super

	criterion = nn.CrossEntropyLoss().to(device)

	# Output / log file (alongside the checkpoint)
	opt.outf = opt.resume if str(opt.resume).endswith('/') else opt.resume + '/'
	if not os.path.exists(opt.outf):
		os.makedirs(opt.outf)
	F_txt = open(os.path.join(opt.outf, 'Test_SuperClass_results.txt'), 'a+')
	print('==================== SuperClass Test ====================')
	print('==================== SuperClass Test ====================', file=F_txt)
	print(opt)
	print(opt, file=F_txt)

	# Load the trained best model (encoder weights are what we care about).
	best_path = os.path.join(opt.resume, 'model_best.pth.tar')
	checkpoint = utils.get_resume_file(best_path, F_txt)
	if checkpoint is not None:
		# The trained classifier head (DN4) has no parameters, so missing/extra
		# keys between DN4 and DN4_SuperClass heads are benign. Use strict=False
		# to tolerate the head-name difference.
		missing, unexpected = model.load_state_dict(
			{k: v.to(device) for k, v in checkpoint['model'].items()}, strict=False)
		print('missing keys   :', missing)
		print('unexpected keys:', unexpected)
		print('missing keys   :', missing, file=F_txt)
		print('unexpected keys:', unexpected, file=F_txt)
	else:
		print('WARNING: no checkpoint found; running with randomly initialized encoder.')
		print('WARNING: no checkpoint found; running with randomly initialized encoder.', file=F_txt)

	total_base, total_full, total_steal, total_geom = 0.0, 0.0, 0.0, 0.0
	ci_h = np.zeros(opt.repeat_num)
	for r in range(opt.repeat_num):
		print('\n---- Round %d ----' % r)
		print('\n---- Round %d ----' % r, file=F_txt)

		test_loader = FewShotDataloader.get_SuperClass_dataloader(opt, mode=opt.mode)

		with torch.no_grad():
			base_avg, full_avg, steal_avg, geom_pct, _, base_accs = test_superclass(
				test_loader, model, criterion, F_txt)

		mean_acc, h = utils.mean_confidence_interval(base_accs)
		ci_h[r] = h
		total_base += base_avg
		total_full += full_avg
		total_steal += steal_avg
		total_geom += geom_pct

		line = 'Round %d  base=%.3f  full=%.3f  steal=%.3f  geom=%.3f%%  CI_h=%.3f' % (
			r, base_avg, full_avg, steal_avg, geom_pct, h)
		print(line)
		print(line, file=F_txt)

	R = opt.repeat_num
	final = ('\nMean over %d rounds:  base=%.3f  full=%.3f  steal=%.3f  geom=%.3f%%  CI_h=%.3f'
			 % (R, total_base / R, total_full / R, total_steal / R, total_geom / R, ci_h.mean()))
	print(final)
	print(final, file=F_txt)
	F_txt.close()
