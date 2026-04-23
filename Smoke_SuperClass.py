#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for the zero-shot super-class pipeline (no checkpoint required).

Instantiates `Fewshot_model(classifier_model='DN4_SuperClass')`, pulls ONE
episode from `get_SuperClass_dataloader`, runs a forward pass, and asserts
the output is [Q, way_num + 1].
"""

from __future__ import print_function
import argparse
import os
import sys
sys.dont_write_bytecode = True

import torch
import torch.backends.cudnn as cudnn
from PIL import ImageFile

import dataset.general_dataloader as FewShotDataloader
import models.network as FewShotNet


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def build_opt():
	p = argparse.ArgumentParser()
	p.add_argument('--dataset_dir', required=True, help='dataset root (expects test.csv)')
	p.add_argument('--data_name', default='miniImageNet')
	p.add_argument('--encoder_model', default='Conv64F_Local')
	p.add_argument('--classifier_model', default='DN4_SuperClass')
	p.add_argument('--mode', default='test', choices=['train', 'val', 'test'])
	p.add_argument('--workers', type=int, default=2)
	p.add_argument('--imageSize', type=int, default=84)
	p.add_argument('--way_num', type=int, default=5)
	p.add_argument('--shot_num', type=int, default=1)
	p.add_argument('--query_num', type=int, default=15)
	p.add_argument('--neighbor_k', type=int, default=3)
	p.add_argument('--base_per_super', type=int, default=2)
	p.add_argument('--testepisodeSize', type=int, default=1)
	p.add_argument('--episode_train_num', type=int, default=1)
	p.add_argument('--episode_val_num', type=int, default=1)
	p.add_argument('--episode_test_num', type=int, default=1)
	p.add_argument('--cuda', action='store_true', default=True)
	return p.parse_args()


def main():
	opt = build_opt()
	cudnn.benchmark = True

	loader = FewShotDataloader.get_SuperClass_dataloader(opt, mode=opt.mode)
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
	model.eval()

	query_images, query_targets, support_images, support_targets = next(iter(loader))

	input1 = torch.cat(query_images, 0).cuda()
	input2 = torch.cat(support_images, 0).squeeze(0).cuda()
	input2 = input2.contiguous().view(-1, input2.size(2), input2.size(3), input2.size(4))
	target = torch.cat(query_targets, 0).cuda()

	with torch.no_grad():
		output = model(input1, input2)

	expected_Q = opt.way_num * opt.query_num
	expected_cols = opt.way_num + 1

	print('query input shape  :', tuple(input1.shape))
	print('support input shape:', tuple(input2.shape))
	print('output shape       :', tuple(output.shape))
	print('target shape       :', tuple(target.shape))
	print('target min/max     :', int(target.min()), int(target.max()))

	assert output.dim() == 2, 'expected 2-D output'
	assert output.size(0) == expected_Q, \
		'query count mismatch: got %d, expected %d' % (output.size(0), expected_Q)
	assert output.size(1) == expected_cols, \
		'column count mismatch: got %d, expected %d' % (output.size(1), expected_cols)
	assert target.size(0) == expected_Q
	assert int(target.max()) < opt.way_num, \
		'targets should be in [0, way_num); super column is an extra signal, not a label'

	print('OK: DN4_SuperClass forward returns [Q=%d, way_num+1=%d]' % (expected_Q, expected_cols))


if __name__ == '__main__':
	main()
