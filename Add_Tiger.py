#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install the ImageNet "tiger" class (synset n02129604, class idx 292) into the
miniImageNet layout expected by this repo:

  dataset/miniImageNet/images/n02129604NNNNNNNN.jpg   (N = 8-digit zero-padded)
  dataset/miniImageNet/test.csv                        (600 rows appended)

Source: HuggingFace `evanarlian/imagenet_1k_resized_256` (non-gated mirror,
images pre-resized to 256px on the shorter side). Streaming mode - we pull
only tiger samples, not the whole 1.28M-image archive.

Idempotent: re-running skips filenames that already exist on disk, and will
only append CSV rows for filenames not already in test.csv.
"""

from __future__ import print_function
import argparse
import csv
import os
import sys


SYNSET = 'n02129604'
IMAGENET_TIGER_LABEL = 292  # ImageNet-1k class index for "tiger, Panthera tigris"
DEFAULT_DATASET = 'evanarlian/imagenet_1k_resized_256'


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--miniimagenet_dir', default='./dataset/miniImageNet')
	ap.add_argument('--hf_dataset', default=DEFAULT_DATASET)
	ap.add_argument('--split', default='train',
					help='HF split to pull from (train has ~1300 tigers, val ~50)')
	ap.add_argument('--num_images', type=int, default=600,
					help='target count, matches miniImageNet per-class convention')
	ap.add_argument('--csv_name', default='test.csv',
					help='which split CSV to append to (lion is already in test.csv)')
	args = ap.parse_args()

	images_dir = os.path.join(args.miniimagenet_dir, 'images')
	csv_path = os.path.join(args.miniimagenet_dir, args.csv_name)
	os.makedirs(images_dir, exist_ok=True)

	# Read existing CSV to compute (a) header presence, (b) already-listed filenames.
	existing_rows = []
	if os.path.exists(csv_path):
		with open(csv_path, newline='') as f:
			existing_rows = list(csv.reader(f))
	if not existing_rows or existing_rows[0] != ['filename', 'label']:
		print('WARNING: {0} does not start with the expected header; not modifying.'
			  .format(csv_path))
		if not existing_rows:
			existing_rows = [['filename', 'label']]
		else:
			sys.exit(1)

	already_in_csv = {r[0] for r in existing_rows[1:] if r}
	existing_on_disk = {f for f in os.listdir(images_dir) if f.startswith(SYNSET)}
	print('existing tiger files on disk :', len(existing_on_disk))
	print('existing tiger rows in CSV   :', sum(1 for r in existing_rows[1:] if r and r[1] == SYNSET))

	saved = 0
	target = args.num_images
	new_filenames = []

	if len(existing_on_disk) >= target:
		print('already at target ({0} on disk >= {1}), skipping download'
			  .format(len(existing_on_disk), target))
	else:
		# Stream only tiger samples.
		from datasets import load_dataset
		print('streaming {0} split={1}...'.format(args.hf_dataset, args.split))
		ds = load_dataset(args.hf_dataset, split=args.split, streaming=True)
		ds = ds.filter(lambda ex: ex['label'] == IMAGENET_TIGER_LABEL)

		idx = 1  # next 8-digit counter (we always start at 1; skip if file present)

		for example in ds:
			if saved + len(existing_on_disk) >= target:
				break

			# Find the next available slot on disk.
			while True:
				fname = '{0}{1:08d}.jpg'.format(SYNSET, idx)
				if fname not in existing_on_disk:
					break
				idx += 1

			img = example['image']
			if img.mode != 'RGB':
				img = img.convert('RGB')
			out_path = os.path.join(images_dir, fname)
			img.save(out_path, format='JPEG', quality=95)

			existing_on_disk.add(fname)
			new_filenames.append(fname)
			saved += 1
			idx += 1

			if saved % 50 == 0:
				print('  saved {0}/{1}'.format(saved, target))

	print('new tiger images saved       :', saved)

	# Append CSV rows for any filename on disk but not yet in the CSV.
	rows_to_append = []
	for f in sorted(os.listdir(images_dir)):
		if f.startswith(SYNSET) and f not in already_in_csv:
			rows_to_append.append([f, SYNSET])

	if rows_to_append:
		with open(csv_path, 'a', newline='') as fh:
			w = csv.writer(fh)
			for row in rows_to_append:
				w.writerow(row)
		print('rows appended to {0}: {1}'.format(csv_path, len(rows_to_append)))
	else:
		print('no new CSV rows to append')

	# Report final state.
	final_on_disk = sum(1 for f in os.listdir(images_dir) if f.startswith(SYNSET))
	with open(csv_path, newline='') as f:
		final_in_csv = sum(1 for r in csv.reader(f) if r and r[1] == SYNSET)
	print('final tiger files on disk    :', final_on_disk)
	print('final tiger rows in CSV      :', final_in_csv)


if __name__ == '__main__':
	main()