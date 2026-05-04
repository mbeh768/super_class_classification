#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild miniImageNet CSV split files to list only images that actually exist
on disk, preserving each synset's original split membership.

The original miniImageNet CSVs reference ImageNet-internal image IDs (e.g.
n0221948600001221.jpg), but our download scripts name images with sequential
counters. After a fresh download, CSV entries point at nonexistent files and
the dataloader crashes mid-episode. Running this script realigns the CSVs to
whatever is actually on disk.

Safe: existing CSVs are backed up to <split>.csv.bak (only on first run) unless
--no_backup is passed. Re-running is idempotent.

Usage:
  python Sync_CSV_to_Disk.py                     # rewrite train/val/test.csv
  python Sync_CSV_to_Disk.py --dry_run           # show what would change
  python Sync_CSV_to_Disk.py --splits test       # only rebuild one split
"""

from __future__ import print_function
import argparse
import csv
import os
import sys
from collections import defaultdict


def read_csv_synsets(csv_path):
	synsets = set()
	if not os.path.exists(csv_path):
		return synsets
	with open(csv_path, newline='') as f:
		r = csv.reader(f)
		next(r, None)  # header
		for row in r:
			if row and len(row) >= 2:
				synsets.add(row[1])
	return synsets


def count_csv_rows(csv_path):
	if not os.path.exists(csv_path):
		return 0
	with open(csv_path, newline='') as f:
		r = csv.reader(f)
		next(r, None)
		return sum(1 for row in r if row)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--miniimagenet_dir', default='./dataset/miniImageNet')
	ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
	ap.add_argument('--dry_run', action='store_true')
	ap.add_argument('--no_backup', action='store_true')
	args = ap.parse_args()

	images_dir = os.path.join(args.miniimagenet_dir, 'images')
	if not os.path.isdir(images_dir):
		print('ERROR: images dir not found: %s' % images_dir)
		sys.exit(1)

	# Which synsets belong to which split (from existing CSVs).
	synset_to_split = {}
	for split in args.splits:
		csv_path = os.path.join(args.miniimagenet_dir, split + '.csv')
		for s in read_csv_synsets(csv_path):
			synset_to_split.setdefault(s, split)

	if not synset_to_split:
		print('no synsets found in CSVs for splits %s' % args.splits)
		sys.exit(1)

	# Group on-disk files by synset prefix (n + 8 digits).
	by_synset = defaultdict(list)
	for f in sorted(os.listdir(images_dir)):
		if not f.lower().endswith(('.jpg', '.jpeg')):
			continue
		if len(f) < 9 or not f.startswith('n'):
			continue
		synset = f[:9]
		if not synset[1:].isdigit():
			continue
		by_synset[synset].append(f)

	# Build new rows per split.
	new_rows = defaultdict(list)
	orphan_synsets = []
	for synset, files in by_synset.items():
		split = synset_to_split.get(synset)
		if split is None:
			orphan_synsets.append((synset, len(files)))
			continue
		for f in files:
			new_rows[split].append((f, synset))

	# Also report synsets that exist in a CSV but have zero files.
	missing_on_disk = []
	for synset, split in synset_to_split.items():
		if synset not in by_synset:
			missing_on_disk.append((synset, split))

	# Summary.
	print('==== planned changes ====')
	for split in args.splits:
		csv_path = os.path.join(args.miniimagenet_dir, split + '.csv')
		old = count_csv_rows(csv_path)
		new = len(new_rows[split])
		print('  %s.csv:  %d -> %d rows' % (split, old, new))

	if orphan_synsets:
		print('\non-disk synsets with no CSV entry (ignored):')
		for s, n in orphan_synsets:
			print('  %s  %d files' % (s, n))

	if missing_on_disk:
		print('\nsynsets in CSV with zero on-disk files (dropped):')
		for s, split in missing_on_disk:
			print('  %s  (split=%s)' % (s, split))

	if args.dry_run:
		print('\n--dry_run: no files written')
		return

	# Write out each CSV, backing up first if --no_backup not set.
	for split in args.splits:
		csv_path = os.path.join(args.miniimagenet_dir, split + '.csv')
		if not args.no_backup and os.path.exists(csv_path):
			backup_path = csv_path + '.bak'
			if not os.path.exists(backup_path):
				os.replace(csv_path, backup_path)
				print('\nbacked up %s -> %s' % (csv_path, backup_path))
			else:
				# Backup already exists; don't clobber it.
				os.remove(csv_path)

		with open(csv_path, 'w', newline='') as fh:
			w = csv.writer(fh)
			w.writerow(['filename', 'label'])
			for fn, syn in sorted(new_rows[split]):
				w.writerow([fn, syn])
		print('wrote %s (%d rows)' % (csv_path, len(new_rows[split])))


if __name__ == '__main__':
	main()
