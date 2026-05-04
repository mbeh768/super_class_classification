#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill in miniImageNet classes by streaming per-synset tarballs from
image-net.org, one class at a time. Only the synsets listed in the
train/val/test CSVs (and that are currently under target on disk) are
downloaded — no 1.28M-image ImageNet stream.

Auth: same image-net.org browser cookie as Add_Tiger.py. Pass via
--cookie or --cookie_file (e.g. ~/.imagenet_cookie).

Layout produced / extended:
  dataset/miniImageNet/images/<synset><8-digit-counter>.jpg
  dataset/miniImageNet/<split>.csv        (rows appended idempotently)

Idempotent: skips filenames already on disk, never duplicates CSV rows.

Usage:
  python Add_MiniImageNet.py --cookie_file ~/.imagenet_cookie
  python Add_MiniImageNet.py --cookie_file ~/.imagenet_cookie --splits train val
  python Add_MiniImageNet.py --cookie_file ~/.imagenet_cookie --synsets n01532829 n01558993
"""

from __future__ import print_function
import argparse
import csv
import io
import os
import sys
import tarfile
import time
from collections import defaultdict


URL_TEMPLATE = 'https://image-net.org/data/winter21_whole/{0}.tar'


def read_csv_synsets(csv_path):
	synsets = set()
	if not os.path.exists(csv_path):
		return synsets
	with open(csv_path, newline='') as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			if row and len(row) >= 2:
				synsets.add(row[1])
	return synsets


def read_csv_filenames(csv_path):
	seen = set()
	if not os.path.exists(csv_path):
		return seen
	with open(csv_path, newline='') as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			if row:
				seen.add(row[0])
	return seen


def count_on_disk(images_dir, synset):
	return sum(1 for f in os.listdir(images_dir) if f.startswith(synset))


def download_synset(synset, cookie, target_count, images_dir,
					resize_short, jpeg_quality, timeout):
	'''Stream one synset tarball, save up to target_count images.
	Returns list of saved filenames.'''
	import requests
	from PIL import Image

	url = URL_TEMPLATE.format(synset)
	headers = {'Cookie': cookie, 'User-Agent': 'Mozilla/5.0'}

	existing = {f for f in os.listdir(images_dir) if f.startswith(synset)}
	saved = []
	idx = 1
	skipped = 0

	resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
	if resp.status_code != 200:
		print('  HTTP %d for %s; body starts: %r'
			  % (resp.status_code, synset, resp.text[:200]))
		return saved
	ctype = resp.headers.get('Content-Type', '')
	if 'html' in ctype.lower():
		print('  ERROR: got HTML for %s (cookie invalid/expired?)' % synset)
		return saved

	try:
		with tarfile.open(fileobj=resp.raw, mode='r|') as tar:
			for m in tar:
				if len(existing) + len(saved) >= target_count:
					break
				if not m.isfile():
					continue
				if not m.name.lower().endswith(('.jpeg', '.jpg')):
					continue

				while True:
					fname = '%s%08d.jpg' % (synset, idx)
					if fname not in existing and fname not in saved:
						break
					idx += 1

				fobj = tar.extractfile(m)
				if fobj is None:
					continue
				data = fobj.read()

				try:
					img = Image.open(io.BytesIO(data))
					if img.mode != 'RGB':
						img = img.convert('RGB')
					if resize_short > 0:
						w, h = img.size
						if min(w, h) != resize_short:
							if w < h:
								nw = resize_short
								nh = int(round(h * resize_short / float(w)))
							else:
								nh = resize_short
								nw = int(round(w * resize_short / float(h)))
							img = img.resize((nw, nh), Image.BICUBIC)
					out_path = os.path.join(images_dir, fname)
					img.save(out_path, format='JPEG', quality=jpeg_quality)
					saved.append(fname)
					idx += 1
				except Exception as e:
					skipped += 1
					if skipped < 5:
						print('  decode-skip %s: %s' % (m.name, e))
	except tarfile.ReadError as e:
		print('  tar read error for %s: %s' % (synset, e))
	finally:
		resp.close()

	if skipped:
		print('  %d image(s) skipped (decode failed)' % skipped)
	return saved


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--cookie', default=None)
	ap.add_argument('--cookie_file', default=None)
	ap.add_argument('--miniimagenet_dir', default='./dataset/miniImageNet')
	ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
	ap.add_argument('--synsets', nargs='+', default=None,
					help='explicit synset list; if set, --splits is ignored')
	ap.add_argument('--target_per_class', type=int, default=600)
	ap.add_argument('--resize_short', type=int, default=0,
					help='if >0, resize so shorter side == this value')
	ap.add_argument('--jpeg_quality', type=int, default=95)
	ap.add_argument('--timeout', type=int, default=120)
	ap.add_argument('--append_csv', action='store_true', default=True)
	args = ap.parse_args()

	cookie = args.cookie
	if cookie is None and args.cookie_file:
		with open(os.path.expanduser(args.cookie_file)) as f:
			cookie = f.read().strip()
	if not cookie:
		print('ERROR: need --cookie or --cookie_file (see Add_Tiger.py header '
			  'for how to grab it from browser devtools).')
		sys.exit(1)

	images_dir = os.path.join(args.miniimagenet_dir, 'images')
	os.makedirs(images_dir, exist_ok=True)

	# Map each synset to the split it belongs to.
	synset_to_split = {}
	if args.synsets:
		for s in args.synsets:
			synset_to_split[s] = 'test'  # default; not used for CSV append unless found
		# Resolve split from CSVs if possible.
		for split in ['train', 'val', 'test']:
			for s in read_csv_synsets(
					os.path.join(args.miniimagenet_dir, split + '.csv')):
				if s in synset_to_split:
					synset_to_split[s] = split
	else:
		for split in args.splits:
			for s in read_csv_synsets(
					os.path.join(args.miniimagenet_dir, split + '.csv')):
				synset_to_split.setdefault(s, split)

	if not synset_to_split:
		print('no synsets found. Did you point --miniimagenet_dir at the right folder?')
		sys.exit(1)

	on_disk = {s: count_on_disk(images_dir, s) for s in synset_to_split}
	needed = [s for s in synset_to_split if on_disk[s] < args.target_per_class]
	needed.sort()

	print('synsets discovered   : %d' % len(synset_to_split))
	print('at or above target   : %d' % (len(synset_to_split) - len(needed)))
	print('to download          : %d' % len(needed))
	if not needed:
		print('nothing to do')
		return
	for s in needed:
		print('  %s  on_disk=%4d  need=%4d  (split=%s)'
			  % (s, on_disk[s], args.target_per_class - on_disk[s],
				 synset_to_split[s]))
	print('')

	existing_csv = {split: read_csv_filenames(
			os.path.join(args.miniimagenet_dir, split + '.csv'))
		for split in ['train', 'val', 'test']}
	rows_to_append = defaultdict(list)

	start = time.time()
	for i, synset in enumerate(needed, 1):
		target_count = args.target_per_class
		print('[%d/%d] %s  (on_disk=%d, target=%d)'
			  % (i, len(needed), synset, on_disk[synset], target_count))
		saved = download_synset(
			synset, cookie, target_count, images_dir,
			args.resize_short, args.jpeg_quality, args.timeout)
		on_disk[synset] = count_on_disk(images_dir, synset)
		split = synset_to_split[synset]
		for fn in saved:
			if fn not in existing_csv[split]:
				rows_to_append[split].append((fn, synset))
		elapsed = time.time() - start
		print('  saved %d this round  |  %d on disk  |  elapsed %.0fs'
			  % (len(saved), on_disk[synset], elapsed))

	if args.append_csv:
		for split, rows in rows_to_append.items():
			if not rows:
				continue
			csv_path = os.path.join(args.miniimagenet_dir, split + '.csv')
			header_needed = (not os.path.exists(csv_path)
							 or os.path.getsize(csv_path) == 0)
			with open(csv_path, 'a', newline='') as fh:
				w = csv.writer(fh)
				if header_needed:
					w.writerow(['filename', 'label'])
				for fn, syn in rows:
					w.writerow([fn, syn])
			print('appended %d rows to %s' % (len(rows), csv_path))

	print('\n==== final on-disk counts ====')
	short = []
	for s in sorted(synset_to_split):
		c = count_on_disk(images_dir, s)
		mark = '' if c >= args.target_per_class else '  <-- short'
		if c < args.target_per_class:
			short.append((s, c))
		print('  %s  %d%s' % (s, c, mark))
	if short:
		print('\n%d class(es) still below target:' % len(short))
		for s, c in short:
			print('  %s  %d' % (s, c))


if __name__ == '__main__':
	main()
