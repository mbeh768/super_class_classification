#!/usr/bin/env python3
"""Build a small Places365 subset for the demo.

This script takes a source Places365 directory that already contains per-class
folders and copies or symlinks only the classes referenced by the demo
triplets. It also writes a filtered CSV with the triplets that were kept.

Example:
    python prepare_places365_subset.py \
        --source_dir /data/Places365 \
        --output_dir ./dataset/Places365 \
        --mode symlink

If you already have a custom triplet CSV, pass it with --csv_path. Otherwise,
the built-in demo triplets from the employer-facing results are used.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


DEFAULT_TRIPLETS = [
	('lake-natural', 'cottage', 'boathouse'),
	('lake-natural', 'house', 'boathouse'),
	('ocean', 'cottage', 'boathouse'),
	('ocean', 'house', 'boathouse'),
	('snowfield', 'mountain', 'mountain_snowy'),
	('snowfield', 'canyon', 'crevasse'),
	('exterior', 'sky', 'skyscraper'),
	('assembly_line', 'car_interior', 'auto_factory'),
	('mountain', 'forest_path', 'mountain_path'),
	('building_facade', 'forest_road', 'street'),
	('hospital_room', 'clean_room', 'operating_room'),
]


def build_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source_dir', required=True,
		help='root of an extracted Places365 dataset with per-class folders')
	parser.add_argument('--output_dir', default='./dataset/Places365',
		help='destination directory for the trimmed dataset')
	parser.add_argument('--images_per_class', type=int, default=10,
		help='number of images to keep from each class')
	parser.add_argument('--csv_path', default=None,
		help='optional triplet CSV; defaults to the built-in demo triplets')
	parser.add_argument('--mode', choices=['copy', 'symlink'], default='copy',
		help='copy files or create symlinks into the subset directory')
	parser.add_argument('--overwrite', action='store_true',
		help='replace the destination directory if it already exists')
	return parser


def read_triplets(csv_path):
	triplets = []
	with open(csv_path, newline='') as handle:
		reader = csv.reader(handle)
		for row in reader:
			if not row or len(row) < 3:
				continue
			base1, base2, super_class = (cell.strip() for cell in row[:3])
			if base1.lower() == 'base1' or base1.startswith('#'):
				continue
			triplets.append((base1, base2, super_class))
	return triplets


def collect_classes(triplets):
	classes = []
	seen = set()
	for base1, base2, super_class in triplets:
		for class_name in (base1, base2, super_class):
			if class_name not in seen:
				seen.add(class_name)
				classes.append(class_name)
	return classes


def resolve_class_dir(source_dir, class_name):
	source_root = Path(source_dir)
	candidates = [source_root / class_name]
	if source_root.is_dir():
		for child in source_root.iterdir():
			if child.is_dir():
				candidates.append(child / class_name)

	for candidate in candidates:
		if candidate.is_dir():
			return candidate

	return None


def list_images(class_dir):
	image_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
	return sorted(
		path for path in class_dir.iterdir()
		if path.is_file() and path.suffix.lower() in image_suffixes
	)


def materialize_class(source_dir, output_dir, class_name, mode, images_per_class):
	source_class_dir = resolve_class_dir(source_dir, class_name)
	if source_class_dir is None:
		return False
	image_paths = list_images(source_class_dir)
	if not image_paths:
		return False
	n = max(0, images_per_class)
	if n <= 0:
		raise RuntimeError(f'images_per_class must be > 0 for {class_name}')
	# choose up to `n` random images from the class (without replacement)
	selected_images = random.sample(image_paths, min(n, len(image_paths)))

	dest_class_dir = Path(output_dir) / class_name
	if dest_class_dir.exists() or dest_class_dir.is_symlink():
		if dest_class_dir.is_dir() and not dest_class_dir.is_symlink():
			shutil.rmtree(dest_class_dir)
		else:
			dest_class_dir.unlink()
	dest_class_dir.mkdir(parents=True, exist_ok=True)

	for image_path in selected_images:
		dest_image_path = dest_class_dir / image_path.name
		if mode == 'copy':
			shutil.copy2(image_path, dest_image_path)
		else:
			os.symlink(image_path.resolve(), dest_image_path)

	return True


def write_subset_csv(output_dir, triplets):
	csv_path = Path(output_dir) / 'potential_supers.csv'
	with open(csv_path, 'w', newline='') as handle:
		writer = csv.writer(handle)
		writer.writerow(['base1', 'base2', 'super'])
		writer.writerows(triplets)
	return csv_path


def main():
	parser = build_parser()
	opt = parser.parse_args()

	if opt.csv_path:
		triplets = read_triplets(opt.csv_path)
	else:
		triplets = list(DEFAULT_TRIPLETS)

	if not triplets:
		raise RuntimeError('no triplets found')

	output_dir = Path(opt.output_dir)
	if output_dir.exists():
		if not opt.overwrite:
			raise FileExistsError(
				f'{output_dir} already exists; pass --overwrite to replace it')
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	classes = collect_classes(triplets)
	valid_triplets = []
	missing_triplets = []
	available_classes = {}
	for base1, base2, super_class in triplets:
		triplet_classes = (base1, base2, super_class)
		resolved = []
		missing = []
		for class_name in triplet_classes:
			if class_name not in available_classes:
				available_classes[class_name] = resolve_class_dir(opt.source_dir, class_name)
			if available_classes[class_name] is None:
				missing.append(class_name)
			else:
				resolved.append(class_name)
		if missing:
			missing_triplets.append((base1, base2, super_class, missing))
			print(f'skipping triplet {(base1, base2, super_class)}; missing folders: {missing}')
			continue
		valid_triplets.append((base1, base2, super_class))

	classes = collect_classes(valid_triplets)
	for class_name in classes:
		ok = materialize_class(opt.source_dir, output_dir, class_name, opt.mode, opt.images_per_class)
		if not ok:
			print(f'skipping class {class_name}; no images found or class missing')

	if not valid_triplets:
		raise RuntimeError('no valid triplets found after filtering missing folders')

	csv_path = write_subset_csv(output_dir, valid_triplets)

	print(f'Created subset at: {output_dir}')
	print(f'Classes imported: {len(classes)}')
	print(f'Images per class : {opt.images_per_class}')
	print(f'Triplets written : {len(valid_triplets)}')
	if missing_triplets:
		print(f'Triplets skipped  : {len(missing_triplets)}')
	print(f'CSV written      : {csv_path}')


if __name__ == '__main__':
	main()