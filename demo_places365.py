#!/usr/bin/env python3
"""Quick Places365 demo entrypoint.

This wraps the zero-shot super-class evaluator with faster defaults so the
repo has a simple, employer-friendly command:

    python demo_places365.py

Runs the 6 super-class triplets discovered by the combinatorial screen in
dataset/Places365/discovered_supers.csv (see research/screen_allpairs.py
and reports/gen_places_allpairs_combined_pdf.py for how they were found), at
--super_alpha 0.15 -- the spread-penalty value found by sweeping
research/sweep_alpha.py against this triplet set, which roughly balances
constituent vs. super-class accuracy instead of the class default (1.0)
that favors constituents.

To run against your own triplets, pass --csv_path pointing at a CSV with a
base1,base2,super header (see prepare_places365_subset.py to build a larger
image subset first). Override any Places365 path with environment
variables or flags accepted by Test_Super_Places.py.
"""

from Test_Super_Places import main


if __name__ == '__main__':
	main([
		'--encoder_model', 'DINOv2_Local',
		'--classifier_model', 'DN4_SuperClass',
		'--imageSize', '224',
		'--csv_path', './dataset/Places365/discovered_supers.csv',
		'--super_alpha', '0.15',
		'--query_num', '5',
		'--episode_test_num', '10',
	])
