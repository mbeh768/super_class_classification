#!/usr/bin/env python3
"""Cross-platform conda environment setup for this repo.

Creates (or reuses) a conda environment with the right Python version and
installs requirements.txt into it. Pure Python + subprocess calls to
`conda`/`conda run` -- no shell script, so it runs the same way on Linux,
macOS, and Windows (as long as conda/Miniconda is already installed and on
PATH; this script does not install conda itself).

Usage:
    python setup_env.py
    python setup_env.py --name my_env --python 3.10.18
    python setup_env.py --dry-run
"""
from __future__ import print_function
import argparse
import platform
import shutil
import subprocess
import sys


def find_conda():
    for exe in ('conda', 'mamba'):
        path = shutil.which(exe)
        if path:
            return exe
    return None


def run(cmd, dry_run=False):
    print('> ' + ' '.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def env_exists(conda, name):
    result = subprocess.run([conda, 'env', 'list', '--json'], capture_output=True, text=True, check=True)
    import json
    envs = json.loads(result.stdout).get('envs', [])
    # env paths end in .../envs/<name> (or are the base env itself)
    return any(e.rstrip('/\\').endswith(('/' + name, '\\' + name)) for e in envs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='super_class', help='conda environment name')
    ap.add_argument('--python', default='3.10.18', help='Python version to create the env with')
    ap.add_argument('--requirements', default='requirements.txt', help='requirements file to install')
    ap.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    args = ap.parse_args(argv)

    conda = find_conda()
    if conda is None:
        print('ERROR: conda (or mamba) not found on PATH.')
        print('Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html')
        return 1

    print(f'Using "{conda}" on {platform.system()} {platform.machine()}')

    exists = env_exists(conda, args.name)  # read-only check, safe even for --dry-run
    if exists:
        print(f'Conda environment "{args.name}" already exists -- reusing it.')
    else:
        run([conda, 'create', '-y', '-n', args.name, f'python={args.python}'], args.dry_run)

    # `conda run` executes inside the target env regardless of OS/shell
    # activation semantics -- avoids "conda activate" needing `conda init`
    # to have been run in the calling shell first.
    run([conda, 'run', '-n', args.name, 'python', '-m', 'pip', 'install', '-r', args.requirements], args.dry_run)

    print()
    print('Setup complete. Activate with:')
    print(f'    conda activate {args.name}')
    print('Then run:')
    print('    python demo_places365.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
