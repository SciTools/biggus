#!/usr/bin/env bash

wget https://raw.githubusercontent.com/pelson/Obvious-CI/v0.1.0/bootstrap-obvious-ci-and-miniconda.py

PY_VERSION=$(python -c "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))")

python bootstrap-obvious-ci-and-miniconda.py ~/miniconda x64 ${PY_VERSION:0:1} --without-obvci
source ~/miniconda/bin/activate root

conda install --yes -n root python=${PY_VERSION} conda conda-build binstar jinja2 setuptools
conda config --set show_channel_urls True
