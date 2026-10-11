#!/usr/bin/env bash
# Build FastCore/_pacman_core*.so for the Python that will use it.
#
#   FastCore/build.sh                 uses GeneticAlgo/.venv/bin/python
#   PYTHON=python3 FastCore/build.sh  uses another interpreter
#
# Needs g++ (C++17) and pybind11 in that Python (pip install pybind11).
set -euo pipefail
cd "$(dirname "$0")"
python=${PYTHON:-../GeneticAlgo/.venv/bin/python}
if ! "$python" -c "import pybind11" 2>/dev/null; then
  echo "pybind11 is not installed for $python; run: $python -m pip install pybind11" >&2
  exit 1
fi
suffix=$("$python" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# Asked for one at a time and quoted: the project path contains spaces.
python_include=$("$python" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
pybind11_include=$("$python" -c "import pybind11; print(pybind11.get_include())")
if [[ ! -f "$python_include/Python.h" ]]; then
  version=$("$python" -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
  echo "Python headers not found in $python_include; on Ubuntu: sudo apt install python$version-dev" >&2
  exit 1
fi
# -ffp-contract=off keeps a*b+c as two roundings, as in Python.
g++ -O3 -march=native -ffp-contract=off -shared -std=c++17 -fPIC -pthread \
  -I"$python_include" -I"$pybind11_include" pacman_core.cpp -o "_pacman_core$suffix"
echo "Built FastCore/_pacman_core$suffix"
