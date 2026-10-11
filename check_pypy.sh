#!/usr/bin/env bash
# Check that the PyPy environment gives the same results as the normal one,
# and how much faster it is. Takes a few minutes; nothing else should be
# training at the same time, or the timings mean little.
#
#   ./check_pypy.sh
#
# 1. The same games, trajectories and diversity figures under both
#    interpreters (check_interpreters.py).
# 2. The same short steady-state and cellular training runs under both: their
#    history.csv and diversity.csv must be byte-identical.
# Results go to runs/_pypy_check/. Exit status 0 only if everything matches.
set -uo pipefail
cd "$(dirname "$0")"

cpython=${BOARD_GRAPH_CPYTHON_VENV:-GeneticAlgo/.venv}/bin/python
pypy=${BOARD_GRAPH_PYPY_VENV:-GeneticAlgo/.venv-pypy}/bin/python
directory=runs/_pypy_check/$(date +%Y%m%d_%H%M%S)
mkdir -p "$directory"
status=0

printf '== Games, trajectories and diversity\n'
"$cpython" check_interpreters.py --output "$directory/cpython.pkl" || status=1
"$pypy" check_interpreters.py --output "$directory/pypy.pkl" || status=1
"$cpython" check_interpreters.py --compare "$directory/cpython.pkl" "$directory/pypy.pkl" || status=1

run() {  # interpreter name flags...
  local interpreter=$1 name=$2; shift 2
  local started=$SECONDS
  "$interpreter" -m GeneticAlgo.train_board_graph --no-video --seed 42 --timestep-limit 1000 \
    --stagnation-penalty 20 --max-initial-depth 8 --max-evolved-depth 20 \
    --crossover-probability 0.7 --mutation-probability 0.3 \
    --output-directory "$directory/$name" "$@" > "$directory/$name.log" 2>&1
  printf '%s: %s s\n' "$name" "$(( SECONDS - started ))"
}
steady=(--evolution-method steady --steady-children 2 --steady-replace-on-tie --population-size 100
        --generations 15 --boards-per-generation 10 --boards-replaced-per-generation 1
        --full-min-initial-depth 4 --grow-terminal-probability 0.4 --save-interval 5)
cellular=(--evolution-method cellular --population-size 100 --generations 4 --boards-per-generation 8
          --boards-replaced-per-generation 2)

printf '\n== Short training runs\n'
run "$cpython" steady_cpython "${steady[@]}"
run "$pypy" steady_pypy "${steady[@]}"
run "$cpython" cellular_cpython "${cellular[@]}"
run "$pypy" cellular_pypy "${cellular[@]}"
for pair in "steady history.csv" "steady diversity.csv" "cellular history.csv"; do
  set -- $pair
  if cmp -s "$directory/$1_cpython/$2" "$directory/$1_pypy/$2"; then
    printf '%s %s: identical\n' "$1" "$2"
  else
    printf '%s %s: DIFFERENT (see %s)\n' "$1" "$2" "$directory"
    status=1
  fi
done
printf '\n%s\n' "$([[ $status == 0 ]] && echo 'All checks passed: PyPy gives the same results.' || echo 'Some checks FAILED: do not train with PyPy until this is understood.')"
exit "$status"
