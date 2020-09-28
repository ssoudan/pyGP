#!/usr/bin/env bash

set -e

function run() {
  cmd=$1

  c=${cmd/_}
  container=${c/.py/}

  if [ ! -e "src/main/python/$cmd" ]; then
      echo "[E] invalid command: $cmd"
      exit 1
  fi

   docker run -ti -v `pwd`/output:/app/output/ pygp-${container}
}

#
# main-like
#
cmd=${1:-all}

if [ "all" != "$cmd" ];
then

    run $cmd

else

  for cmd in run_gpflow.py run_bo.py run_sklearn.py run_tfp.py
  do

    run $cmd

  done
fi

echo "[I] OK"
