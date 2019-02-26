#!/usr/bin/env bash


set -e

if [ "${PROJECT_ID}" = "" ]; then
		echo "[E] set PROJECT_ID"
		exit 1
fi

cmd=${1:-run_gpflow.py}
c=${cmd/_}
container=${c/.py/}

#if [ ! -e "src/main/python/$cmd" ]; then
#		echo "[E] invalid command"
#		exit 1
#fi

docker pull gcr.io/${PROJECT_ID}/pygp/${container}:latest

docker run -ti -v `pwd`/output:/app/output/ gcr.io/${PROJECT_ID}/pygp/${container}:latest

