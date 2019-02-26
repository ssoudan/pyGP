#!/usr/bin/env bash


set -e

if [ "${PROJECT_ID}" = "" ]; then
		echo "[E] set PROJECT_ID"
		exit 1
fi

cmd=${1:-run_gpflow.py}
c=${cmd/_}
container=${c/.py/}

if [ ! -e "src/main/python/$cmd" ]; then
		echo "[E] invalid command"
		exit 1
fi

docker build --build-arg cmd=${cmd} -t pygp .

docker tag pygp gcr.io/${PROJECT_ID}/pygp/${container}
docker push gcr.io/${PROJECT_ID}/pygp/${container}:latest


