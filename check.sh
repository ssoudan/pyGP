#!/bin/sh

set -e

echo "----------TODO-------------"
grep -r 'TODO(' * | grep -v grep | sed 's/.*TODO/TODO/'

echo "----------LINT-------------"
pylama -l "pycodestyle,mccabe"
echo "----------OK---------------"
