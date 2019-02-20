#!/bin/sh

set -e

echo "----------LINT-------------"
pylama -l "pycodestyle,mccabe"
echo "----------OK---------------"
