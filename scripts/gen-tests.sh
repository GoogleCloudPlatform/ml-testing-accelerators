#!/bin/bash

p=`echo $PWD | sed s/scripts//`
cd $p
set -e
set -x
which jsonnet &> /dev/null
rm -f k8s/*/gen/*.yaml
jsonnet -S -J . tests/cronjobs.jsonnet -m k8s/
cd -
