#!/bin/bash

source /setup.sh

set -u
set -x

"$@"
export STATUS="$?"

source /publish.sh
exit "$STATUS"
