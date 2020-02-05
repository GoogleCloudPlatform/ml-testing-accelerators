#!/bin/bash
set -e

source /setup.sh

set -u

"$@"
export STATUS="$?"

source /publish.sh
exit "$STATUS"
