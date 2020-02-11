#!/bin/bash

source /setup.sh

set -u
set -x

source /publish.sh

"$@"
