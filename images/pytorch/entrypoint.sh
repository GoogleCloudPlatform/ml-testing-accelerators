#!/bin/bash
set -e

source /setup.sh

# Trim grpc:// prefix
export XRT_TPU_CONFIG="tpu_worker;0;${KUBE_GOOGLE_CLOUD_TPU_ENDPOINTS:7}"

set -u

docker-entrypoint.sh "$@"
export STATUS="$?"

source /publish.sh
exit "$STATUS"
