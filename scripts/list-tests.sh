#!/bin/bash

set -u

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--filter)
      filter="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option '$1'. Usage: scripts/list-tests.sh [--filter \$FILTER]"
      exit 1
      ;;
  esac
done

set -x

jsonnet -J . -S tests/list_tests.jsonnet --tla-str filter=$filter
