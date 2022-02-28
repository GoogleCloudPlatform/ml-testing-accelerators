#!/bin/bash

jsonnet tests/list_tests.jsonnet -J . > all_tests.json

python3 scripts/list-tests.py

rm all_tests.json
