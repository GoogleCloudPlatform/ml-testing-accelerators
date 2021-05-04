#!/bin/bash
shopt -s globstar

set -x
jsonnetfmt -i -- {tests,templates}/**/*.*sonnet
