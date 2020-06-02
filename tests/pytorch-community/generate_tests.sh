#!/bin/bash
for file in ./*; do
  filename="${file##*/}"
  if [[ "$filename" != "base.libsonnet" ]] && [[ "$filename" == *"libsonnet"* ]]; then
    echo ../../k8s/gen/${filename/libsonnet/yaml}
    jsonnet -J . $filename > ../../k8s/gen/${filename/libsonnet/yaml}
    echo $filename
  fi
done
