#!/bin/bash
dryrun=
region=
test_name=

help()
{
  echo "Utility script to run one shots with XLML test."
  echo "This script will:"
  echo "- connect to the appropriate one shot cluster"
  echo "- run the test"
  echo
  echo "Syntax: ./scripts/run-oneshot.sh [-a|d|f|h|t]"
  echo "-d | --dryrun        Dryrun. If set, then the test does not run and only prints commands."
  echo "-t | --test          The name of the test, e.g. `tf.nightly-resnet-imagenet-func-v2-8`."
  echo "-h | --help          Print this help."
}

validate()
{
  if [ -z "$test_name" ]
  then
    echo "Test name must be provided."
    exit
  fi

  region=$(jsonnet -J . -S tests/get_cluster.jsonnet --tla-str test=$test_name)

  echo "Args:"
  echo "Test name:   " $test_name
  echo "Region:      " $region
  echo
}

set_test_name()
{
  # Splits the test filename by forward slash and
  # constructs the test name. This follows the convention of:
  # tests/{tensorflow|pytorch}/{nightly|r2.3|...}/testname.libsonnet
  IFS=/ read -a split <<< "$filename"
  platform="${split[1]}"
  if grep -q "tensorflow" <<< "$platform"; then
    platform="tf"
  elif grep -q "pytorch" <<< "$platform"; then
    platform="pt"
  else
    platform="flax"
  fi

  if [ -z "$patch" ]
  then
    release="${split[2]}"
  else
    release="${split[2]}.${patch}"
  fi

  name="${split[3]%%.*}"
  test_name="$platform-$release-$name-$testtype-$accelerator"
}

run()
{
  validate

  p=`echo $PWD | sed s/scripts//`
  cd $p

  set -e
  set -x
  cd -

  CLUSTER="xl-ml-test-$region"

  which jsonnet &> /dev/null
  if [ -z "$dryrun" ]
  then
    gcloud container clusters get-credentials $CLUSTER --region $region --project xl-ml-test
    temp_file=$(mktemp)
    jsonnet tests/oneshot.jsonnet -J . -S --tla-str test=$test_name > $temp_file

    job_name=$(kubectl create -f $temp_file -o name)

    echo "GKE job name: ${job_name#job.batch/}"
    kubectl wait --for=condition=ready --timeout=10m pod -l job-name=${job_name#job.batch/}
    kubectl logs -f $job_name
  else
    echo "gcloud container clusters get-credentials $CLUSTER --region $region --project xl-ml-test"
    jsonnet tests/oneshot.jsonnet -J . -S --tla-str test=$test_name
  fi
}

while [ "$1" != "" ]; do
  case $1 in
    -d | --dryrun )         shift
                            dryrun=1
                            ;;
    -h | --help )           help
                            exit
                            ;;
    -t | --test)            shift
                            test_name="$1"
                            ;;
    * )                     echo "Invalid option: $1"
                            help
                            exit 1
  esac
  shift
done

run
