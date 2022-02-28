#!/bin/bash
accelerator=
dryrun=
filename=
testtype=
test_name=
patch=

help()
{
  echo "Utility script to run one shots with XLML test."
  echo "This script will:"
  echo "- connect to the appropriate one shot cluster"
  echo "- run the test"
  echo
  echo "Syntax: ./scripts/run-oneshot.sh [-a|d|f|h|t]"
  echo "-a | --accelerator   The accelerator type, e.g. v2-8, v3-8, 8xv100, etc."
  echo "-d | --dryrun        Dryrun. If set, then the test does not run and only prints commands."
  echo "-f | --file          The filepath for the test."
  echo "-h | --help          Print this help."
  echo "-t | --type          Test type, e.g. [func|conv|functional|convergence]."
  echo "-p | --patch         Patch number, if applicable."
}

validate()
{
  if [ -z "$accelerator" ]
  then
    echo "Accelerator must be provided."
    exit
  fi
  if [ -z "$filename" ]
  then
    echo "File name must be provided."
    exit
  fi
  if [ -z "$testtype" ]
  then
    echo "Test type must be provided."
    exit
  fi

  case $testtype in
    "functional" | "func" )  testtype="func"
                             ;;
    "convergence" | "conv" ) testtype="conv"
                             ;;
  esac

  region=
  case $accelerator in
    # Pods are in europe-west4. Add in more cases here
    # as necessary.
    "v2-32" | "v3-32" ) region="europe-west4"
                        ;;
    * )                 region="us-central1"
                        ;;
  esac

  echo "Args:"
  echo "Accelerator: " $accelerator
  echo "Filename:    " $filename
  echo "Type:        " $testtype
  echo "Region:      " $region
  echo
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

  set_test_name

  set -e
  set -x
  cd -

  which jsonnet &> /dev/null
  if [ -z "$dryrun" ]
  then
    gcloud container clusters get-credentials oneshots-$region --region $region --project xl-ml-test
    temp_file=$(mktemp)
    jsonnet tests/oneshot.jsonnet -J . -S --tla-str test=$test_name > $temp_file

    job_name=$(kubectl create -f $temp_file -o name)
    pod_name=$(kubectl get pod -l job-name=${job_name#job.batch/} -o name)

    echo "GKE pod name: ${pod_name#pod/}"
    kubectl wait --for=condition=ready --timeout=10m $pod_name
    kubectl logs -f $pod_name --container=train
  else
    echo "gcloud container clusters get-credentials oneshots-$region --region $region --project xl-ml-test"
    jsonnet tests/oneshot.jsonnet -J . -S --tla-str test=$test_name
  fi
}

while [ "$1" != "" ]; do
  case $1 in
    -a | --accelerator )    shift
                            accelerator="$1"
                            ;;
    -d | --dryrun )         shift
                            dryrun=1
                            ;;
    -f | --file )           shift
                            filename="$1"
                            ;;
    -h | --help )           help
                            exit
                            ;;
    -t | --type)            shift
                            testtype="$1"
                            ;;
    -p | --patch)           shift
                            patch="$1"
                            ;;
    * )                     echo "Invalid option: $1"
                            help
                            exit 1
  esac
  shift
done

run
