#!/bin/bash
accelerator=
dryrun=
filename=
testtype=
test_name=

help()
{
  echo "Utility script to run one shots with XLML test."
  echo "This script will:"
  echo "- connect to the appropriate one shot cluster"
  echo "- run the test"
  echo
  echo "Syntax: ./scripts/run-oneshot.sh [-a|d|f|h|t]"
  echo "-a | --accelerator   The accelerator type, e.g. v2-8, v3-8, 8xv100, etc."
  echo "-d | --dryrun        Dryrun. If True, then the test does not run."
  echo "-f | --file          The filepath for the test."
  echo "-h | --help          Print this help."
  echo "-t | --type          Test type, e.g. [func|conv|functional|convergence]."
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
    * )                      echo "Invalid test type provided."
                             exit
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
  else
    platform="pt"
  fi

  release="${split[2]}"
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
    jsonnet tests/oneshot.jsonnet -J . -S --tla-str test=$test_name | kubectl create -f -
  else
    echo "gcloud container clusters get-credentials oneshots-$region --region $region --project xl-ml-test"
    echo "jsonnet tests/oneshot.jsonnet -J . -S --tla-str test=$test_name | kubectl create -f -"
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
    -t | --testtype)        shift
                            testtype="$1"
                            ;;
    * )                     usage
                            exit 1
  esac
  shift
done

run
