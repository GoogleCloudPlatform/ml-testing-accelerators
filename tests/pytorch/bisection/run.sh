# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: This algorithm assumes that the test you supply follows the pattern of:
# #
# fail, fail, fail, ..., pass, pass, pass, ...
#
# Where the 0 index is the latest image and the last index is the oldest
# image. In other words, it is assumed that the test you write was passing for
# some period of time and then switched to consistently failing. If the test
# passes intermittently, this bisection won't work.
IMAGE_NAME=gcr.io/tpu-pytorch/xla_debug
MIN_TIMESTAMP='20200825_02_12'
MAX_TIMESTAMP='20200825_08_00'
TAG_PREFIX='nightly_3.6_'
MAX_CHECKS=60  # Check 60 times w/ 30 sec wait = 1800s total.
LOGS_DIR=bisection_logs/

OLD_IFS=$IFS
IFS=$'\n'

# Collect all the Docker images that we want to try.
# These will be appended with newer images as the first elements.
images_to_try=()
indices_already_tried=()
for image in $(gcloud container images list-tags $IMAGE_NAME)
do
  timestamp=$(echo $image | awk '{ print $2 }')
  timestamp=${timestamp#$TAG_PREFIX}
  if [[ $timestamp =~ [0-9]{8}\_.* ]] && [[ "$timestamp" <  "$MAX_TIMESTAMP" ]] && [[ "$timestamp" >  "$MIN_TIMESTAMP" ]] ; then
    images_to_try+=($(echo $image | awk '{ print $2 }'))
    # Use a placeholder for the test result of each image.
    indices_already_tried+=(-2)
  fi
done
IFS=$OLD_IFS
echo "List of image digests to try in bisection loop: ${images_to_try[@]}"
echo "Indices already tried: ${indices_already_tried[@]}"
hi=${#images_to_try[@]}
lo=0
mid=$(($hi/2))
while true; do
  echo "hi: $hi"
  echo "lo: $lo"
  echo "mid: $mid"
  image_tag=${images_to_try[$mid]}
  echo "Trying image with tag: $image_tag"
  job_name=$(jsonnet -J ../ bisection_template.jsonnet --ext-str image-tag=$image_tag | kubectl create -f -)
  job_name=${job_name#job.batch/}
  job_name=${job_name% created}
  echo "Waiting on kubernetes job: $job_name" && \
  i=0 && \
  status_code=2 && \
  # Check on the job periodically. Set the status code depending on what
  # happened to the job in Kubernetes. If we try max_checks times and
  # still the job hasn't finished, give up and return the starting
  # non-zero status code.
  while [ $i -lt $MAX_CHECKS ]; do ((i++)); if kubectl get jobs $job_name -o jsonpath='Failed:{.status.failed}' | grep "Failed:1"; then status_code=1 && break; elif kubectl get jobs $job_name -o jsonpath='Succeeded:{.status.succeeded}' | grep "Succeeded:1" ; then status_code=0 && break; else echo "Job not finished yet"; fi; sleep 30; done && \
  echo "Done waiting. Job status code: $status_code" && \
  pod_name=$(kubectl get po -l controller-uid=`kubectl get job $job_name -o "jsonpath={.metadata.labels.controller-uid}"` | awk 'match($0,!/NAME/) {print $1}') && \
  echo "GKE pod name: $pod_name" && \
  kubectl logs -f $pod_name --container=train > "${LOGS_DIR}/${image_tag}.txt"
  echo "Done with log retrieval attempt." && \
  indices_already_tried[$mid]=$status_code

  # Decide the index of the next image to try.
  if [[ $status_code -eq 0 ]] ; then
    # Test succeeded so we need to go further ahead time to find first failure.
    # Images are ordered with newer images at lower index, so we lower the index.
    hi=$mid
    new_mid=$(((($hi+$lo)/2)))
  else
    # Test failed so we need to go further back in time to find earlier failure.
    # Images are ordered with newer images at lower index, so we raise the index.
    lo=$mid
    new_mid=$(((($hi+$lo+1)/2)))
  fi

  echo "new_mid: $new_mid"
  echo "Indices already tried: ${indices_already_tried[@]}"

  # Check termination condition.
  if [[ ${indices_already_tried[$new_mid]} -gt -2 ]] ; then
    # If we've already tried this index, then it's time to stop.
    # Find the earliest (i.e. highest index) image that was tried and returned
    # a non-0 exit code.
    max_failing_index=-1
    for j in {0..${#images_to_try[@]}}
    do
      if [[ ${indices_already_tried[$new_mid]} -gt -2 ]] && [[ ${indices_already_tried[$new_mid]} -ne 0 ]] && [[ $j -gt $max_failing_index ]]; then
        max_failing_index=$j
      fi
    done

    # Find the image associated with that index and report it.
    if [[ $max_failing_index -gt -1 ]] ; then
      echo "Earliest failing image was: ${images_to_try[$max_failing_index]}"
    else
      echo "No failing image was found."
    fi
    # End the bisection loop.
    break
  fi

  # If we make it here, prepare to do another pass of the while loop.
  mid=$new_mid
done
echo "See $LOGS_DIR for logs of all the attempted runs."
