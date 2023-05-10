#!/bin/bash
set -x
set -u

zones=("us-central1-b" "us-central2-b" "europe-west4-a")

for zone in ${zones[@]}; do
  region=${zone%-*}
  cluster=xl-ml-test-$region
  gcloud container clusters get-credentials $cluster --region $region --project xl-ml-test

  # Save the UID of any pod that might be using a TPU so we
  # don't delete any in-use TPUs.
  pods_to_save=$(mktemp)
  for namespace in default automated; do
    for status in Running Pending; do
      kubectl get pods -n $namespace --field-selector status.phase=$status -o jsonpath='{.items[*].metadata.uid}' \
        | tr ' ' '\n' \
          >> $pods_to_save
      # Ensure there is a newline between outputs
      sed -ie '$a\' $pods_to_save
    done
  done

  leaks=$(
    gcloud compute tpus tpu-vm list --zone $zone --project=xl-ml-test --format="value(name)" \
      | grep "^tpu-" \
      | grep -v -f $pods_to_save)
  for leak in $leaks; do
    gcloud compute tpus tpu-vm delete --quiet --async --zone $zone --project=xl-ml-test $leak
  done
done
