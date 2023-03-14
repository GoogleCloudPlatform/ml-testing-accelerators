#!/bin/bash

# Create UUID for job creation / lookup
uuid=$(uuidgen)

# First, check that the type is set
if [[ -z "${XLML_TEST_TYPE}" ]]; then
    echo "Please set XLML_TEST_TYPE"
    exit 1
else
    test_type="${XLML_TEST_TYPE}"
fi

# Loop over regions and connect to each cluster
regions=(us-central1 us-central2 europe-west4)
for region in ${regions[@]}; do
    gcloud container clusters get-credentials xl-ml-test-"$region" --region "$region" --project xl-ml-test

    # Loop over cron jobs, grepping for the test type
    for i in $(kubectl get cronjobs -n automated -o name | grep "$test_type"); do
        echo "$i"

        # Finally, parse the jobname from the cronjob and create the new job
        jobName=$(echo "$i" | awk -F/ '{print $2}')
        echo "Creating $jobName-$uuid"
        kubectl create job --from="$i" "$jobName-$uuid" -n automated
    done
done

echo "Find the scheduled jobs by filtering with the uuid: $uuid"
