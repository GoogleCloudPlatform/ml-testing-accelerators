#!/bin/bash

regions=(us-central1 us-central2 europe-west4)

function schedule()
{
    # Loop over regions and connect to each cluster
    for region in ${regions[@]}; do
        gcloud container clusters get-credentials xl-ml-test-"$region" --region "$region" --project xl-ml-test >/dev/null 2>&1

        # Loop over cron jobs, grepping for the test type
        for i in $(kubectl get cronjobs -n automated -o name | grep "$test_type"); do
            # Finally, parse the jobname from the cronjob and create the new job
            jobName=$(echo "$i" | awk -F/ '{print $2}')
            echo "Creating $jobName-$uuid"
            kubectl create job --from="$i" "$jobName-$uuid" -n automated
        done
    done
}

# Create UUID for job creation / lookup
uuid=$(uuidgen -r | awk -F- '{print $1}')

# First, check that the type is set
if [[ -z "${XLML_TEST_TYPE}" ]]; then
    echo "Please set XLML_TEST_TYPE"
    exit 1
else
    test_type="${XLML_TEST_TYPE}"
fi

echo "Fetching all jobs of type $test_type"

# Loop over regions to count the total number of jobs
total_jobs=0
for region in ${regions[@]}; do
    gcloud container clusters get-credentials xl-ml-test-"$region" --region "$region" --project xl-ml-test >/dev/null 2>&1
    jobs_in_region=$(kubectl get cronjobs -n automated -o name | grep -o "$test_type" | wc -l)
    total_jobs=$((total_jobs+jobs_in_region))
done

# Confirm that the jobs should be scheduled
echo "Scheduling $total_jobs jobs of the type $test_type."
read -p "Is this correct? (If so, press 'y') " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Scheduling jobs..."
  schedule
  echo "All jobs have been scheduled. Find the scheduled jobs by filtering with the uuid: $uuid"
else
  echo "Aborting"
fi
