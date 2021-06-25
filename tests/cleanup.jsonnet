local clusters = import 'clusters.jsonnet';
local utils = import 'templates/utils.libsonnet';

local all_tests = import 'all_tests.jsonnet';

local clusterTests = utils.splitByCluster(
  std.objectValues(all_tests),
  clusters.defaultCluster,
  clusters.clusterAccelerators,
);
local clusterTestNames = {
  [cluster]: [test.testName for test in clusterTests[cluster]]
  for cluster in std.objectFields(clusterTests)
};

local cleaners = {
  [cluster]: {
    apiVersion: 'batch/v1',
    kind: 'CronJob',
    metadata: {
      name: 'cronjob-cleanup'
    },
    spec: {
      schedule: '0 */4 * * *',
      jobTemplate: {
        spec: {
          backoffLimit: 0,
          template: {
            spec: {
              containers: [
                {
                  name: 'clean',
                  image: 'google/cloud-sdk',
                  command: [
                    'bash',
                    '-cx'
                  ],
                  args: [
                    |||
                      cat > tests.txt <<EOF
                      %s
                      EOF

                      delete_list=$(kubectl get cronjob --namespace=automated -l benchmarkId -o name | grep -v -f tests.txt)
                      if [[ "$delete_list" ]]; then
                        kubectl delete $delete_list;
                      fi
                    ||| % std.join('\n', clusterTestNames[cluster]),
                  ],
                },
              ],
              restartPolicy: 'Never',
              serviceAccount: 'cronjob-cleaner',
            },
          },
        },
      },
    },
  }
  for cluster in std.objectFields(clusterTestNames)
};

{
  [cluster + '/gen/cronjob-cleanup.yaml']: std.manifestYamlDoc(cleaners[cluster])
  for cluster in std.objectFields(cleaners)
}
