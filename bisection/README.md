# Bisection tool

This is an example of using bisection to check for regressions in a TPU model.

It would be useful in a case where you have several Docker images at different points in time and a TPU model that was passing consistently up until a certain point in time and then it switched to failing consistently. This tool can identify the docker image at which the test started failing. The test could be e.g. checking for a crash or checking for a slowdown in performance.

Setup instructions:
1. Clone the code.
2. Modify `bisection_template.jsonnet`. In particular, update the `command` to be the test you want to run with the success check that makes sense for your case.
3. Modify `run.sh`. In particular, update `MIN_TIMESTAMP` and `MAX_TIMESTAMP` to reflect the time period you want to check. Update `IMAGE_NAME` and `MAX_CHECKS` if necessary.
4. Install [prerequisites](https://github.com/GoogleCloudPlatform/ml-testing-accelerators/blob/master/doc/developing.md#Prerequisites), including `jsonnet`, if you haven't already.
5. [Set up a GKE cluster](https://github.com/GoogleCloudPlatform/ml-testing-accelerators/tree/master/deployments) if you don't already have one and then connect to it. You can find the command to connect in your cluster page: `https://console.cloud.google.com/kubernetes/list?project=MY-PROJECT`

Usage:
`bash run.sh > my_output.txt`
