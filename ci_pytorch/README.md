# Continuous Integration (CI) using TPUs.

This repo shows an example CI setup that runs a PyTorch test on TPUs using Github Actions.

There are 2 required pieces:
1. Github Actions workflow file ([example](../.github/workflows/ci_pytorch.yml)) or CircleCI config file ([example](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/.circleci/config.yml)).
2. The files found in this directory.

## CircleCI vs Github Actions

In addition to pricing and user interface differences, there are important
differences between these platforms with regards to security and privacy around
forked repo PRs.

Credentials for your GKE cluster are stored as Secrets in the CI platform.

CircleCI allows the option to [share Secrets with forked
PRs](https://circleci.com/docs/2.0/oss/#pass-secrets-to-builds-from-forked-pull-requests)
while Github Actions does not.

If most of your repo's contributions come from forked PRs **and** you want to
run pre-submit checks on those PRs, CircleCI is the recommended choice.

If post-submit checks on submitted PRs plus scheduled runs are sufficient for
your case **or** most PRs on your repo are branches off the main repo (not
forked repos), then Github Actions is recommended.

**Do consider the security ramifications**. Running pre-submit checks on every
PR regardless of the author is very convenient, but it does mean that any user
can kick off jobs on your GKE cluster. A user could replace the contents of
your test files with whatever code they want and start up that job on your
cluster. Consider adding a cleanup job like
[this](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/.circleci/config.yml#L86-L97)
that checks periodically for long-running jobs on your cluster and kills them.
Note that all of the above applies generally to CI frameworks - the only
difference here is that the user has access both to CI machines using your quota
(as with most CI setups) as well as GKE machines (unique to this setup).


## Setup steps

1. Go to the "Actions" page in your Github repo and create a new sample workflow. Submit the auto-generated PR.
2. Set up a GKE cluster following instructions in the [deployments dir](../deployments).
3. Collect a service account key for Github Actions to use.
  a. https://console.cloud.google.com/iam-admin/serviceaccounts?project=MY-PROJECT
  b. Click “Create Service Account” at the top of page.
  c. Add 3 roles to the account: "Kubernetes Engine Developer", "Logs Viewer", "Storage Admin".
  d. Once the account is created, click the “...” button on the Service Accounts page for the new account and click “Create Key” (use the JSON option).
4. Navigate to Github -> Settings -> Secrets and set some Key:Value pairs:
  a. `GKE_PROJECT`: my-project
  b. `GKE_CLUSTER`: my-cluster-name
  c. `GKE_SA_KEY_BASE64`: `cat ~/Downloads/my-project-3aaad123f0a.json | base64` (use the service account key you downloaded in step 3. Make sure there are no newlines in the base-64 encoded string you paste into the Secret).
5. Begin work on a new PR where you modify the files in this directory as needed and replace the contents of your auto-generated workflow with the contents of [this example workflow](../.github/workflows/ci_pytorch.yml) (modified as needed). Recommended to uncomment "pull_request" in the workflow so that it runs for every commit rather than only when you submit the PR.
