#!/bin/bash
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


echo "Getting GCE metadata..."

export PROJECT=`curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google"`
echo "Project: ${PROJECT}"

# "/project/[project number]/zone/us-central1-a" -> "us-central1-a"
export ZONE=`curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}'`
echo "Zone: ${ZONE}"

# "us-central1-a" -> "us-central1"
export REGION=`echo ${ZONE} | awk -F'-' '{print $1"-"$2}'`
echo "Region: ${REGION}"

export CLUSTER=`curl -sS http://metadata.google.internal/computeMetadata/v1/instance/attributes/cluster-name -H "Metadata-Flavor: Google"`
echo "Cluster: ${CLUSTER}"

export STACKDRIVER_LOGS="https://console.cloud.google.com/logs?project=$PROJECT&advancedFilter=resource.type%3Dk8s_container%0Aresource.labels.project_id%3D${PROJECT}%0Aresource.labels.location:${REGION}%0Aresource.labels.cluster_name=${CLUSTER}%0Aresource.labels.namespace_name=${POD_NAMESPACE}%0Aresource.labels.pod_name:${JOB_NAME}"
echo "Stackdriver Log link: ${STACKDRIVER_LOGS}"
