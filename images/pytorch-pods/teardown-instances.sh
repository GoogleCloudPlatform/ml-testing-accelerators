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

set -x

echo "Tearing down GCE Instance Group: ${INSTANCE_GROUP_NAME}"
gcloud -q compute --project="${PROJECT}" instance-groups managed \
  delete \
  "${INSTANCE_GROUP_NAME}" \
  --zone="${ZONE}"

echo "Tearing down GCE Instance Template: ${INSTANCE_TEMPLATE_NAME}"
gcloud -q compute --project="${PROJECT}" instance-templates \
  delete \
  "${INSTANCE_TEMPLATE_NAME}"
