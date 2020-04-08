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

FROM google/cloud-sdk:slim

RUN apt-get update && apt-get install -y --no-install-recommends jq

RUN mkdir -p /root/google-cloud-sdk/bin/
RUN ln -s /usr/bin/gcloud /root/google-cloud-sdk/bin/gcloud

COPY images/setup.sh /
COPY images/publish.sh /
COPY images/pytorch-pods/setup-instances.sh /
COPY images/pytorch-pods/teardown-instances.sh /
COPY images/pytorch-pods/entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
