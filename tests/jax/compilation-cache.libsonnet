// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local common = import 'common.libsonnet';
local mixins = import 'templates/mixins.libsonnet';

{
  local compilationCacheTest = self.compilationCacheTest,
  compilationCacheTest:: common.JaxTest + mixins.Functional {
    modelName: 'compilation-cache-test',

    testScript:: |||
      set -x
      set -u
      set -e

      # .bash_logout sometimes causes a spurious bad exit code, remove it.
      rm .bash_logout

      pip install --upgrade pip
      pip install --upgrade numpy==1.18.5 scipy wheel future six cython pytest \
          absl-py opt-einsum msgpack

      %(installLocalJax)s
      %(maybeBuildJaxlib)s
      %(printDiagnostics)s

      num_devices=`python3 -c "import jax; print(jax.device_count())"`
      if [ "$num_devices" = "1" ]; then
        echo "No TPU devices detected"
        exit 1
      fi

      mkdir "/tmp/compilation_cache_integration_test"
      cat >integration.py <<'END_SCRIPT'
      import jax
      from jax.experimental.compilation_cache import compilation_cache as cc
      from jax import pmap, lax
      from jax._src.config import config
      import numpy as np

      config.update('jax_persistent_cache_min_compile_time_secs', 0)
      cc.initialize_cache("/tmp/compilation_cache_integration_test")
      f = pmap(lambda x: x - lax.psum(x, 'i'), axis_name='i')
      print(f(np.arange(8)))
      END_SCRIPT

      cat >directory_size.py <<'END_SCRIPT'
      import os
      num_of_files = sum(1 for f in os.listdir("/tmp/compilation_cache_integration_test"))
      assert num_of_files == 1, f"The number of files in the cache should be 1 but is {num_of_files}"
      END_SCRIPT

      python3 integration.py
      python3 directory_size.py
      python3 integration.py
      python3 directory_size.py

    ||| % self.scriptConfig,
  },

  configs: [
    compilationCacheTest + common.jaxlibLatest + common.tpuVmBaseImage,
  ],
}
