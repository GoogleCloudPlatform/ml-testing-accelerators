# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from absl.testing import absltest
from absl.testing import parameterized

from handler import utils

class UtilsTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ('less', 5., (-math.inf, 10.), False, True),
    ('equal', 5., (5., 5.), True, True),
    ('less_inclusive', 5., (-math.inf, 5.), True, True),
    ('within', 3., (0., 5.), False, True),
    ('less_exclusive', 5., (-math.inf, 5.), False, False),
    ('greater', 5., (-math.inf, 5.), False, False),
    ('outside', 10, (0., 5.), False, False),
    ('greater_inclusive', 5., (5., math.inf), True, True),
  )
  def test_within_bounds(self, value, lower_upper, inclusive, expected):
    point = utils.MetricPoint("metric_key", value, 
        utils.Bounds(*lower_upper, inclusive))
    self.assertIs(point.within_bounds(), expected)

if __name__ == '__main__':
  absltest.main()
