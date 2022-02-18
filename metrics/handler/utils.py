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

import dataclasses
import math

@dataclasses.dataclass(frozen=True)
class Bounds:
  """Represents numerical bounds for a metric."""
  lower: float
  upper: float
  inclusive: bool = False

  def contains(self, value: float) -> bool:
    """Determines if the given value is contained in these bounds."""
    if self.inclusive and (
      math.isclose(value, self.lower) or math.isclose(value, self.upper)):
      return True
  
    return value > self.lower and value < self.upper

NO_BOUNDS = Bounds(-math.inf, math.inf, True)

@dataclasses.dataclass(frozen=True)
class MetricPoint:
  """Represents a single data point for a metric.

  Attributes:
    metric_key: The unique name of a metric.
    metric_value: The value of this metric point.
    bounds: The computed bounds for this metric point.
  """
  metric_key: str
  metric_value: float
  bounds: Bounds

  def __iter__(self):
    yield self.metric_key
    yield self.metric_value
    yield self.bounds

  def within_bounds(self) -> bool:
    """Determines if a MetricPoint's value is within its bounds."""
    return self.bounds.contains(self.metric_value)