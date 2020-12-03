import dataclasses
import math

@dataclasses.dataclass(frozen=True)
class Bounds:
  lower: float
  upper: float
  inclusive: bool = False

  def contains(self, value: float) -> bool:
    if self.inclusive and (
      math.isclose(value, self.lower) or math.isclose(value, self.upper)):
      return True
  
    return value > self.lower and value < self.upper

NO_BOUNDS = Bounds(-math.inf, math.inf, True)

@dataclasses.dataclass(frozen=True)
class MetricPoint:
  metric_key: str
  metric_value: float
  bounds: Bounds

  def __iter__(self):
    yield self.metric_key
    yield self.metric_value
    yield self.bounds

  def within_bounds(self) -> bool:
    return self.bounds.contains(self.metric_value)