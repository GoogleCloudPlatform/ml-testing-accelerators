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