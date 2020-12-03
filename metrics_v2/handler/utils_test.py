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
