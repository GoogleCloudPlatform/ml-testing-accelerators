{
  local timeouts = self,

  # Conversions in secondss
  one_minute: 60,
  one_hour: 3600,
  ten_hours: 10 * timeouts.one_hour,

  minutes(x):: {timeout: x * timeouts.one_minute},
  hours(x):: {timeout: x * timeouts.one_hour},
}
