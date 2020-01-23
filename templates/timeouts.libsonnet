{
  local timeouts = self,

  # Conversions in secondss
  one_minute: 60,
  one_hour: 3600,
  ten_hours: 10 * timeouts.one_hour,

  Minutes(x):: {timeout: x * timeouts.one_minute},
  Hours(x):: {timeout: x * timeouts.one_hour},
}
