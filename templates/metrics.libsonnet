// Copyright 2020 Google LLC
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

{
  local metrics = self,

  TensorBoardSourceHelper:: {
    local aggregateAssertionsMapToList(map) = [
      {
        tag: tag,
        strategy: strategy,
        assertion: map[tag][strategy],
      }
      for tag in std.objectFields(map)
      for strategy in std.objectFields(map[tag])
    ],

    aggregateAssertionsMap:: {},
    aggregate_assertions: aggregateAssertionsMapToList(self.aggregateAssertionsMap),
  },
  MetricCollectionConfigHelper:: {
    local helper = self,

    sourceMap:: {
      tensorboard: {},
      perfzero: {},
      literals: {},
    },

    sources: [
      { [source]: helper.sourceMap[source] }
      for source in std.objectFields(std.prune(self.sourceMap))
    ],
  },

  // Experimental: convert from old metrics config format to the new one.
  // Only works for TensorBoard metrics. Does not work for all fields.
  // This is a hack. Don't rely on it for new code.
  CompatMetrics(metricCollectionConfig, regressionTestConfig)::
    local hasValidAssertions = regressionTestConfig != null && std.objectHas(regressionTestConfig, 'metric_success_conditions');
    local hasDurationMetric = hasValidAssertions && std.objectHas(regressionTestConfig.metric_success_conditions, 'total_wall_time');
    local hasPzMetrics = hasValidAssertions && std.objectHas(regressionTestConfig.metric_success_conditions, 'exp_per_second');
    local getOrDefault(obj, field, default) =
      if std.objectHas(obj, field) then
        obj[field]
      else
        default;
    local tagAndStrategy(metric) =
      local len = std.length(metric);
      local split = std.split(metric, '_');
      local suffix = split[std.length(split) - 1];
      local suffixLen = std.length(suffix);
      local tag = std.substr(metric, 0, len - suffixLen - 1);

      { tag: tag, strategy: std.asciiUpper(suffix), metric: metric };
    local convertComparison(comparison) = std.asciiUpper(std.split(comparison, '_')[0]);
    local convertAssertion(assertion) =
      {
        inclusive_bounds: std.findSubstr('equal', assertion.comparison) != [],
        wait_for_n_data_points: getOrDefault(assertion, 'wait_for_n_points_of_history', 0),
      } + if std.objectHas(assertion.success_threshold, 'stddevs_from_mean') then
        {
          std_devs_from_mean: {
            std_devs: assertion.success_threshold.stddevs_from_mean,
            comparison: convertComparison(assertion.comparison),
          },
        } else
        {
          fixed_value: {
            value: assertion.success_threshold.fixed_value,
            comparison: convertComparison(assertion.comparison),
          },
        };
    local isTbMetric(s) = !std.member(['exp_per_second', 'total_wall_time', 'time_to_accuracy', 'startup_time'], s);

    metrics.MetricCollectionConfigHelper {
      sourceMap: {
        literals: if hasDurationMetric then {
          assertions: {
            duration: convertAssertion(regressionTestConfig.metric_success_conditions.total_wall_time),
          },
        },
        perfzero: if hasPzMetrics then {
          assertions: {
            exp_per_second: convertAssertion(regressionTestConfig.metric_success_conditions.exp_per_second),
          },
        },
        tensorboard: metrics.TensorBoardSourceHelper {
          merge_runs: !getOrDefault(metricCollectionConfig, 'use_run_name_prefix', false),
          include_tags: [
            {
              tag_pattern: '*',
              strategies: std.map(std.asciiUpper, metricCollectionConfig.default_aggregation_strategies),
            },
          ],
          exclude_tags: getOrDefault(metricCollectionConfig, 'tags_to_ignore', []),
          aggregateAssertionsMap::
            if hasValidAssertions then
              std.foldl(
                function(last, x) last {
                  [x.tag]+: {
                    [x.strategy]: convertAssertion(regressionTestConfig.metric_success_conditions[x.metric]),
                  },
                },
                std.map(
                  tagAndStrategy,
                  std.filter(
                    isTbMetric,
                    std.objectFields(regressionTestConfig.metric_success_conditions)
                  ),
                ),
                {},
              )
            else
              {},
        },
      },
    },

}
