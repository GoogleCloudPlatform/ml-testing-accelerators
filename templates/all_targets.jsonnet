local garden_tests = import "garden/targets.jsonnet";
local pytorch_tests = import "pytorch/targets.jsonnet";

local all_tests = garden_tests + pytorch_tests;

# Mapping from unique test name to test config
{
  [test.testName]: test for test in all_tests
}
