import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filter", type=str, help="Test name filter.", default="")
args = parser.parse_args()

if not os.path.exists("all_tests.json"):
  raise ValueError("Please run `scripts/list-tests.sh`.")

with open("all_tests.json") as f:
  all_tests = json.load(f).keys()

if args.filter:
  all_tests = filter(lambda x: args.filter in x, all_tests)

print("\n".join(all_tests))
