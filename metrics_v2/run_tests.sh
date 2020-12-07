set -x
set -e

PYTHONPATH=.
python handler/utils_test.py
python handler/collectors/base_test.py
python handler/collectors/literal_collector_test.py
python handler/collectors/perfzero_collector_test.py
python handler/collectors/tensorboard_collector_test.py
