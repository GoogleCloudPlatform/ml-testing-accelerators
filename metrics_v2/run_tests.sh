set -x
set -e

PYTHONPATH=.
coverage run -a handler/utils_test.py
coverage run -a handler/alerts_test.py
coverage run -a handler/collectors/base_test.py
coverage run -a handler/collectors/literal_collector_test.py
coverage run -a handler/collectors/perfzero_collector_test.py
coverage run -a handler/collectors/tensorboard_collector_test.py

coverage report -m