set -x
set -e

rm -f .coverage

PYTHONPATH=handler/
coverage run -a publisher/event_publisher_test.py
coverage run -a -m handler.utils_test
coverage run -a -m handler.alerts_test
coverage run -a -m handler.collectors.base_test
coverage run -a -m handler.collectors.literal_collector_test
coverage run -a -m handler.collectors.perfzero_collector_test
coverage run -a -m handler.collectors.tensorboard_collector_test
coverage run -a integration_test.py

coverage report -m