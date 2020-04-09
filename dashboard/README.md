This README is under construction. More information coming soon.

Example dashboard setup for the data written by the metrics handler.

Contents of this repository:

* `dashboard`: Python code to generate the dashboard using the Bokeh library. This folder also contains a `Dockerfile` in case you wish to build the container.
* `kubernetes`: Configuration files to deploy the application using [Kubernetes](https://kubernetes.io/).

Command to run locally: `python3 -m bokeh serve --show dashboard/dashboard.py dashboard/metrics.py`
