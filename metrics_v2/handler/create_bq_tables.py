from absl import app
from absl import flags
from absl import logging

import bigquery_client

flags.DEFINE_string('dataset', None, 'Name of BigQuery dataset to create.')
flags.DEFINE_string('project', None, 'Name of project to create BQ tables in.')

FLAGS = flags.FLAGS

def main(argv):
  bq = bigquery_client.BigQueryMetricStore(FLAGS.dataset, FLAGS.project)
  bq.create_tables()
  logging.info('Done.')

if __name__ == "__main__":
  flags.mark_flags_as_required(['dataset', 'project'])
  app.run(main)