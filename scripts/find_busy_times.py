"""Computes the expected number of tests on TPUs at different times of day.
  
Parses all the kubernetes test configs in a given directory. This script
relies on the `activeDeadlineSeconds` field in the configs to tell how long
a job is expected to last. If that field is overly generous in your test
configs, then this script will overestimate the number of active jobs.

This script outputs an HTML file to `job_frequency.html` with a table of
days/times and the number of expected tests running at that time.

The `chunk_minutes` flag determines granularity of the grid. For example if
`chunk_minutes=10`, then the grid will show 10-minute intervals in the
output grid.

Example usage:

python3 scripts/find_busy_times.py --dir_path=/Users/me/ml-testing-accelerators/k8s/us-central1/gen/* --chunk_minutes=15
"""

import datetime
import glob
import re
import sys

import croniter

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dir_path', None, 'Dir to search for test configs.')
flags.DEFINE_integer('chunk_minutes', 10, 'Number of minutes per chunk.')

_DEADLINE_REGEX = r'"?activeDeadlineSeconds"?: "?(\d+)"?'
_SCHEDULE_REGEX = r'"?schedule"?: "?([ \*0-9,\-%LW#\?/H]+)"?'

def get_deadline_and_schedules():
  file_to_schedule_and_deadline = {}
  for file in glob.iglob(FLAGS.dir_path):
    is_tpu_test = False
    deadline_sec = None
    schedule = None
    for line in open(file, 'r'):
      if re.search("""cloud-tpus.google.com/\S+: [0-9]+""", line):
        is_tpu_test = True
      dd = re.findall(_DEADLINE_REGEX, line)
      if dd:
        deadline_sec = int(dd[0])
      ss = re.findall(_SCHEDULE_REGEX, line)
      if ss:
        schedule = ss[0]
    if is_tpu_test:
      file_to_schedule_and_deadline[file] = (deadline_sec, schedule)
  return file_to_schedule_and_deadline

def _hour_and_minute_to_index(hour, minute, chunk_minutes):
  return ((hour * 60) + minute) // chunk_minutes

def schedules_to_counts(schedules_dict):
  chunk_minutes = FLAGS.chunk_minutes
  # We will record the number of jobs expected to be active at intervals
  # throughout the day for every day of the week.
  raw_counts = [[0 for _ in range(7)] for _ in range(24*60//chunk_minutes)]
  start_time = datetime.datetime.utcnow()
  end_time = start_time + datetime.timedelta(days=7)
  for file in schedules_dict:
    deadline_sec, schedule = schedules_dict[file]
    cron = croniter.croniter(schedule, start_time)
    start_of_job = cron.get_next(datetime.datetime)
    while start_of_job < end_time:
      number_of_chunks = deadline_sec // (chunk_minutes * 60)
      for n in range(number_of_chunks):
        dt = start_of_job + datetime.timedelta(minutes=n*chunk_minutes)
        raw_counts[_hour_and_minute_to_index(
            dt.hour, dt.minute, chunk_minutes)][dt.weekday()] += 1
      start_of_job = cron.get_next(datetime.datetime)
  return(raw_counts)

def counts_to_table(raw_counts):
  chunk_minutes = FLAGS.chunk_minutes
  assert len(raw_counts) == 24*60//chunk_minutes
  assert len(raw_counts[0]) == 7
  cell_width = 50
  cell_style = f'style="width:{cell_width}px; border:1px solid #cfcfcf"'
  table_width = cell_width * len(raw_counts[0])
  table_html = '<p>Each cell contains the estimated number of jobs running at a time.</p>'
  table_html += '<p>All times are in UTC. Days of week match cron schedule, i.e.:</p>'
  table_html += '<ul><li>0 = Sun</li><li>1 = Mon</li><li>2 = Tue</li><li>3 = Wed</li>'
  table_html += '<li>4 = Thu</li><li>5 = Fri</li><li>6 = Sat</li></ul>'
  table_html += f'<table style="width:{table_width}px">'
  table_html += f'<tr><th {cell_style}>Time of day</th>'
  for i in range(7):
    table_html += f'<th {cell_style}>Day: {i}</th>'
  table_html += '</tr>'
  base_dt = datetime.datetime(hour=0, minute=0, year=2000, month=1, day=1)
  for i, row in enumerate(raw_counts):
    # First column is the time of day.
    current_dt = base_dt + datetime.timedelta(minutes=i*chunk_minutes)
    table_html += f"<tr><td {cell_style}>{current_dt.strftime('%H:%M')}</td>"
    for col in row:
      table_html += f'<td {cell_style}>{col}</td>'
    table_html += '</tr>'
  table_html += '</table>'
  return table_html

def main(_):
  schedules_dict = get_deadline_and_schedules()
  assert 60 % FLAGS.chunk_minutes == 0
  raw_counts = schedules_to_counts(schedules_dict)
  html_table = counts_to_table(raw_counts)
  with open('job_frequency.html', 'w') as outfile:
    outfile.write(html_table)

if __name__ == '__main__':
  app.run(main)
