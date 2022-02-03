# Running locally

## Building the client

```
cd client
npm run build
```

(or "npm run watch" for continuous building)

## Running the server

```
export JOB_HISTORY_TABLE_NAME='your-project-name.metrics_handler_dataset.job_history'
export TEST_NAME_PREFIXES='prefix1,prefix2'

cd server
python3 main.py
```
