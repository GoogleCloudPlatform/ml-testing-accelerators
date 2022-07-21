# Event Publisher Deployment

Deploy the event publisher with `envsubst`:

```
PUBSUB_TOPIC=... PUBLISHER_IMAGE_URL=... GCS_BUCKET=... envsubst < publisher/deployment/event-publisher.yaml | kubectl apply -f -
```
