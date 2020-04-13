export DASHBOARD_NAME_PREFIX="xl-ml-test-dashboard"
export BACKEND_PORT=30033

# TODO: This should match the zone used for your cluster.
gcloud config set compute/zone us-central1-a

export PROJECT_ID=$(gcloud info --format='value(config.project)')
export SERVICE_ACCOUNT_NAME="testing-dashboard"

echo "Creating dashboard service account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME --display-name="${DASHBOARD_NAME_PREFIX}"

echo "Adding Bigquery permission to service account..."
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" --role roles/bigquery.user

echo "Creating service account key..."
gcloud iam service-accounts keys create --iam-account "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" service-account-key.json
#gcloud iam service-accounts keys create --iam-account "1030754782689-compute@developer.gserviceaccount.com" uber-service-account-key.json

echo "Creating kubernetes cluster..."
gcloud container clusters create "${DASHBOARD_NAME_PREFIX}-cluster" --tags "${DASHBOARD_NAME_PREFIX}-node" --num-nodes=1

echo "Creating secret to hold the project id..."
kubectl create secret generic project-id --from-literal project-id=$PROJECT_ID

echo "Creating secret to hold the service account key..."
kubectl create secret generic service-account-key --from-file service-account-key=service-account-key.json

echo "Creating static ip..."
gcloud compute addresses create "${DASHBOARD_NAME_PREFIX}-static-ip" --global

export STATIC_IP=$(gcloud compute addresses describe "${DASHBOARD_NAME_PREFIX}-static-ip" --global --format="value(address)")
export DOMAIN="${STATIC_IP}.xip.io"
echo "My domain: ${DOMAIN}"

echo "Creating secret to hold the domain..."
kubectl create secret generic domain --from-literal domain=$DOMAIN

echo "Deploying the bokeh service on kubernetes..."
kubectl create -f kubernetes/bokeh.yaml

echo "Deploying the memcached service..."
kubectl create -f kubernetes/memcached.yaml

echo "Making the ssl certificate..."
mkdir "/tmp/${DASHBOARD_NAME_PREFIX}-ssl"
cd "/tmp/${DASHBOARD_NAME_PREFIX}-ssl"
openssl genrsa -out ssl.key 2048
openssl req -new -key ssl.key -out ssl.csr -subj "/CN=${DOMAIN}"
openssl x509 -req -days 365 -in ssl.csr -signkey ssl.key -out ssl.crt
cd -

echo "Creating firewall rules..."
gcloud compute firewall-rules create "${DASHBOARD_NAME_PREFIX}-lb7-fw" --target-tags "${DASHBOARD_NAME_PREFIX}-node" --allow "tcp:${BACKEND_PORT}" --source-ranges 130.211.0.0/22,35.191.0.0/16

echo "Creating health checks..."
gcloud compute health-checks create http "${DASHBOARD_NAME_PREFIX}-basic-check" --port $BACKEND_PORT --healthy-threshold 1 --unhealthy-threshold 10 --check-interval 60 --timeout 60

echo "Creating an instance group..."
export INSTANCE_GROUP=$(gcloud container clusters describe "${DASHBOARD_NAME_PREFIX}-cluster" --format="value(instanceGroupUrls)" | awk -F/ '{print $NF}')

echo "Creating named ports..."
gcloud compute instance-groups managed set-named-ports $INSTANCE_GROUP --named-ports "port${BACKEND_PORT}:${BACKEND_PORT}"

echo "Creating the backend service..."
gcloud compute backend-services create "${DASHBOARD_NAME_PREFIX}-service" --protocol HTTP --health-checks "${DASHBOARD_NAME_PREFIX}-basic-check" --port-name "port${BACKEND_PORT}" --global

echo "Connecting instance group to backend service..."
export INSTANCE_GROUP_ZONE=$(gcloud config get-value compute/zone)
gcloud compute backend-services add-backend "${DASHBOARD_NAME_PREFIX}-service" --instance-group $INSTANCE_GROUP --instance-group-zone $INSTANCE_GROUP_ZONE --global

echo "Creating URL map..."
gcloud compute url-maps create "${DASHBOARD_NAME_PREFIX}-urlmap" --default-service "${DASHBOARD_NAME_PREFIX}-service"

echo "Uploading SSL certificates..."
gcloud compute ssl-certificates create "${DASHBOARD_NAME_PREFIX}-ssl-cert" --certificate "/tmp/${DASHBOARD_NAME_PREFIX}-ssl/ssl.crt" --private-key "/tmp/${DASHBOARD_NAME_PREFIX}-ssl/ssl.key"

echo "Creating HTTPS target proxy..."
gcloud compute target-https-proxies create "${DASHBOARD_NAME_PREFIX}-https-proxy" --url-map "${DASHBOARD_NAME_PREFIX}-urlmap" --ssl-certificates "${DASHBOARD_NAME_PREFIX}-ssl-cert"

echo "Creating global forwarding rule..."
gcloud compute forwarding-rules create "${DASHBOARD_NAME_PREFIX}-gfr" --address $STATIC_IP --global --target-https-proxy "${DASHBOARD_NAME_PREFIX}-https-proxy" --ports 443

echo "Extending the connection timeout..."
gcloud compute backend-services update "${DASHBOARD_NAME_PREFIX}-service" --global --timeout=86400

gcloud config unset compute/zone

printf "\n\n\nCheck for dashboard in 5-10 minutes at: https://${STATIC_IP}.xip.io/dashboard\n"

