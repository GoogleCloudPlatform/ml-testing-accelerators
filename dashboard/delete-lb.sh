export DASHBOARD_NAME_PREFIX="xl-ml-test-dashboard"
export BACKEND_PORT=30033
export PROJECT_ID=$(gcloud info --format='value(config.project)')
export SERVICE_ACCOUNT_NAME="testing-dashboard"

# TODO: This should match the zone used for your cluster.
gcloud config set compute/zone us-central1-a

echo "Deleting global forwarding rule..."
gcloud compute forwarding-rules delete "${DASHBOARD_NAME_PREFIX}-gfr" --global

echo "Deleting HTTPS target proxy..."
gcloud compute target-https-proxies delete "${DASHBOARD_NAME_PREFIX}-https-proxy"

echo "Deleting SSL certificates..."
gcloud compute ssl-certificates delete "${DASHBOARD_NAME_PREFIX}-ssl-cert"

echo "Deleting URL map..."
gcloud compute url-maps delete "${DASHBOARD_NAME_PREFIX}-urlmap"

#echo "Removing instance group to backend service..."
#export INSTANCE_GROUP_ZONE=$(gcloud config get-value compute/zone)
#gcloud compute backend-services add-backend xl-ml-test-dashboard-service --instance-group $INSTANCE_GROUP --instance-group-zone $INSTANCE_GROUP_ZONE --global

echo "Deleting the backend service..."
gcloud compute backend-services delete "${DASHBOARD_NAME_PREFIX}-service" --global

#echo "Deleting named ports..."
#gcloud compute instance-groups managed set-named-ports $INSTANCE_GROUP --named-ports "port${BACKEND_PORT}:${BACKEND_PORT}"

#echo "Setting instance group..."
#export INSTANCE_GROUP=$(gcloud container clusters describe xl-ml-test-dashboard-cluster --format="value(instanceGroupUrls)" | awk -F/ '{print $NF}')

echo "Delete health checks..."
gcloud compute health-checks delete "${DASHBOARD_NAME_PREFIX}-basic-check"

echo "Creating firewall rules..."
gcloud compute firewall-rules delete "${DASHBOARD_NAME_PREFIX}-lb7-fw"

echo "Deleting the Kubernetes cluster..."
gcloud container clusters delete "${DASHBOARD_NAME_PREFIX}-cluster" --quiet

echo "Deleting the service account..."
gcloud iam service-accounts delete "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" --quiet

echo "Deleting the static IP address..."
gcloud compute addresses delete "${DASHBOARD_NAME_PREFIX}-static-ip" --global --quiet

# TODO: delete docker image?
# Delete the local Docker image
# docker rmi -f gcr.io/cloud-solutions-images/dashboard-demo

gcloud config unset compute/zone
