#!/usr/bin/env bash
# Build the image, push it to Azure Container Registry (ACR) and run it on Azure Container Instances (ACI).
# Prerequisites: Azure CLI installed, `az login` done, and model/ exported by the notebook.
# Usage (from anywhere):   bash deploy/deploy_aci.sh
# Override any setting:    LOCATION=westeurope BUILD_MODE=local bash deploy/deploy_aci.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-mlops-heart}"
LOCATION="${LOCATION:-germanywestcentral}"           # change if your subscription restricts regions
ACR_NAME="${ACR_NAME:-acrheart$RANDOM}"              # globally unique, lowercase letters and digits only
IMAGE="${IMAGE:-heart-api:v1}"
ACI_NAME="${ACI_NAME:-heart-api}"
DNS_LABEL="${DNS_LABEL:-heart-api-$RANDOM}"          # gives <label>.<region>.azurecontainer.io
BUILD_MODE="${BUILD_MODE:-local}"                    # cloud = az acr build | local = docker build + push
IDENTITY_NAME="${IDENTITY_NAME:-id-heart-api}"       # user-assigned managed identity name

[ -f model/MLmodel ] || { echo "model/ is empty: run the notebook's export step first."; exit 1; }

echo ">> Registering resource providers (first time only, can take a minute)"
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.ContainerInstance --wait

echo ">> Resource group $RESOURCE_GROUP in $LOCATION"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo ">> Container registry $ACR_NAME"
# --admin-enabled false: we authenticate with a managed identity, admin creds are not needed.
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled false --output none
LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
ACR_ID=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv | tr -d '\r')

if [ "$BUILD_MODE" = "cloud" ]; then
  echo ">> Building the image in Azure (no local Docker needed)"
  az acr build --registry "$ACR_NAME" --image "$IMAGE" .
else
  echo ">> Building locally for linux/amd64 and pushing"
  docker build --platform linux/amd64 -t "$LOGIN_SERVER/$IMAGE" .
  az acr login --name "$ACR_NAME"
  docker push "$LOGIN_SERVER/$IMAGE"
fi

echo ">> Ensuring user-assigned managed identity $IDENTITY_NAME exists"
# Idempotent: create if missing, otherwise reuse.
az identity show --resource-group "$RESOURCE_GROUP" --name "$IDENTITY_NAME" --output none 2>/dev/null \
  || az identity create --resource-group "$RESOURCE_GROUP" --name "$IDENTITY_NAME" --output none

IDENTITY_ID=$(az identity show --resource-group "$RESOURCE_GROUP" --name "$IDENTITY_NAME" --query id -o tsv | tr -d '\r')
SPID=$(az identity show --resource-group "$RESOURCE_GROUP" --name "$IDENTITY_NAME" --query principalId -o tsv | tr -d '\r')

echo ">> Granting AcrPull on $ACR_NAME to $IDENTITY_NAME"
# Idempotent: ignores "role assignment already exists" errors.
az role assignment create --assignee "$SPID" --scope "$ACR_ID" --role AcrPull --output none 2>/dev/null || true

# RBAC propagation to ACR can take a minute; give it a short wait.
echo ">> Waiting for role assignment to propagate..."
sleep 30

echo ">> Creating container instance $ACI_NAME"
az container create \
  --resource-group "$RESOURCE_GROUP" --name "$ACI_NAME" \
  --image "$LOGIN_SERVER/$IMAGE" --os-type Linux --cpu 1 --memory 1.5 \
  --acr-identity "$IDENTITY_ID" --assign-identity "$IDENTITY_ID" \
  --ip-address Public --ports 8000 --dns-name-label "$DNS_LABEL" \
  --restart-policy OnFailure --output none

FQDN=$(az container show --resource-group "$RESOURCE_GROUP" --name "$ACI_NAME" --query ipAddress.fqdn -o tsv | tr -d '\r')
cat <<MSG

Deployed. Save these names: RESOURCE_GROUP=$RESOURCE_GROUP ACR_NAME=$ACR_NAME ACI_NAME=$ACI_NAME IDENTITY_NAME=$IDENTITY_NAME
  Form    : http://$FQDN:8000
  Swagger : http://$FQDN:8000/docs
  Test    : python deploy/test_endpoint.py http://$FQDN:8000
  Logs    : az container logs -g $RESOURCE_GROUP -n $ACI_NAME
  Delete everything when done (stops billing): az group delete -n $RESOURCE_GROUP --yes --no-wait
MSG