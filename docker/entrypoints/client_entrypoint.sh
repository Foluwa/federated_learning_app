#!/bin/sh
SERVER_ADDRESS="$1"
CLIENT_ID="$2"
exec python3 -m adaptive_federated_healthcare.client.app "$SERVER_ADDRESS" "$CLIENT_ID"


# set -eu
# SERVER_ADDRESS=${1:-server:8080}
# CLIENT_ID=${2:?CLIENT_ID argument missing}
# exec python3 -m adaptive_federated_healthcare.client.app "$SERVER_ADDRESS" "$CLIENT_ID"
