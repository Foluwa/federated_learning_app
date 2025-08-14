#!/bin/bash
SERVER_ADDRESS=$1
CLIENT_ID=$2
python3 -m adaptive_federated_healthcare.client.app "$SERVER_ADDRESS" "$CLIENT_ID"
