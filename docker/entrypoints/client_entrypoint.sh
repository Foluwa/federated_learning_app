#!/bin/bash

# Extract command-line arguments
SERVER_ADDRESS=$1
CLIENT_ID=$2

# Execute the Flower client application with the provided arguments
python3 -m adaptive_federated_healthcare.client.app --server-address "$SERVER_ADDRESS" --cid "$CLIENT_ID"