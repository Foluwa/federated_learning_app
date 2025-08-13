#!/bin/bash
echo "Starting Flower client with ID $CLIENT_ID..."
python -m adaptive_federated_healthcare.client.app server:8080 $CLIENT_ID
