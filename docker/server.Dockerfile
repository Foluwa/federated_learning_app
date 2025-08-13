# File: docker/client.Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY adaptive_federated_healthcare ./adaptive_federated_healthcare
COPY docker/entrypoints/client_entrypoint.sh /app/docker/entrypoints/
EXPOSE 8080
ARG CLIENT_ID=0
ENV CLIENT_ID=$CLIENT_ID
ENTRYPOINT ["/app/docker/entrypoints/client_entrypoint.sh"]