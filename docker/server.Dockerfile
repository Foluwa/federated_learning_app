FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY adaptive_federated_healthcare ./adaptive_federated_healthcare
COPY docker/entrypoints/server_entrypoint.sh /app/docker/entrypoints/
EXPOSE 8080
ENTRYPOINT ["/app/docker/entrypoints/server_entrypoint.sh"]
