FROM python:3.9-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install "poetry==1.1.13"
RUN poetry install --no-dev --no-root
COPY adaptive_federated_healthcare ./adaptive_federated_healthcare
COPY docker/entrypoints/server_entrypoint.sh /app/docker/entrypoints/
EXPOSE 8080
ENTRYPOINT ["/app/docker/entrypoints/server_entrypoint.sh"]
