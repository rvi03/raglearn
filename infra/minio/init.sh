#!/bin/sh
# Provision the ingestion bucket and bind it to the Redpanda notification target.
#
# Run once after MinIO starts (as a one-shot init container, or by hand with the
# minio/mc image). Idempotent: creating an existing bucket or rule is a no-op, so
# re-running on every `up` is safe.
#
# The "ingest" Kafka target itself is configured on the MinIO server via the
# MINIO_NOTIFY_KAFKA_*_ingest env vars in compose.yaml; here we only create the
# bucket and attach the per-bucket event rule that publishes to it.
set -eu

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://finrag-minio:9000}"
MINIO_USER="${MINIO_ROOT_USER:-finrag}"
MINIO_PASSWORD="${MINIO_ROOT_PASSWORD:-finrag-secret}"
BUCKET="${INGEST_BUCKET:-filings}"

# MinIO may still be coming up; retry the alias until the server answers.
echo "waiting for MinIO at ${MINIO_ENDPOINT} ..."
until mc alias set finrag "${MINIO_ENDPOINT}" "${MINIO_USER}" "${MINIO_PASSWORD}" >/dev/null 2>&1; do
  sleep 1
done

mc mb --ignore-existing "finrag/${BUCKET}"

# Publish object-creation events for this bucket to the Kafka target "ingest".
mc event add --ignore-existing "finrag/${BUCKET}" arn:minio:sqs::ingest:kafka --event put

echo "ready: bucket '${BUCKET}' notifies Kafka target 'ingest' on object upload"
mc event ls "finrag/${BUCKET}"
