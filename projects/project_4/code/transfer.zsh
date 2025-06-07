#!/bin/zsh

# === Config ===
LOCAL_DIR="."
REMOTE_USR="paultalma"
REMOTE_HOST="dlvm-pytorch"
REMOTE_DIR="/home/paultalma/proj4/code"

# === scp ===
echo "Transferring contents of ${LOCAL_DIR} (nonrecursively) to ${REMOTE_DIR}..."
gcloud compute scp "${LOCAL_DIR}/"* "${REMOTE_USR}@${REMOTE_HOST}:${REMOTE_DIR}"
echo "Transfer complete."
