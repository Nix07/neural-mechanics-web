#!/bin/bash
# Upload papers to baulab.info server

PAPERS_DIR="$(dirname "$0")/../papers"
REMOTE_HOST="baulab.info"
REMOTE_PATH="/srv/baulab/www/files/neural-mechanics/papers"

echo "Uploading papers from $PAPERS_DIR to $REMOTE_HOST:$REMOTE_PATH"

# Create remote directory if it doesn't exist
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PATH"

# Upload all PDFs
scp "$PAPERS_DIR"/*.pdf "$REMOTE_HOST:$REMOTE_PATH/"

echo "Done."
