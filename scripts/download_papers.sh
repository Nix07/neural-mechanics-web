#!/bin/bash
# Download papers from baulab.info server

PAPERS_DIR="$(dirname "$0")/../papers"
REMOTE_HOST="baulab.info"
REMOTE_PATH="/srv/baulab/www/files/neural-mechanics/papers"

echo "Downloading papers from $REMOTE_HOST:$REMOTE_PATH to $PAPERS_DIR"

# Create local directory if it doesn't exist
mkdir -p "$PAPERS_DIR"

# Download all PDFs
rsync -avz "$REMOTE_HOST:$REMOTE_PATH/" "$PAPERS_DIR/"

echo "Done."
