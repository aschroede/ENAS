#!/bin/bash

# Set destination path (edit if needed)
DEST_DIR="./data"
FILENAME="NAS-Bench-201-v1_1-096897.pth"
URL="https://github.com/D-X-Y/AutoDL-Projects/releases/download/NAS-Bench-201-v1_1/$FILENAME"

# Create the directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download the file
echo "Downloading NAS-Bench-201 to $DEST_DIR/$FILENAME..."
wget -O "$DEST_DIR/$FILENAME" "$URL"

echo "Download complete."
