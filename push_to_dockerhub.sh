#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dockerhub_username> [image_name]"
    echo "Example: $0 myuser camera-agent"
    exit 1
fi

USERNAME=$1
IMAGE_NAME=${2:-camera-agent}
FULL_IMAGE_NAME="$USERNAME/$IMAGE_NAME:latest"

echo "Building image: $FULL_IMAGE_NAME"
docker build -t "$FULL_IMAGE_NAME" .

echo "Pushing image to Docker Hub..."
echo "Note: Make sure you have run 'docker login' first."
docker push "$FULL_IMAGE_NAME"

echo "Successfully pushed $FULL_IMAGE_NAME"
