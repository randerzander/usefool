#!/bin/bash
# 1. Let container create default settings.yml
docker run --name temp-searxng --rm \
  -p 8081:8080 \
  -v "$(pwd)/config:/etc/searxng" \
  docker.io/searxng/searxng:latest sleep 5

# 2. Now settings.yml exists and has been generated
echo "Default settings.yml generated. Contents:"
cat config/settings.yml | head -20

# 3. Stop any running container
docker stop searxng 2>/dev/null || true

