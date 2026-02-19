#!/bin/sh
set -e

CERT_PATH="/etc/letsencrypt/live/coact-ai.com/fullchain.pem"
KEY_PATH="/etc/letsencrypt/live/coact-ai.com/privkey.pem"

# If real SSL certs don't exist, generate a self-signed placeholder
# so nginx can start. Replace with real certs via certbot later.
if [ ! -f "$CERT_PATH" ] || [ ! -f "$KEY_PATH" ]; then
    echo "SSL certificates not found. Generating self-signed placeholder..."
    mkdir -p /etc/letsencrypt/live/coact-ai.com
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$KEY_PATH" \
        -out "$CERT_PATH" \
        -subj "/CN=localhost" 2>/dev/null
    echo "Self-signed certificate generated. Replace with real certs using certbot."
else
    echo "SSL certificates found."
fi

exec nginx -g "daemon off;"
