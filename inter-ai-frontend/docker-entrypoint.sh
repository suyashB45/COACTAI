#!/bin/sh
set -e

CERT_PATH="/etc/letsencrypt/live/coact-ai.com/fullchain.pem"
KEY_PATH="/etc/letsencrypt/live/coact-ai.com/privkey.pem"
NGINX_CONF="/etc/nginx/conf.d/default.conf"

# The config may be mounted read-only, so copy to a writable location
cp "$NGINX_CONF" /tmp/default.conf
sed -i 's/\r$//' /tmp/default.conf

if [ ! -f "$CERT_PATH" ] || [ ! -f "$KEY_PATH" ]; then
    echo "SSL certificates not found. Disabling HTTPS server block..."
    # Remove everything from the HTTPS server block to end of file
    sed -i '/^# HTTPS Server/,$d' /tmp/default.conf
    echo "HTTPS disabled. Run certbot to enable SSL."
else
    echo "SSL certificates found. HTTPS enabled."
fi

# Replace the config with our processed version
cp /tmp/default.conf "$NGINX_CONF" 2>/dev/null || {
    # If default.conf is read-only (volume mount), use nginx include workaround
    rm -f /etc/nginx/conf.d/default.conf 2>/dev/null || true
    mv /tmp/default.conf /etc/nginx/conf.d/app.conf
}

exec nginx -g "daemon off;"
