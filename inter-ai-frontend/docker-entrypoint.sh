#!/bin/sh
set -e

CERT_PATH="/etc/letsencrypt/live/coact-ai.com/fullchain.pem"
KEY_PATH="/etc/letsencrypt/live/coact-ai.com/privkey.pem"
NGINX_CONF="/etc/nginx/conf.d/default.conf"

if [ ! -f "$CERT_PATH" ] || [ ! -f "$KEY_PATH" ]; then
    echo "SSL certificates not found. Writing HTTP-only config..."

    # Overwrite the config entirely with an HTTP-only version
    cat > "$NGINX_CONF" <<'HTTPCONF'
server {
    listen 80;
    server_name coactai.centralindia.cloudapp.azure.com coact-ai.com www.coact-ai.com;

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_connect_timeout 60s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;

        client_max_body_size 50M;
    }
}
HTTPCONF

    echo "HTTPS disabled. HTTP-only config written."
    echo "--- Active nginx config ---"
    cat "$NGINX_CONF"
    echo "--- End config ---"
else
    echo "SSL certificates found. HTTPS enabled."
fi

exec nginx -g "daemon off;"
