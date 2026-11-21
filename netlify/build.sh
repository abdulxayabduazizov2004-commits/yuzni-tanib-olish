#!/bin/sh
# Netlify build: web/ ni dist/ ga ko'chiradi va backendga proksi qoidalarini (_redirects) yozadi.
set -eu

# 1) Netlify sozlamalaridagi BACKEND_URL; 2) bo'lmasa - netlify/backend_url.txt fayli
if [ -z "${BACKEND_URL:-}" ] && [ -f netlify/backend_url.txt ]; then
  BACKEND_URL=$(tr -d ' \r\n' < netlify/backend_url.txt)
  echo "BACKEND_URL netlify/backend_url.txt dan olindi"
fi
if [ -z "${BACKEND_URL:-}" ]; then
  echo "XATO: BACKEND_URL muhit o'zgaruvchisi berilmagan."
  echo "Netlify -> Site configuration -> Environment variables -> BACKEND_URL = https://<backend-manzil>"
  exit 1
fi
BACKEND_URL="${BACKEND_URL%/}"
case "$BACKEND_URL" in
  https://*) ;;
  *) echo "XATO: BACKEND_URL https:// bilan boshlanishi kerak (hozir: $BACKEND_URL)"; exit 1 ;;
esac

mkdir -p dist
cp -R web/. dist/

# Qoidalar tartib bo'yicha tekshiriladi. 200 = rewrite/proksi (manzil satri o'zgarmaydi).
cat > dist/_redirects <<EOF
/api/*      ${BACKEND_URL}/api/:splat   200!
/healthz    ${BACKEND_URL}/healthz      200!
/static/*   /:splat                     200
/*          /index.html                 200
EOF

echo "Tayyor: dist/ -> backend: ${BACKEND_URL}"
cat dist/_redirects
