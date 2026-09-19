# The front end: built with node, served by nginx.
#
#   docker build -t frontier-ui ./frontend
#
# `docker compose up` builds this as the `frontend` service and points it at the
# API over the compose network; see ../compose.yaml.

ARG NODE_VERSION=26

# --- build ---------------------------------------------------------------
FROM node:${NODE_VERSION}-slim AS builder

WORKDIR /app

# package.json and the lockfile first: this layer survives a source edit, so a
# rebuild after changing a component does not reinstall node_modules.
COPY package.json package-lock.json ./
# `npm ci` installs exactly the lockfile, and fails if the two disagree.
RUN npm ci

COPY . .
# tsc first: a type error must fail the image build, not reach a browser.
RUN npx tsc --noEmit && npx vite build

# --- runtime -------------------------------------------------------------
# The unprivileged image listens on 8080 and runs as uid 101, which matches the
# API image's refusal to run anything as root.
FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=4s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/ || exit 1
