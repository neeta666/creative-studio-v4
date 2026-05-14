#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$APP_DIR"

if [[ ! -f .env ]]; then
  echo ".env file is missing in $APP_DIR" >&2
  exit 1
fi

git fetch origin
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  git checkout "$BRANCH"
fi

git reset --hard "origin/$BRANCH"
npm ci
npm run build
sudo systemctl restart creative-studio
sudo systemctl status creative-studio --no-pager -l
