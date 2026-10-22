#!/usr/bin/env bash
# Shared HTTPS proxy for phones on the local network (Mac and Linux).
set -euo pipefail
cd "$(dirname "$0")"
[ ! -f .gpu_env ] || . ./.gpu_env
exec .venv/bin/python windows/lan_proxy.py "$@"
