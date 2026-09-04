#!/usr/bin/env bash
# Supply the benchmark-transfer password to OpenSSH's SSH_ASKPASS mechanism.
# This helper deliberately reads only an inherited environment variable; the
# secret remains in the ignored repository environment file, not in source code.

set -euo pipefail

: "${BENCHMARK_COPY_SSH_PASSWORD:?BENCHMARK_COPY_SSH_PASSWORD is required}"
printf '%s\n' "$BENCHMARK_COPY_SSH_PASSWORD"
