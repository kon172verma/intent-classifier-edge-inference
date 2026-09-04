#!/usr/bin/env bash
# Copy a local model-version directory to an edge device, one file per scp call.
#
# The destination is a repository root.  For example:
#   scripts/copy_profile_artifacts.sh \
#     --remote quad@192.168.2.154 \
#     --destination /mnt/sdcard/intent-classifier-edge-inference \
#     --verify
#
# Re-running the command is safe: regular files with matching byte counts are
# skipped.  With --verify, matching files are also checked using SHA-256.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() {
  echo "copy_profile_artifacts: error: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  scripts/copy_profile_artifacts.sh \
    --remote USER@HOST \
    --destination REMOTE_REPOSITORY_ROOT \
    [--source models/v2.1] [--verify]

Copies every regular file below the local source directory. Its path beneath
the local models/ directory is preserved remotely; for example,
models/v2.1 is copied to REMOTE_REPOSITORY_ROOT/models/v2.1, while
models/v2.1/SmolLM2-360M is copied to
REMOTE_REPOSITORY_ROOT/models/v2.1/SmolLM2-360M. Each file is sent by a
separate scp call. Existing destination files with the same byte size are
skipped, making interrupted transfers resumable.

Options:
  --remote USER@HOST             SSH destination, for example quad@192.168.2.154.
  --destination PATH             Remote repository root, not its models directory.
  --source PATH                  Local model-version directory (default: models/v2.1).
  --verify                       SHA-256-check both skipped and copied files.
  -h, --help                     Show this message.

Authentication uses BENCHMARK_COPY_SSH_PASSWORD from the ignored .env.copy
file. The password is passed to SSH through an askpass helper and is never
printed.
EOF
}

load_copy_password() {
  local env_file="$REPO_ROOT/.env.copy"
  local line
  local password=""

  [[ -r "$env_file" ]] || die "missing readable environment file: $env_file"
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      BENCHMARK_COPY_SSH_PASSWORD=*)
        password="${line#BENCHMARK_COPY_SSH_PASSWORD=}"
        password="${password%$'\r'}"
        break
        ;;
    esac
  done <"$env_file"
  [[ -n "$password" ]] || die "BENCHMARK_COPY_SSH_PASSWORD is missing from $env_file"
  export BENCHMARK_COPY_SSH_PASSWORD="$password"
}

remote=""
destination=""
source_root="$REPO_ROOT/models/v2.1"
verify=false

while (($#)); do
  case "$1" in
    --remote)
      (($# >= 2)) || die "--remote requires USER@HOST"
      remote="$2"
      shift 2
      ;;
    --destination)
      (($# >= 2)) || die "--destination requires a path"
      destination="$2"
      shift 2
      ;;
    --source)
      (($# >= 2)) || die "--source requires a path"
      source_root="$2"
      shift 2
      ;;
    --verify)
      verify=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$remote" ]] || die "--remote is required"
[[ -n "$destination" ]] || die "--destination is required"
[[ "$destination" == /* ]] || die "--destination must be an absolute remote path"
[[ -d "$source_root" ]] || die "source directory does not exist: $source_root"

command -v ssh >/dev/null || die "ssh is required"
command -v scp >/dev/null || die "scp is required"
if "$verify"; then
  command -v shasum >/dev/null || die "shasum is required for --verify"
fi
load_copy_password
askpass_helper="$REPO_ROOT/scripts/ssh_askpass_from_env.sh"
[[ -x "$askpass_helper" ]] || die "SSH askpass helper is not executable: $askpass_helper"
export SSH_ASKPASS="$askpass_helper"
export SSH_ASKPASS_REQUIRE=force
export DISPLAY="${DISPLAY:-benchmark-copy}"

source_root="$(cd "$source_root" && pwd)"
local_models_root="$REPO_ROOT/models"
case "$source_root/" in
  "$local_models_root/"*) ;;
  *) die "--source must be inside $local_models_root" ;;
esac
source_relative="${source_root#"$local_models_root"/}"

# Reject quote characters so the remote shell commands below remain unambiguous.
[[ "$destination" != *"'"* && "$destination" != *$'\n'* ]] || die "unsupported characters in --destination"

file_list="$(mktemp "${TMPDIR:-/tmp}/copy-profile-artifacts.XXXXXX")"
# macOS can set TMPDIR to a path too long for OpenSSH Unix sockets. Keep the
# control socket under /tmp, where the %C hash keeps it unique per destination.
control_path="/tmp/copy-profile-artifacts-${UID}-%C"
ssh_options=(
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o "ControlPath=$control_path"
  -o StrictHostKeyChecking=accept-new
)

cleanup() {
  ssh "${ssh_options[@]}" -O exit "$remote" >/dev/null 2>&1 || true
  rm -f "$file_list"
}
trap cleanup EXIT

find "$source_root" -type f -print0 >"$file_list"

[[ -s "$file_list" ]] || die "the source directory contains no regular files"

echo "Opening one SSH control connection to $remote."
ssh "${ssh_options[@]}" -MNf "$remote" </dev/null

remote_root="$destination/models/$source_relative"
ssh "${ssh_options[@]}" "$remote" "mkdir -p -- '$remote_root'" </dev/null

file_size() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

copied=0
skipped=0

while IFS= read -r -d '' file; do
  relative="${file#"$source_root"/}"
  remote_path="$remote_root/$relative"
  remote_dir="${remote_path%/*}"
  local_bytes="$(file_size "$file")"
  remote_bytes="$(ssh "${ssh_options[@]}" "$remote" "if [ -f '$remote_path' ]; then wc -c < '$remote_path'; else true; fi" </dev/null)"

  if [[ "$remote_bytes" == "$local_bytes" ]]; then
    if "$verify"; then
      local_hash="$(sha256 "$file")"
      remote_hash="$(ssh "${ssh_options[@]}" "$remote" "sha256sum '$remote_path' | cut -d ' ' -f 1" </dev/null)"
      if [[ "$local_hash" == "$remote_hash" ]]; then
        echo "SKIP  $relative"
        ((skipped += 1))
        continue
      fi
      echo "RETRY $relative (same size, SHA-256 mismatch)"
    else
      echo "SKIP  $relative"
      ((skipped += 1))
      continue
    fi
  fi

  ssh "${ssh_options[@]}" "$remote" "mkdir -p -- '$remote_dir'" </dev/null
  echo "COPY  $relative"
  scp "${ssh_options[@]}" -p "$file" "$remote:$remote_path" </dev/null

  if "$verify"; then
    local_hash="$(sha256 "$file")"
    remote_hash="$(ssh "${ssh_options[@]}" "$remote" "sha256sum '$remote_path' | cut -d ' ' -f 1" </dev/null)"
    [[ "$local_hash" == "$remote_hash" ]] || die "SHA-256 verification failed: $relative"
  fi
  ((copied += 1))
done <"$file_list"

echo "Transfer complete: copied $copied file(s), skipped $skipped already-complete file(s)."
