#!/bin/bash
# Usage: verify_qm9.sh <DB_PATH> [N] [-- <extra_args>]
# Select N (or all) molecules from DB_PATH and runs qm9 calcs with qmist
# All arguments after '--' are passed to submit.sh

DB_PATH="$1"
N="$2"
shift 2

# Capture any remaining arguments to pass through to submit.sh
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  EXTRA_ARGS+=("$1")
  shift
done

if [ -z "$DB_PATH" ]; then
  echo "Usage: $0 <DB_PATH> [N] [-- <extra_args>]"
  exit 1
fi

QUERY="select smiles from molecules"

if [ -n "$N" ]; then
  litecli -e "$QUERY" "$DB_PATH/merged.sqlite" | shuf -n "$N"
else
  litecli -e "$QUERY" "$DB_PATH/merged.sqlite"
fi | xargs -I{} sbatch ../qmist/submit.sh \
  --output "$(realpath "$DB_PATH")/qm9/{InChIKey}.json" \
  "${EXTRA_ARGS[@]}" \
  {}
