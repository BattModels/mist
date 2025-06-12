#!/bin/bash
# Usage: verify_qm9.sh <DB_PATH> [N]
# Select N (or all) molecules from DB_PATH and runs qm9 calcs with qmist

DB_PATH="$1"
N="$2"

if [ -z "$DB_PATH" ]; then
  echo "Usage: $0 <DB_PATH> [N]"
  exit 1
fi

QUERY="select smiles from molecules"

if [ -n "$N" ]; then
  litecli -e "$QUERY" "$DB_PATH/merged.sqlite" | shuf -n "$N"
else
  litecli -e "$QUERY" "$DB_PATH/merged.sqlite"
fi | xargs -I{} sbatch ../qmist/submit.sh --output "$(realpath "$DB_PATH")/qm9/{InChIKey}.json" {}

