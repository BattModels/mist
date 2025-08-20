#!/bin/bash
# Usage: verify_qm9.sh <DB_PATH> [name] [N] [-- <extra_args>]
# Select N (or all) molecules from DB_PATH and run qm9 calcs with qmist.
# All arguments after '--' are passed to submit.sh.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <DB_PATH> [name] [N] [-- <extra_args>]" >&2
  exit 1
fi

DB_PATH="$1"; shift

OUTNAME="qm9"
N=""

# Look ahead: first optional may be a name (not purely digits, not `--`)
if [[ $# -gt 0 && "$1" != "--" && ! "$1" =~ ^[0-9]+$ ]]; then
  OUTNAME="$1"
  shift
fi

# Next optional may be N (digits)
if [[ $# -gt 0 && "$1" != "--" && "$1" =~ ^[0-9]+$ ]]; then
  N="$1"
  shift
fi

# Swallow literal '--' if present, then capture extra args
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi
EXTRA_ARGS=( "$@" )

# Ensure output directory exists
OUTDIR="$(realpath "$DB_PATH")/$OUTNAME"
mkdir -p "$OUTDIR"

QUERY="select smiles from molecules"

if [ -n "$N" ]; then
  litecli -e "$QUERY" "$DB_PATH/merged.sqlite" | shuf -n "$N"
else
  litecli -e "$QUERY" "$DB_PATH/merged.sqlite"
fi | xargs -I{} sbatch ../qmist/submit.sh \
  --output "$(realpath "$DB_PATH")/qm9_conf/{InChIKey}.json" \
  "${EXTRA_ARGS[@]}" \
  {}
