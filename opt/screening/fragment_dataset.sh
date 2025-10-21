#!/bin/bash

zinc=$(find /lustre/fs0/shared/zinc_v1/data -name '*.txt' -print0)

process_zinc_shard() {
    smi_file=$1
    frag_file="zinc_fragment/$(basename $smi_file).frag"
    mkdir -p $(dirname $frag_file)
    echo "processing $smi_file"
    uv run python -m src.fragment $smi_file $frag_file
}
export -f process_zinc_shard

find /lustre/fs0/shared/zinc_v1/data -name '*.txt' -print0 | \
    xargs -0 -P 64 -I {} bash -c 'process_zinc_shard "$@"' _ {}
