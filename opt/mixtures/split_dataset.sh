#!/bin/bash
input_file=$1
output_dir=$2
set -ex

uv run python -m electrolyte_fm.data_modules.mixture_dataset \
    $input_file "${output_dir}/random" \
    --split random \
    --num-shards 8 \

for split_type in k-compound k-compound-strict; do
    uv run python -m electrolyte_fm.data_modules.mixture_dataset \
        $input_file \
        "${output_dir}/${split_type}" \
        --split $split_type \
        --num-shards 8
done
