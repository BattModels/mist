#!/bin/bash
input_file=$1
output_dir=$2
set -ex
for split_type in k-compound k-compound-strict; do
    for split_idx in 0 1 2 3 4; do
        uv run python -m electrolyte_fm.data_modules.mixture_dataset $input_file "${output_dir}/${split_type}-${split_idx}" --split $split_type --split-idx $split_idx
    done
done
