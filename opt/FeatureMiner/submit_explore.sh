#!/bin/bash
for ckpt in $(find ../../linear-probes/ -maxdepth 1 -mindepth 1 -type d); do
    if [ ! -f "${ckpt}/linear_probes.jld2" ]; then
        echo "Submitting ${ckpt}"
        sbatch ./run_explore.sh "${ckpt}"
    fi
done
