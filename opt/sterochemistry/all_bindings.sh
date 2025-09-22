#!/bin/bash
set -ex
for binding in eta2 eta6 center2 auto auto6; do
    ./evaluate.py run --binding $binding "$1" > results_${binding}.json
done
