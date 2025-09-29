#!/bin/bash
set -euo pipefail

echo "Looking for notebooks under nbs/core..."

# Populate an array of notebooks (or empty array if none found)
mapfile -t notebooks < <(git ls-files 'nbs/core/*.ipynb' || true)

# number of notebooks found (useful for logging/conditional checks)
count_found_notebooks=${#notebooks[@]}

if [[ ${count_found_notebooks} -eq 0 ]]; then
    echo "No notebooks found in nbs/core; nothing to run."
    exit 0
else
    echo "Found ${count_found_notebooks} notebook(s) in nbs/core."

    # list the detected notebooks for clarity
    echo "Detected notebooks:"
    for nb in "${notebooks[@]}"; do
        echo "  - ${nb}"
    done
fi

for nb in "${notebooks[@]}"; do
    echo "----- Executing: ${nb}"
    uv run jupyter execute "${nb}" --timeout=600
    echo "----- Finished: ${nb}"
done
