#!/usr/bin/env bash
# Runs the test suite and every experiment of the replication of
# Egami (2026), "Loss-Given-Default Modeling by Post-Last Passage Time
# Process".
set -uo pipefail

python -m pytest -m "not market_data"

for exp in exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8; do
    if [ -f "exp/${exp}.py" ]; then
        echo "=== running exp.${exp} ==="
        python -m "exp.${exp}"
    fi
done
