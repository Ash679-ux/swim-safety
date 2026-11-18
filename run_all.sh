#!/usr/bin/env bash
# Run the pipeline steps in order, stopping at the first error.
#
#   ./run_all.sh        run steps 04-12 (the full pipeline)
#   ./run_all.sh 06     start from step 06 (earlier outputs must already exist)
#   ./run_all.sh 09 10  run only steps 09 to 10
#
# Steps 01-03 are the initial data audit and aren't needed to rebuild results.
set -euo pipefail
cd "$(dirname "$0")"

FROM=${1:-04}
TO=${2:-12}
start=$SECONDS

for script in scripts/[0-9][0-9]_*.py; do
    step=$(basename "$script" | cut -c1-2)
    if ((10#$step < 10#$FROM || 10#$step > 10#$TO)); then
        continue
    fi
    echo
    echo "=== $(basename "$script") ==="
    t=$SECONDS
    python "$script"
    echo "--- finished $(basename "$script") in $((SECONDS - t)) s"
done

echo
echo "All done (steps $FROM-$TO) in $(((SECONDS - start) / 60)) min $(((SECONDS - start) % 60)) s"
