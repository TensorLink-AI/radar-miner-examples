#!/usr/bin/env bash
# Wrapper script for starting the miner with DNS retry logic.
# The miner crashes on startup if DNS resolution fails (e.g., transient
# network issues). This script waits until DNS is working before launching
# neuron.py, preventing rapid crash-restart loops under PM2.
#
# Usage:
#   ./start_miner.sh --agent_dir agents/frontier_sniper/ \
#       --wallet.name miner1 --netuid <N> --subtensor.network <network>

set -euo pipefail

MAX_DNS_RETRIES=10
DNS_CHECK_HOST="entrypoint-finney.opentensor.ai"

wait_for_dns() {
    local attempt=1
    local delay=2

    while [ "$attempt" -le "$MAX_DNS_RETRIES" ]; do
        if getent hosts "$DNS_CHECK_HOST" > /dev/null 2>&1; then
            echo "[start_miner] DNS resolution OK"
            return 0
        fi

        echo "[start_miner] DNS resolution failed (attempt $attempt/$MAX_DNS_RETRIES), retrying in ${delay}s..."
        sleep "$delay"

        attempt=$((attempt + 1))
        # Exponential backoff, capped at 60s
        delay=$((delay * 2))
        if [ "$delay" -gt 60 ]; then
            delay=60
        fi
    done

    echo "[start_miner] WARNING: DNS still failing after $MAX_DNS_RETRIES attempts, starting miner anyway"
    return 0
}

wait_for_dns

exec python miner/neuron.py "$@"
