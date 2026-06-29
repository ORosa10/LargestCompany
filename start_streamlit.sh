#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8501}"
LOG_FILE="/tmp/largestcompany-streamlit-${PORT}.log"
PROCESS_PATTERN="python -m streamlit run app.py.*--server.port ${PORT}"

if pgrep -f "${PROCESS_PATTERN}" >/dev/null 2>&1; then
    echo "LargestCompany is already running on port ${PORT}."
else
    nohup python -m streamlit run app.py \
        --server.address 0.0.0.0 \
        --server.port "${PORT}" \
        >"${LOG_FILE}" 2>&1 &
    sleep 3
fi

if pgrep -f "${PROCESS_PATTERN}" >/dev/null 2>&1; then
    echo "LargestCompany is running on port ${PORT}."
    echo "Open the PORTS tab and click the globe icon for port ${PORT}."
    echo "Log: ${LOG_FILE}"
else
    echo "Streamlit did not start. Last log lines:"
    tail -n 30 "${LOG_FILE}" || true
    exit 1
fi
