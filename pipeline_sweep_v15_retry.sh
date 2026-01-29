#!/bin/bash

MAX_RETRIES=1000
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt #$((COUNT+1)) - Starting pipeline_sweep_v15.py"
    python pipeline_sweep_v15.py

    EXIT_CODE=$?
    echo "Run exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Script finished successfully. Exiting loop."
        break
    fi

    COUNT=$((COUNT+1))
    echo "Restarting... ($COUNT/$MAX_RETRIES)"
done

echo "Done after $COUNT attempts."
