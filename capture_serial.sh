#!/bin/bash
# Serial Monitor Output Capture Script
# Usage: ./capture_serial.sh [output_filename]

# Default filename with timestamp
OUTPUT_FILE="${1:-serial_output_$(date +%Y%m%d_%H%M%S).txt}"

echo "================================================"
echo "  GAINS Serial Monitor Capture"
echo "================================================"
echo "Output file: $OUTPUT_FILE"
echo "Press Ctrl+C to stop capturing"
echo "================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Move to logs directory
OUTPUT_PATH="logs/$OUTPUT_FILE"

# Capture serial output with timestamp
pio device monitor | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $line"
done | tee "$OUTPUT_PATH"

echo ""
echo "================================================"
echo "Capture saved to: $OUTPUT_PATH"
echo "================================================"
