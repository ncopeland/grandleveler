#!/bin/bash
# GrandLeveler Bot Startup Script

# Change to the script directory
cd "$(dirname "$0")"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "grandleveler.conf" ]; then
    echo "Error: grandleveler.conf not found"
    echo "Please create the configuration file first"
    exit 1
fi

# Check if log file exists and is readable
LOG_FILE=$(grep "log_file" grandleveler.conf | cut -d'=' -f2 | tr -d ' ')
if [ ! -f "$LOG_FILE" ]; then
    echo "Warning: Log file $LOG_FILE not found"
    echo "The bot will start but may not be able to parse reputation scores"
fi

# Make the bot executable
chmod +x grandleveler_bot.py

# Start the bot
echo "Starting GrandLeveler bot..."
echo "Press Ctrl+C to stop the bot"
echo "Logs will be written to grandleveler.log"
echo ""

python3 grandleveler_bot.py
