# GrandLeveler IRC Bot

A Python IRC bot that automatically levels reputation scores on IRC channels by parsing WeeChat logs and incrementing the lowest reputation users until everyone reaches the same score.

## Features

- **Automatic Reputation Leveling**: Parses WeeChat logs to find current reputation scores and automatically increments the lowest scoring users
- **Online User Detection**: Uses WHOIS commands to check if users are online before attempting to change their reputation
- **Intelligent Timing**: Respects IRC bot cooldowns and schedules leveling attempts appropriately
- **Admin Commands**: Basic IRC administration functionality (join, part, say, restart, status)
- **Flood Control**: Implements 1-second delays between WHOIS commands to prevent server flooding
- **Multi-Network Support**: Configurable for multiple IRC networks

## Requirements

- Python 3.7 or higher
- No external dependencies (uses only Python standard library)
- WeeChat log file with reputation scores
- IRC network access (tested with Rizon)

## Installation

1. Clone or download the bot files
2. Configure the bot using `grandleveler.conf`
3. Make the bot executable: `chmod +x grandleveler_bot.py`
4. Run the bot: `python3 grandleveler_bot.py`

## Configuration

Edit `grandleveler.conf` to configure the bot:

```ini
[DEFAULT]
# Log file to parse for reputation scores
log_file = /home/boliver/.local/share/weechat/logs/irc.rizon.#computertech.weechatlog

# Timing configuration
reputation_cooldown = 3600  # 1 hour in seconds
leveling_interval = 4200    # 70 minutes in seconds
flood_control_delay = 1     # 1 second between WHOIS commands

[network:rizon]
server = irc.rizon.net/6667
ssl = off
bot_nick = GrandLeveler,GL,GrandL,Leveler
ident = gleveler
channel = #computertech
perform = PRIVMSG nickserv :identify YOUR_PASSWORD_HERE ; PRIVMSG Boliver :I am here
owner = Boliver
admin = loulan,nevodka
```

## Usage

### Starting the Bot

```bash
python3 grandleveler_bot.py
```

The bot will:
1. Connect to the configured IRC network
2. Join the specified channel(s)
3. Parse the log file for current reputation scores
4. Begin the reputation leveling process

### Admin Commands

The following commands are available to bot administrators:

- `!join #channel` - Join a channel
- `!part #channel [reason]` - Leave a channel with optional reason
- `!say #channel message` - Send a message to a channel
- `!restart` - Restart the bot
- `!status` - Show current bot status and next leveling time

### How It Works

1. **Log Parsing**: The bot reads the WeeChat log file and extracts current reputation scores using regex pattern matching
2. **Target Selection**: Finds the user with the lowest reputation score
3. **Online Check**: Uses WHOIS commands to verify the user is online
4. **Reputation Increment**: Sends a `username++` message to the channel
5. **Cooldown Handling**: Monitors for Yuzu bot cooldown notices and adjusts timing accordingly
6. **Continuous Process**: Repeats the process every 70 minutes until all users reach the same score

### Log File Format

The bot expects the log file to contain reputation scores in the format:
```
username score
username2 score2
```

For example:
```
peorth 359
h4 193
wez 131
updog 49
computertech -424
```

## Troubleshooting

### Common Issues

1. **Bot won't connect**: Check server settings and network connectivity
2. **Can't parse log file**: Verify the log file path and format
3. **WHOIS timeouts**: Users may be offline or network may be slow
4. **Permission denied**: Ensure the bot has read access to the log file

### Logs

The bot creates detailed logs in `grandleveler.log` including:
- Connection status
- WHOIS responses
- Reputation leveling attempts
- Error messages

### Network Issues

If the bot disconnects:
- It will attempt to reconnect automatically
- Check network configuration
- Verify IRC server is accessible

## Security Notes

- The bot requires NickServ identification on most networks
- Admin commands are restricted to configured administrators
- The bot implements flood control to avoid server limits
- No sensitive data is stored locally

## License

This project is based on the DuckHunt bot architecture and is released under GPL v2.

## Contributing

Feel free to submit issues and enhancement requests!
