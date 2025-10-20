#!/usr/bin/env python3
"""
GrandLeveler IRC Bot
A bot that automatically levels reputation scores on IRC channels by parsing logs
and incrementing the lowest reputation users until everyone is equal.

Author: Based on DuckHunt bot architecture
License: GPLV2
"""

import asyncio
import socket
import ssl
import time
import re
import json
import os
import configparser
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('grandleveler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ReputationEntry:
    """Represents a user's reputation score"""
    username: str
    score: int

class NetworkConnection:
    """Represents a connection to a single IRC network"""
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.reader = None
        self.writer = None
        self.registered = False
        self.nick = config['bot_nick'].split(',')[0]
        self.altnicks = config['bot_nick'].split(',')[1:] if ',' in config['bot_nick'] else []
        self.channels = set(config['channel'].split(','))
        self.whois_responses = {}
        self.whois_timeout = 60
        self.last_flood_time = 0
        self.flood_delay = float(config.get('flood_control_delay', '1').split('#')[0].strip())
        
    async def connect(self):
        """Establish connection to IRC server"""
        try:
            server_host, server_port = self.config['server'].split('/')
            server_port = int(server_port)
            
            if self.config.get('ssl', 'off').lower() == 'on':
                ssl_context = ssl.create_default_context()
                self.reader, self.writer = await asyncio.open_connection(
                    server_host, server_port, ssl=ssl_context
                )
            else:
                self.reader, self.writer = await asyncio.open_connection(
                    server_host, server_port
                )
            
            logger.info(f"Connected to {self.name} ({server_host}:{server_port})")
            
            # Send initial IRC commands
            await self.send_command(f"NICK {self.nick}")
            await self.send_command(f"USER {self.config['ident']} 0 * :GrandLeveler Bot")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    async def send_command(self, command: str):
        """Send a command to the IRC server"""
        if self.writer:
            self.writer.write(f"{command}\r\n".encode())
            await self.writer.drain()
            logger.debug(f"Sent: {command}")
    
    async def send_message(self, target: str, message: str):
        """Send a PRIVMSG to a channel or user"""
        await self.send_command(f"PRIVMSG {target} :{message}")
    
    async def whois_user(self, username: str) -> bool:
        """Send WHOIS command and return True if user is online"""
        if time.time() - self.last_flood_time < self.flood_delay:
            await asyncio.sleep(self.flood_delay)
        
        # Clear any existing response and set to None
        self.whois_responses[username] = None
        self.whois_responses[username.title()] = None  # Also clear title case
        
        logger.info(f"Sending WHOIS for {username}")
        await self.send_command(f"WHOIS {username}")
        self.last_flood_time = time.time()
        
        # Wait for WHOIS response with proper async yielding
        logger.info(f"Waiting for WHOIS response for {username}, timeout: {self.whois_timeout}s")
        
        start_time = time.time()
        while time.time() - start_time < self.whois_timeout:
            # Check both cases
            if username in self.whois_responses and self.whois_responses[username] is not None:
                result = self.whois_responses[username]
                logger.info(f"WHOIS result for {username}: {result}")
                return result
            if username.title() in self.whois_responses and self.whois_responses[username.title()] is not None:
                result = self.whois_responses[username.title()]
                logger.info(f"WHOIS result for {username}: {result}")
                return result
            
            # Yield control to allow message processing
            await asyncio.sleep(0.1)
        
        logger.info(f"WHOIS timeout for {username} after {self.whois_timeout}s, assuming offline")
        return False

class LogParser:
    """Parses WeeChat logs to extract reputation scores"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        # Use the same regex patterns as reputation_stats.py
        self.rep_increase_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+[+&]\w+\s+(\w+)\s+has\s+increased\s+(\w+)'s\s+reputation\s+score\s+to\s+(-?\d+)", re.I)
        self.rep_decrease_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+[+&]\w+\s+(\w+)\s+has\s+decreased\s+(\w+)'s\s+reputation\s+score\s+to\s+(-?\d+)", re.I)
    
    def parse_reputation_scores(self) -> List[ReputationEntry]:
        """Parse the log file and extract current reputation scores"""
        reputation_entries = []
        
        if not os.path.exists(self.log_file):
            logger.error(f"Log file not found: {self.log_file}")
            return reputation_entries
        
        try:
            # Track last known scores for each user (same logic as reputation_stats.py)
            last_scores = {}
            
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for reputation increase
                    match = self.rep_increase_pattern.search(line)
                    if match:
                        timestamp, user, target, score = match.groups()
                        target = target.lower()
                        score = int(score)
                        
                        # Only update if this is a more recent change for this user
                        if target not in last_scores or timestamp > last_scores[target]['timestamp']:
                            last_scores[target] = {
                                'score': score,
                                'timestamp': timestamp
                            }
                        continue
                    
                    # Check for reputation decrease
                    match = self.rep_decrease_pattern.search(line)
                    if match:
                        timestamp, user, target, score = match.groups()
                        target = target.lower()
                        score = int(score)
                        
                        # Only update if this is a more recent change for this user
                        if target not in last_scores or timestamp > last_scores[target]['timestamp']:
                            last_scores[target] = {
                                'score': score,
                                'timestamp': timestamp
                            }
            
            # Convert to ReputationEntry objects
            for target, score_data in last_scores.items():
                reputation_entries.append(ReputationEntry(
                    username=target, 
                    score=score_data['score']
                ))
            
            # Sort by score (lowest first)
            reputation_entries.sort(key=lambda x: x.score)
            logger.info(f"Parsed {len(reputation_entries)} reputation entries")
            
        except Exception as e:
            logger.error(f"Error parsing log file: {e}")
        
        return reputation_entries

class GrandLevelerBot:
    """Main IRC bot class for reputation leveling"""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        self.networks = {}
        self.log_parser = LogParser(self.config['DEFAULT']['log_file'])
        
        # Timing
        self.reputation_cooldown = int(self.config['DEFAULT'].get('reputation_cooldown', '3600').split('#')[0].strip())
        self.leveling_interval = int(self.config['DEFAULT'].get('leveling_interval', '4200').split('#')[0].strip())
        self.last_reputation_change = 0
        self.next_leveling_time = 0
        
        # State
        self.running = True
        self.current_targets = []
        self.target_index = 0
        self.leveling_started = False  # Flag to track if we've started the first leveling
        self.leveling_in_progress = False  # Flag to prevent multiple simultaneous leveling attempts
        self.leveling_task = None  # Track the current leveling task
        
        # Persistence
        self.state_file = 'grandleveler_state.json'
        self.load_state()
    
    def load_state(self):
        """Load bot state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_reputation_change = state.get('last_reputation_change', 0)
                    self.next_leveling_time = state.get('next_leveling_time', 0)
                    self.leveling_started = state.get('leveling_started', False)
                    logger.info(f"Loaded state: last_change={self.last_reputation_change}, next_leveling={self.next_leveling_time}, started={self.leveling_started}")
            else:
                logger.info("No state file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Reset to defaults
            self.last_reputation_change = 0
            self.next_leveling_time = 0
            self.leveling_started = False
    
    def save_state(self):
        """Save bot state to file"""
        try:
            state = {
                'last_reputation_change': self.last_reputation_change,
                'next_leveling_time': self.next_leveling_time,
                'leveling_started': self.leveling_started
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved state: {state}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
        
    def setup_networks(self):
        """Initialize network connections"""
        for section_name in self.config.sections():
            if section_name.startswith('network:'):
                network_name = section_name.split(':', 1)[1]
                network_config = dict(self.config[section_name])
                self.networks[network_name] = NetworkConnection(network_name, network_config)
                logger.info(f"Configured network: {network_name}")
    
    async def start(self):
        """Start the bot"""
        logger.info("Starting GrandLeveler bot...")
        self.setup_networks()
        
        # Connect to all networks
        for network in self.networks.values():
            if await network.connect():
                await asyncio.sleep(1)  # Small delay between connections
        
        # Start main loop
        await self.main_loop()
    
    async def main_loop(self):
        """Main bot loop"""
        while self.running:
            try:
                # Process messages from all networks
                for network in self.networks.values():
                    if network.reader:
                        await self.process_network_messages(network)
                
                # Check if it's time for reputation leveling (only after first leveling has started)
                if self.leveling_started:
                    current_time = time.time()
                    if current_time >= self.next_leveling_time:
                        # Only start a new task if one isn't already running
                        if self.leveling_task is None or self.leveling_task.done():
                            # Run reputation leveling as a separate task to avoid blocking the main loop
                            self.leveling_task = asyncio.create_task(self.perform_reputation_leveling())
                        # Don't override next_leveling_time here - let perform_reputation_leveling set it
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        await self.shutdown()
    
    async def process_network_messages(self, network: NetworkConnection):
        """Process incoming messages from a network"""
        try:
            while True:
                line = await asyncio.wait_for(network.reader.readline(), timeout=0.1)
                if not line:
                    break
                
                message = line.decode('utf-8', errors='ignore').strip()
                logger.info(f"Received: {message}")
                
                await self.handle_message(network, message)
                
        except asyncio.TimeoutError:
            pass  # No message available
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
    
    async def handle_message(self, network: NetworkConnection, message: str):
        """Handle incoming IRC messages"""
        parts = message.split()
        if not parts:
            return
        
        # Handle server responses
        if parts[0].startswith(':'):
            prefix = parts[0][1:]
            command = parts[1] if len(parts) > 1 else ''
            params = parts[2:] if len(parts) > 2 else []
        else:
            command = parts[0]
            params = parts[1:]
        
        # Handle different IRC commands
        if command == '001':  # Welcome message
            network.registered = True
            logger.info(f"Registered on {network.name}")
            
            # Join channels
            for channel in network.channels:
                await network.send_command(f"JOIN {channel}")
                logger.info(f"Sent JOIN command for {channel}")
            
            # Execute perform commands
            if 'perform' in network.config:
                perform_commands = network.config['perform'].split(' ; ')
                for cmd in perform_commands:
                    if cmd.strip():
                        logger.info(f"Executing perform command: {cmd.strip()}")
                        await network.send_command(cmd.strip())
            
            # Don't start reputation leveling here - wait for channel join confirmation
            
        elif command == 'PING':
            # Respond to PING to stay connected
            pong_msg = params[0] if params else ''
            await network.send_command(f"PONG {pong_msg}")
            logger.info(f"Responded to PING with PONG {pong_msg}")
            
        elif command == 'JOIN':
            # Handle JOIN confirmations
            if len(params) >= 1:
                channel = params[0][1:] if params[0].startswith(':') else params[0]
                logger.info(f"Successfully joined {channel}")
                # Don't start reputation leveling here - wait for NickServ auth
            
        elif command == '311':  # WHOIS response - user is online
            if len(params) >= 2:
                username = params[1]  # Keep original case
                # Store with both original case and lowercase for case-insensitive lookup
                network.whois_responses[username] = True
                network.whois_responses[username.lower()] = True
                logger.info(f"WHOIS: {username} is online")
        
        elif command == '401':  # No such nick/channel
            if len(params) >= 1:
                username = params[1].lower()
                network.whois_responses[username] = False
                logger.info(f"WHOIS: {username} is offline (401)")
        
        elif command == '318':  # End of WHOIS list
            if len(params) >= 1:
                username = params[1].lower()  # Use lowercase for consistent lookup
                if username not in network.whois_responses:
                    network.whois_responses[username] = False
                logger.info(f"WHOIS: End of list for {username}, status: {network.whois_responses.get(username)}")
        
        elif command == 'NOTICE':
            # Check for Yuzu cooldown messages
            if len(params) >= 2:
                # Parse sender from the original message parts[0]
                sender = parts[0].split('!')[0][1:] if parts[0].startswith(':') else parts[0]
                notice_text = ' '.join(params[1:])[1:]  # Remove leading colon
                
                logger.info(f"NOTICE from {sender}: {notice_text}")
                
                if 'yuzu' in sender.lower() and 'change someone\'s rep again in' in notice_text:
                    await self.handle_cooldown_notice(notice_text)
                elif 'nickserv' in sender.lower() and 'password accepted' in notice_text.lower():
                    # NickServ authentication completed, start reputation leveling after delay
                    logger.info("NickServ authentication completed, waiting 1 minute before starting reputation leveling")
                    self.leveling_started = True
                    # Schedule reputation leveling to start in 1 minute
                    asyncio.create_task(self.delayed_start_leveling())
        
        elif command == 'PRIVMSG':
            # Handle channel messages
            if len(params) >= 2:
                # Parse sender from the original message parts[0]
                sender = parts[0].split('!')[0][1:] if parts[0].startswith(':') else parts[0]
                target = params[0]
                message_text = ' '.join(params[1:])[1:]  # Remove leading colon
                await self.handle_privmsg(network, sender, target, message_text)
    
    async def handle_privmsg(self, network: NetworkConnection, sender: str, target: str, message: str):
        """Handle PRIVMSG (channel/user messages)"""
        # Check if this is a private message to the bot
        if target == network.nick:
            # Private message to bot - handle admin commands
            logger.info(f"Received private message from {sender}: {message}")
            admin_list = [admin.strip().lower() for admin in network.config.get('admin', '').split(',')]
            if sender.lower() in admin_list:
                logger.info(f"Processing admin command from {sender}")
                await self.handle_admin_command(network, sender, target, message)
            else:
                logger.info(f"Private message from non-admin {sender}")
        else:
            # Channel message - could add channel-specific handling here if needed
            pass
    
    async def handle_admin_command(self, network: NetworkConnection, sender: str, target: str, message: str):
        """Handle admin commands"""
        logger.info(f"Processing admin command: '{message}' from {sender}")
        parts = message.split()
        if not parts:
            logger.info("No command parts found")
            return
        
        command = parts[0].lower()
        logger.info(f"Command: '{command}', parts: {parts}")
        
        # Handle say command (with or without ! prefix)
        if (command == 'say' or command == '!say') and len(parts) > 1:
            target_channel = parts[1]
            say_message = ' '.join(parts[2:])
            logger.info(f"Executing say command: channel='{target_channel}', message='{say_message}'")
            await network.send_message(target_channel, say_message)
            logger.info(f"Sent message to {target_channel}: {say_message}")
            # Send confirmation back to admin
            await network.send_message(sender, f"Sent to {target_channel}: {say_message}")
        
        elif command == '!join' and len(parts) > 1:
            channel = parts[1]
            await network.send_command(f"JOIN {channel}")
            await network.send_message(sender, f"Joined {channel}")
        
        elif command == '!part' and len(parts) > 1:
            channel = parts[1]
            reason = ' '.join(parts[2:]) if len(parts) > 2 else 'Leaving'
            await network.send_command(f"PART {channel} :{reason}")
            await network.send_message(sender, f"Left {channel}")
        
        elif command == '!restart':
            await network.send_message(sender, "Restarting bot...")
            await self.shutdown()
            self.running = False
        
        elif command == '!status':
            current_time = time.time()
            time_until_next = self.next_leveling_time - current_time
            await network.send_message(sender, f"Next reputation leveling in {int(time_until_next/60)} minutes")
    
    async def handle_cooldown_notice(self, notice_text: str):
        """Handle Yuzu cooldown notice and adjust timing"""
        # Parse the cooldown time from the notice
        # Format: "You can change someone's rep again in X minutes, Y seconds."
        cooldown_match = re.search(r'(\d+) minutes?,?\s*(\d+)?\s*seconds?', notice_text)
        if cooldown_match:
            minutes = int(cooldown_match.group(1))
            seconds = int(cooldown_match.group(2)) if cooldown_match.group(2) else 0
            cooldown_seconds = minutes * 60 + seconds
            
            # Set next leveling time based on cooldown
            self.next_leveling_time = time.time() + cooldown_seconds
            self.save_state()
            logger.info(f"Adjusted next leveling time due to cooldown: {cooldown_seconds} seconds")
    
    async def delayed_start_leveling(self):
        """Wait 1 minute after joining channel before starting reputation leveling"""
        logger.info("Waiting 60 seconds after channel join before starting reputation leveling...")
        await asyncio.sleep(60)  # Wait 1 minute
        logger.info("Starting reputation leveling after delay")
        # Don't call perform_reputation_leveling directly - let the main loop handle it
        self.leveling_started = True
        self.save_state()
    
    async def perform_reputation_leveling(self):
        """Perform reputation leveling - find lowest score user and increment them"""
        if self.leveling_in_progress:
            logger.info("Reputation leveling already in progress, skipping")
            return
        
        self.leveling_in_progress = True
        logger.info("Starting reputation leveling process...")
        
        # Check if bot is registered
        network = None
        for net_name, net in self.networks.items():
            if 'rizon' in net_name.lower() and net.registered:
                network = net
                break
        
        if not network:
            logger.warning("Bot not registered on any network, skipping leveling")
            return
        
        # Parse current reputation scores
        reputation_entries = self.log_parser.parse_reputation_scores()
        if not reputation_entries:
            logger.warning("No reputation entries found, skipping leveling")
            return
        
        # Find the user with the lowest score
        target_entry = reputation_entries[0]  # Already sorted by score
        
        # Check if we can change reputation (cooldown check)
        current_time = time.time()
        if self.last_reputation_change > 0 and current_time - self.last_reputation_change < self.reputation_cooldown:
            time_remaining = self.reputation_cooldown - (current_time - self.last_reputation_change)
            logger.info(f"Reputation cooldown active, {time_remaining:.0f} seconds remaining")
            self.next_leveling_time = current_time + time_remaining
            self.save_state()
            return
        
        
        # Check if target user is online
        logger.info(f"Checking if {target_entry.username} is online...")
        is_online = await network.whois_user(target_entry.username)
        
        if is_online:
            # Send reputation increment message
            increment_message = f"{target_entry.username}++"
            # Send to the first channel in the network's channel list
            channel = list(network.channels)[0] if network.channels else None
            if channel:
                await network.send_message(channel, increment_message)
            
            self.last_reputation_change = current_time
            self.save_state()
            logger.info(f"Sent reputation increment for {target_entry.username} (score: {target_entry.score})")
            
            # Schedule next leveling attempt
            self.next_leveling_time = current_time + self.leveling_interval
            self.save_state()
            logger.info(f"Scheduled next leveling attempt in {self.leveling_interval} seconds")
            
        else:
            logger.info(f"{target_entry.username} is offline, skipping")
            
            # Try next lowest score user
            if len(reputation_entries) > 1:
                next_target = reputation_entries[1]
                logger.info(f"Trying next lowest user: {next_target.username}")
                
                # Check if next user is online
                is_online = await network.whois_user(next_target.username)
                if is_online:
                    increment_message = f"{next_target.username}++"
                    # Send to the first channel in the network's channel list
                    channel = list(network.channels)[0] if network.channels else None
                    if channel:
                        await network.send_message(channel, increment_message)
                    
                    self.last_reputation_change = current_time
                    self.save_state()
                    logger.info(f"Sent reputation increment for {next_target.username} (score: {next_target.score})")
                    
                    # Schedule next leveling attempt
                    self.next_leveling_time = current_time + self.leveling_interval
                    self.save_state()
                    logger.info(f"Scheduled next leveling attempt in {self.leveling_interval} seconds")
                else:
                    logger.info(f"{next_target.username} is also offline, waiting for next cycle")
                    # Schedule next leveling attempt even when no one is online
                    self.next_leveling_time = current_time + self.leveling_interval
                    self.save_state()
                    logger.info(f"Scheduled next leveling attempt in {self.leveling_interval} seconds")
            else:
                logger.info("No other users to try, waiting for next cycle")
                # Schedule next leveling attempt even when no one is online
                self.next_leveling_time = current_time + self.leveling_interval
                self.save_state()
                logger.info(f"Scheduled next leveling attempt in {self.leveling_interval} seconds")
        
        # Reset the flag to allow future leveling attempts
        self.leveling_in_progress = False
    
    async def shutdown(self):
        """Gracefully shutdown the bot"""
        logger.info("Shutting down bot...")
        
        for network in self.networks.values():
            if network.writer:
                await network.send_command("QUIT :GrandLeveler shutting down")
                network.writer.close()
                await network.writer.wait_closed()
        
        logger.info("Bot shutdown complete")

async def main():
    """Main entry point"""
    config_file = "grandleveler.conf"
    
    if not os.path.exists(config_file):
        logger.error(f"Configuration file not found: {config_file}")
        return
    
    bot = GrandLevelerBot(config_file)
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
