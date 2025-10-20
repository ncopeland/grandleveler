Grand Leveler is a new project
I want to make an IRC bot that works a lot like my duckhunt bot in /home/boliver/Projects/duckhunt 
Multithreaded python that uses asyncio to connect to IRC

But this IRC bot will not serve a game of duckhunt
it will just be a bot with the purpose to adjust peoples reputation score on the rizon IRC network channel #computertech
The bot should use a config file for network and channel information, name, nick, ident, server/port, channel(s). 
Copy my DuckHuntBot config file for rizon information, but this bot will be named gleveler in ident/username, and its irc nick will be GrandLeveler or GL or GrandL or Leveler (altnicks). 
I want it to perform like the duckhunt bot does.

I want this IRC bot to have basic IRC admin functionality- the join part channel commands, say command, restart



Every 70 mins I want it to do this, because you can only change a person's reputation score once every 1 hour.

Much like the script in /var/www/devforge/cgi-bin/reputation_stats.py
I want this script to parse 
/home/boliver/.local/share/weechat/logs/irc.rizon.#computertech.weechatlog
and just make a list of the  Current Reputation Scores

peorth 359
h4 193
wez 131
updog 49
basti 45
antitheus 13
boliver 11
deadly 10
madafaka 8
milambar 8
mikoolo 7
naemuti 6
wildting2 5
cba 4
rizon 4
sander 4
elcasey 3
dragon 3
cherimoya 3
admiral_hubris 3
papafrog 3
magic 3
wh 3
phonekait 2
p1x4l 2
fensta 2
socraticdev 2
girafe2 2
wobotkoala 2
waz 1
hotties 1
mcflying 1
iwof 1
suzubrah 1
hks 1
raya 1
knownsyntax 1
magicalpig 1
lambdacomplex 1
techy 0
mysteria 0
cloudlan -1
colby -2
fifel -3
alterego_ -3
loulan -22
tabb -28
url -30
alephium -31
duckhunt -83
kaitphone -104
computertech -424

I want you to incrase the worst reputation person until everyone is at the same score, and if two people are tied for last place just pick the one that comes first in the list. This should happen until everyone has a score of the highest scoring person. In this case peorth with 359, but this number will always change.

So, then, whoever is in this list has the worst reputation score - in this case it is computertech with a score of -424
I want you to do a /whois on the person to verify that they are online
If they are online, you will get a response that might look like this

19:38:41	--	[ComputerTech] (ComputerTe@ComputerTech.Network): ComputerTech
19:38:41	--	[ComputerTech] ~#computertech @#homescreen +#politics-uncensored ~#Canada &#premium +#RizonIRPG
19:38:41	--	[ComputerTech] *.rizon.net (Where are you?)
19:38:41	--	[ComputerTech] is an IRC Operator - Server Administrator
19:38:41	--	[ComputerTech] has identified for this nick
19:38:41	--	[ComputerTech] is using a secure connection
19:38:41	--	[computertech] End of /WHOIS list.


I want you to send a message to the channel that says "computertech++" using their username and Yuzu will reply with a score increased message.
If they are not online, skip them and go to the person with the next lowest reputation score.

If they are not online, you might get a response that will look like this:
19:38:51	--	phonekait: No such nick/channel
19:38:51	--	[phonekait] End of /WHOIS list.



When the bot will first join IRC and join the rizon #computertech channel, try a reputation_leveling;
find the target to change and try to change their score.
If you aren't able to change their score, you will get an IRC notice from Yuzu
Yuzu (~yuzu@oh.Kaede.my.Kaede): You can change someone's rep again in 42 minutes, 48 seconds.
Schedule the next reputation_leveling after the amount of time Yuzu tells you you have to wait, and again every 70 minutes after that. 

After you send the reputation changing message to the channel, check to see if Yuzu sent you a notice that you have to wait and wait the time

So this script will improve the reputation of computer tech every 70 minutes until it gets to kaitphone and then it will level kaitphone then computertech then it will get to duckhunt and do duckhunt, then kaitphone, then computertech, and after they're all the same it will do alephium, duckhunt, kaitphone, and computer tech each once every 70 minutes... Get it? until everyone is tied. Then it will change nobody's reputation. And it only does the online people. 

If you're going through the list and all the lowest rep people are not online, wait at least 1 second before WHOISing the next person. You don't want to flood the IRC server. In general there should be 1s flood control. 

