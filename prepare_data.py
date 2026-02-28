import urllib.request
import json

# Last ned NL2Bash-datasettet fra GitHub
url = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm"
urllib.request.urlretrieve(url, "raw_commands.txt")

# Les og rydd opp kommandoene (encoding spesifikt UTF-8)
with open("raw_commands.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

commands = []
for line in lines:
    line = line.strip()
    # Behold bare rimelig korte, enkle kommandoer
    if line and len(line) < 80 and "\n" not in line:
        commands.append(line)

# Skriv til input.txt (ett element per linje, slik microgpt forventer)
with open("input.txt", "w", encoding="utf-8") as f:
    for cmd in commands:
        f.write(cmd + "\n")

print(f"Lagret {len(commands)} bash-kommandoer til input.txt")