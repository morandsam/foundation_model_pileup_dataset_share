#!/bin/bash
echo "=== Script started ==="
cd /afs/cern.ch/user/s/smorand/submitCondor
python3 /afs/cern.ch/user/s/smorand/private/script/JVT_scratch.py "$1" "$2" "$3" "$4"
echo "=== Script finished ==="