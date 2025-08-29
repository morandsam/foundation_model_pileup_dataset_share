#!/bin/bash
echo "=== Script started ==="
cd /afs/cern.ch/user/s/smorand/submitCondor
python3 /afs/cern.ch/user/s/smorand/private/script/MAE.py "$1" "$2" "$3"
echo "=== Script finished ==="