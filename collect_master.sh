#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

# var for session name (to avoid repeated occurences)
sn=$1

# Start the session and window 0 in /etc
#   This will also be the default cwd for new windows created
#   via a binding unless overridden with default-path.
tmux new-session -s "$sn" -n "ofa" -d

# Create a bunch of windows in /var/log
for i in {1..8}; do tmux new-window -t "$sn:$i" -n "ofa$i"; done

for i in {1..8}; do
    tmux send -t "$sn:$i" "bash collect.sh $((i-1))" ENTER
done
