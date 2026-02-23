"""
Find neurons whose simulation.json was NOT updated after 5pm EST on Sunday Feb 22, 2026.
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Cutoff: 5pm EST on Feb 22, 2026
# EST = UTC-5, so 5pm EST = 22:00 UTC
CUTOFF_EPOCH = 1771797600  # date -j -f "%Y-%m-%d %H:%M:%S" "2026-02-22 17:00:00" +%s (EST)

EXPLANATIONS_DIR = Path("explanations")

# All neurons we're tracking (from activation_collector/config.py)
from activation_collector.config import NEURONS

all_neurons = set(NEURONS)

# Find neurons with simulation files updated after the cutoff
updated_neurons = set()
for sim_file in EXPLANATIONS_DIR.glob("*_simulation.json"):
    mtime = sim_file.stat().st_mtime
    if mtime > CUTOFF_EPOCH:
        neuron_id = sim_file.name.replace("_simulation.json", "")
        updated_neurons.add(neuron_id)

# Neurons with no simulation file at all, or file not updated after cutoff
missing_neurons = sorted(all_neurons - updated_neurons)

print(f"Total neurons:   {len(all_neurons)}")
print(f"Updated (>5pm):  {len(updated_neurons)}")
print(f"NOT updated:     {len(missing_neurons)}")
print()
print("MISSING_NEURONS = [")
for nid in missing_neurons:
    print(f'    "{nid}",')
print("]")
