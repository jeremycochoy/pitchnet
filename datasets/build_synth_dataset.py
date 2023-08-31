#!/usr/bin/env python3
#
# Build the synthetic monophonic dataset
#

# Look for pitchnet and v2p in current directory and parent directory
import sys
sys.path = ['.', '..'] + sys.path

# Dependencies
import os
import shutil
import pitchnet.dataset
import pitchnet.io
from datetime import datetime

# Configuration
tmp_dir = "__tmp_dtbuilder_monophonic_dataset"
output_filename = "MonophonicSynthDataset.zip"
activate_wav_export = False


def log(string):
    print(f"{datetime.now()} - {string}")


# Start the script
print("--- Synth Data Builder ---")

if len(sys.argv) <= 1:
    cmd = "help"
else:
    cmd = sys.argv[1]

if cmd == "clean":
    print(f"rm -rf {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    sys.exit(0)
elif cmd == "help":
    print(f"This script build the {output_filename} dataset.")
    print("./build_synth_dataset.py [help|clean|build]")
    sys.exit(0)
elif cmd != "build":
    print(f"Unknown command {cmd}.")

log(f"Current directory: {os.getcwd()}")
os.makedirs(tmp_dir, exist_ok=True)

# Export the dataset
log("Export dataset...")
pack_directory = "pack_directory"
dt = pitchnet.dataset.MonophonicDataset(
    seed=11, size=6000, notes_per_sample=15, frequency_noise_amplitude=0.1, static_noise_amplitude=0.1, duration=7, frame_length=1024)
pitchnet.io.export_dataset(dt, f"{tmp_dir}/{pack_directory}", verbose=True, skip_exists=True)
if activate_wav_export:
    log("Export wav files...")
    pitchnet.io.export_wav(dt, f"{tmp_dir}/{pack_directory}")

# Compress data
log("Packing dataset...")

os.system(f"cd {tmp_dir}/{pack_directory} && zip -9 -r ../{output_filename} .")
os.system(f"mv {tmp_dir}/{output_filename} {output_filename}")
shutil.rmtree(tmp_dir, ignore_errors=True)
