#!/usr/bin/env python3
#
# Build datasets used by pitchnet
#
# It require sox. Install with `apt-get install sox` or `brew install sox`.
#

# Look for pitchnet in current directory and parent directory
import sys
sys.path = ['.', '..'] + sys.path

# Dependencies
import os
import inspect
import shutil
import pitchnet.dataset
import pitchnet.io
from datetime import datetime

# Configuration
tmp_dir = "__tmp_dtbuilder_voice_dataset"
output_filename = "MonophonicVoiceDataset.zip"


def log(string):
    print(f"{datetime.now()} - {string}")


def run_and_log(cmd):
    log(cmd)
    os.system(cmd)


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
    print("./build_voice_dataset.py [help|clean|build]")
    sys.exit(0)
elif cmd != "build":
    print(f"Unknown command {cmd}.")

log(f"Current directory: {os.getcwd()}")
os.makedirs(tmp_dir, exist_ok=True)

# Download datasets of singers
log("Download opera dataset")
os.makedirs(tmp_dir, exist_ok=True)
if not os.path.isfile("SingingDatabase.zip"):
    os.system(f"curl -OL http://www.isophonics.net/sites/isophonics.net/files/SingingDatabase.zip")
os.system(f"unzip SingingDatabase.zip -d  {tmp_dir}/SingingDatabase")

# Run wav format conversion
log("Run conversion")
run_and_log(f"mv {tmp_dir}/SingingDatabase/monophonic {tmp_dir}/voice_audio")
# Add manually added files
run_and_log(f"cp -r voice_manual_pack {tmp_dir}/voice_audio")

# Split in smaller files
current_script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
run_and_log(f"cd {tmp_dir}; {current_script_directory}/split_audio_files_into_samples.sh voice_audio")
voice_dataset_path = f"{tmp_dir}/voice_audio_split"

# Run preprocess to audio
log("Export audio")

# 7s audio dataset based on cut files
dt = pitchnet.dataset.WavFolderDataset(voice_dataset_path, duration=7, frame_length=1024)

# Export dataset
log("Export dataset")
pitchnet.io.export_dataset(dt, f"{tmp_dir}/voice_unziped", verbose=True, skip_exists=True)

# Zip it
log("Zip dataset")
os.system(f"rm -f {output_filename}.zip")
os.system(f"cd {tmp_dir}/voice_unziped; zip -9 -r ../{output_filename} .")
os.system(f"mv {tmp_dir}/{output_filename} ./")

# Remove working directory
shutil.rmtree(tmp_dir, ignore_errors=True)
