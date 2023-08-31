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
import inspect
import pitchnet.dataset
import pitchnet.io
from datetime import datetime

# Configuration
tmp_dir = "__tmp_dtbuilder_sample_based_dataset"
output_filename = "MonophonicSampleBasedDataset.zip"
activate_wav_export = False
current_script_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


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
# Setup directory and look for samples
log("Look for samples in current working directory")
if os.path.isfile("voice_samples.zip"):
    os.system("rm -rf ./voice_samples")
    os.system("unzip voice_samples.zip -d ./voice_samples")
if not os.path.isdir("./voice_samples") and not os.path.isdir("../voice_samples"):
    # Download voice samples
    os.system(f"python3 {current_script_directory}/north_texas_vowel_downloader.py")
    os.system(f"mv ntv_dataset voice_samples")
    os.system(f"{current_script_directory}/wmu_vowel_data.sh voice_samples")
    # Export them into a zip file for next run
    os.system(f"cd ./voice_samples; zip -9 -r ../voice_samples.zip ")

# Select samples directory
if os.path.isdir("../voice_samples"):
    sample_dir = "../voice_samples"
if os.path.isdir("./voice_samples"):
    sample_dir = "./voice_samples"
if os.path.isdir(f"{tmp_dir}/voice_samples"):
    sample_dir = f"{tmp_dir}/voice_samples"

print(f"{datetime.now()} - Selected samples from: {sample_dir}")

# 7s audio dataset based on cut files
dt = pitchnet.dataset.WavSamplerDataset(notes_per_sample=15, size=3000, sample_folder=sample_dir, duration=7, frame_length=1024)

# Export dataset
print(f"{datetime.now()} - Export dataset")
pitchnet.io.export_dataset(dt, f"{tmp_dir}/samplebased_unzipped", verbose=True, skip_exists=True)

# Compress dataset
os.system(f"cd {tmp_dir}/samplebased_unzipped; zip -9 -r ../samplebased.zip .")
os.system(f"mv {tmp_dir}/samplebased.zip {output_filename}")
os.system(f"rm -rf {tmp_dir}")
