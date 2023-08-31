#!/bin/sh
# This script upload the datasets computed to google cloud.
#
# It expect that gsutil is installed, and that the machine
# has its service account configured.

# Make a unique identifier
unique_identifier=$(datasets/compute_unique_identifier.sh)
if [ $? != 0 ]
then
  exit 1
fi

mv MonophonicSynthDataset.zip MonophonicSynthDataset-$unique_identifier.zip
mv MonophonicVoiceDataset.zip MonophonicVoiceDataset-$unique_identifier.zip
mv MonophonicSampleBasedDataset.zip MonophonicSampleBasedDataset-$unique_identifier.zip

# Copy dataset to google cloud storage
gsutil cp *-$unique_identifier.zip gs://datasets.pitchnet.net/builds/
