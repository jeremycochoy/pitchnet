#!/bin/bash
# Generate a unique identifier, build datasets, and upload them on google
# cloud.
#
# This script expect to be run from the root of the directory

# Check we can build a unique identifier
if [ -z $(datasets/compute_unique_identifier.sh) ]
then
  exit 1
fi

# Build the datasets
datasets/build_all.sh

# Upload computed datasets
datasets/upload_datasets.sh
