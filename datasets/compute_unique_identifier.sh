#!/bin/sh
# Compute a unique identifier based on code version and input data version
#

BASEDIR=$(dirname $0)

COMMIT_ID=$($BASEDIR"/compute_unique_commit_identifier.sh")
if [ $? != 0 ]
then
  echo "ERROR Unique identifier halted" 1>&2
  exit 1
fi
DATA_ID=$($BASEDIR"/compute_unique_files_identifier.sh" voice_samples.zip SingingDatabase.zip)
if [ $? != 0 ]
then
  echo "ERROR Unique identifier halted"  1>&2
  exit 1
fi
DATE_ID=$(date "+%Y%m%d")

echo $DATE_ID-$COMMIT_ID"-"$DATA_ID
