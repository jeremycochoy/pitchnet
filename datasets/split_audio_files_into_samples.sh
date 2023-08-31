#!/bin/bash
#
# This script take a directory as argument:
# ./script.sh /my_directory
#
# It brows the directory searching for .wav files, convert them
# to the right 32 bit format, resample them, and finaly cut files
# into small bites of 7 secondes.
# The output is located in $1_split where $1 is the directory
# processed.
#
# It require sox. Remember to `brew install sox` / `apt-get install sox`.
#

if [ -z "$1" ]
then
    echo "Browse a directory of wav files. Convert them to Mono 44.8kHz, and split them in smaller files."
    echo $0 "./audio_directory"
    exit 0
fi

# Remove last slash from argument
dir=${1%/}

# This script change wav files to mono 32 bits sampled at 48kHz
mkdir -p '__'$dir'_converted'
for file in `cd $dir; find . -iname '*.wav'`
do
    echo "Filter ${file}..."
    mkdir -p '__'$dir'_converted/'$(dirname $file)
    sox $dir'/'$file -r 48000 -c 1 -b32 '__'$dir'_converted/'$file
done

# We then split them to short peaces in the split directory
mkdir -p $dir'_split'
find '__'$dir'_converted/' -iname '*.wav' \
 -exec sh -c 'mkdir -p '$dir'_split/$(dirname "{}")' \; \
 -exec sox "{}" $dir'_split/{}' trim 0 7 : newfile : restart \;

rm -rf $dir'_converted'