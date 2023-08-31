# This script is the main script for generating the datasets

rm -f MonophonicOperaDataset.zip SampleBasedDataset.zip MonophonicSynthDataset.zip

BASEDIR=$(dirname $0)

# Build the synthetic dataset based on monophonic synthesizers.
$BASEDIR/build_synth_dataset.py build

# Build opera dataset.
$BASEDIR/build_voice_dataset.py build

# Build the sample based voice dataset.
$BASEDIR/build_sample_based_dataset.py build
